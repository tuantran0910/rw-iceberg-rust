// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Catalog-agnostic upsert bindings for PyIceberg.
//!
//! The public entry point is [`py_upsert`], which:
//!   1. Builds a `Table` from metadata JSON (no catalog needed).
//!   2. Runs the Arrow-based upsert compute: reads existing files, matches source rows,
//!      writes new data/equality-delete files to storage.
//!   3. Returns the file delta (`files_added`, `files_removed`) and row counts.
//!
//! The Python caller is responsible for committing the delta to whichever catalog
//! the table belongs to (REST, Glue, Hive, DynamoDB, SQL, BigQuery, …).

use std::collections::HashMap;
use std::sync::Arc;

use arrow::datatypes::{Field as ArrowField, Schema as ArrowSchema};
use arrow::pyarrow::FromPyArrow;
use arrow::record_batch::RecordBatch;
use iceberg::io::FileIOBuilder;
use iceberg::spec::{Schema as IcebergSchema, TableMetadata};
use iceberg::table::Table;
use iceberg::upsert::{UpsertConfig, UpsertFileDelta, UpsertWriteMode};
use iceberg::{Error, ErrorKind, TableIdent};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::data_file::PyPrimitiveLiteral;
use crate::error::to_py_err;
use crate::runtime::runtime;

/// Metadata for a single data file produced or consumed by an upsert operation.
///
/// Returned by [`py_upsert`] so that Python can commit the file delta
/// to any catalog using PyIceberg's native transaction API.
#[pyclass(name = "DataFileInfo")]
#[derive(Clone)]
pub struct PyDataFileInfo {
    /// Content type: 0 = DATA, 1 = POSITION_DELETES, 2 = EQUALITY_DELETES.
    #[pyo3(get)]
    pub content: i32,
    /// Full URI with FS scheme (e.g. ``s3://bucket/path/file.parquet``).
    #[pyo3(get)]
    pub file_path: String,
    /// File format name: ``"PARQUET"``, ``"ORC"``, or ``"AVRO"``.
    #[pyo3(get)]
    pub file_format: String,
    #[pyo3(get)]
    pub record_count: u64,
    #[pyo3(get)]
    pub file_size_in_bytes: u64,
    /// Map from column field-id to total on-disk size.
    #[pyo3(get)]
    pub column_sizes: HashMap<i32, u64>,
    /// Map from column field-id to total value count (including nulls/NaNs).
    #[pyo3(get)]
    pub value_counts: HashMap<i32, u64>,
    /// Map from column field-id to null value count.
    #[pyo3(get)]
    pub null_value_counts: HashMap<i32, u64>,
    /// Map from column field-id to NaN value count.
    #[pyo3(get)]
    pub nan_value_counts: HashMap<i32, u64>,
    /// Field IDs used to determine row equality (equality-delete files only).
    #[pyo3(get)]
    pub equality_ids: Option<Vec<i32>>,
    #[pyo3(get)]
    pub sort_order_id: Option<i32>,
    #[pyo3(get)]
    pub partition_spec_id: i32,
    /// Partition values as a list of optional primitives.
    /// Use ``partition_spec`` and ``table_metadata.schema()`` to reconstruct a ``Record``.
    #[pyo3(get)]
    pub partition: Vec<Option<PyPrimitiveLiteral>>,
}

#[pymethods]
impl PyDataFileInfo {
    fn __repr__(&self) -> String {
        format!(
            "DataFileInfo(content={}, file_path={:?}, record_count={})",
            self.content, self.file_path, self.record_count
        )
    }
}

/// Result of a catalog-agnostic upsert compute operation.
///
/// Returned by [`py_upsert`]. The caller is responsible for committing the
/// file delta to the catalog using PyIceberg's transaction API.
#[pyclass(name = "UpsertFilesResult")]
#[derive(Clone)]
pub struct PyUpsertFilesResult {
    /// New data files written to storage by Rust (to be added in the catalog commit).
    #[pyo3(get)]
    pub files_added: Vec<PyDataFileInfo>,
    /// Existing data files that were replaced (CoW only; empty for MoR).
    #[pyo3(get)]
    pub files_removed: Vec<PyDataFileInfo>,
    #[pyo3(get)]
    pub rows_updated: u64,
    #[pyo3(get)]
    pub rows_inserted: u64,
    /// Files rewritten (CoW) or equality-delete files created (MoR).
    #[pyo3(get)]
    pub files_affected: u64,
}

#[pymethods]
impl PyUpsertFilesResult {
    fn __repr__(&self) -> String {
        format!(
            "UpsertFilesResult(rows_updated={}, rows_inserted={}, \
             files_added={}, files_removed={}, files_affected={})",
            self.rows_updated,
            self.rows_inserted,
            self.files_added.len(),
            self.files_removed.len(),
            self.files_affected,
        )
    }
}

/// Convert an iceberg-rust `DataFile` to the Python-facing `PyDataFileInfo`.
fn data_file_to_py(df: &iceberg::spec::DataFile) -> PyDataFileInfo {
    PyDataFileInfo {
        content: df.content_type() as i32,
        file_path: df.file_path().to_string(),
        file_format: format!("{:?}", df.file_format()).to_uppercase(),
        record_count: df.record_count(),
        file_size_in_bytes: df.file_size_in_bytes(),
        column_sizes: df.column_sizes().clone(),
        value_counts: df.value_counts().clone(),
        null_value_counts: df.null_value_counts().clone(),
        nan_value_counts: df.nan_value_counts().clone(),
        equality_ids: df.equality_ids(),
        sort_order_id: df.sort_order_id(),
        partition_spec_id: df.partition_spec_id(),
        partition: df
            .partition()
            .iter()
            .map(|lit| {
                lit.and_then(|l| {
                    Some(PyPrimitiveLiteral {
                        inner: l.as_primitive_literal()?,
                    })
                })
            })
            .collect(),
    }
}

/// Convert a Python `pa.RecordBatch` to a Rust `RecordBatch`.
///
/// Python must call `pa.concat_batches(df.to_batches())` before passing here
/// if the input is a `pa.Table` with multiple batches.
fn pyarrow_to_record_batch(source: &Bound<'_, PyAny>) -> PyResult<RecordBatch> {
    RecordBatch::from_pyarrow_bound(source).map_err(|e| {
        PyValueError::new_err(format!(
            "Failed to convert source_batch from PyArrow. \
             Ensure you pass a pa.RecordBatch (for pa.Table, use pa.concat_batches(df.to_batches())): {e}"
        ))
    })
}

/// Enrich the source batch's Arrow schema with `PARQUET:field_id` metadata by
/// looking up each column name in the Iceberg table schema.
///
/// This is a **no-op** when all fields already carry field ID metadata, so there
/// is no cost for callers that already annotate their schemas correctly.
///
/// When IDs are absent the function performs a case-sensitive name lookup against
/// the Iceberg schema's top-level fields.  If a column name cannot be found an
/// error is returned with a message that tells the caller exactly which column
/// caused the problem, so they can either rename it or add the metadata manually.
///
/// # Why this is needed
///
/// The Rust upsert pipeline uses field IDs — not column positions or names — to
/// map Arrow columns to Iceberg fields.  This mirrors the Iceberg spec requirement
/// for column projection (§ "Column Projection") and matches how the Parquet
/// reader handles files that were written without embedded field IDs.  Python
/// users who build `pa.RecordBatch` objects from plain NumPy arrays or Pandas
/// DataFrames typically don't add that metadata, so we inject it here at the
/// binding boundary rather than forcing every caller to do it manually.
fn enrich_batch_with_field_ids(
    batch: RecordBatch,
    iceberg_schema: &IcebergSchema,
) -> Result<RecordBatch, Error> {
    const PARQUET_FIELD_ID_META_KEY: &str = "PARQUET:field_id";

    // Fast path: all fields already have IDs — nothing to do.
    let needs_enrichment = batch
        .schema()
        .fields()
        .iter()
        .any(|f| f.metadata().get(PARQUET_FIELD_ID_META_KEY).is_none());

    if !needs_enrichment {
        return Ok(batch);
    }

    // Slow path: inject missing field IDs via name lookup.
    let enriched_fields: Vec<ArrowField> = batch
        .schema()
        .fields()
        .iter()
        .map(|arrow_field| {
            // Keep fields that already carry an ID unchanged.
            if arrow_field.metadata().get(PARQUET_FIELD_ID_META_KEY).is_some() {
                return Ok(arrow_field.as_ref().clone());
            }

            // Look up the field ID by column name in the Iceberg schema.
            let field_id = iceberg_schema
                .field_id_by_name(arrow_field.name())
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Column '{}' not found in the Iceberg table schema. \
                             Either add PARQUET:field_id metadata to the Arrow field \
                             or ensure the column name matches a field in the table schema.",
                            arrow_field.name()
                        ),
                    )
                })?;

            let mut metadata = arrow_field.metadata().clone();
            metadata.insert(PARQUET_FIELD_ID_META_KEY.to_string(), field_id.to_string());

            Ok(ArrowField::new(
                arrow_field.name(),
                arrow_field.data_type().clone(),
                arrow_field.is_nullable(),
            )
            .with_metadata(metadata))
        })
        .collect::<Result<_, Error>>()?;

    let enriched_schema = Arc::new(ArrowSchema::new_with_metadata(
        enriched_fields,
        batch.schema().metadata().clone(),
    ));

    RecordBatch::try_new(enriched_schema, batch.columns().to_vec()).map_err(|e| {
        Error::new(
            ErrorKind::Unexpected,
            format!("Failed to rebuild RecordBatch with enriched schema: {e}"),
        )
    })
}

/// Compute a high-performance Rust upsert without committing to any catalog.
///
/// This function is **catalog-agnostic**: it builds the table in-memory from
/// the supplied metadata JSON and file I/O properties, runs the upsert compute
/// (reads existing files, matches rows, writes new files to storage), and returns
/// the resulting file delta.  The **caller is responsible for committing** the
/// delta to the catalog using PyIceberg's native transaction API — which works
/// with any catalog type (REST, Glue, Hive, DynamoDB, SQL, BigQuery, etc.).
///
/// Parameters
/// ----------
/// metadata_json:
///     JSON string of the table metadata. Pass ``self.metadata.model_dump_json()``.
/// file_io_properties:
///     I/O properties dict for storage access. Pass ``dict(self.io.properties)``.
/// metadata_location:
///     Full metadata file URI, e.g. ``"s3://bucket/warehouse/ns/table/metadata/..."``.
///     Used to derive the storage scheme (``"s3"``, ``"gs"``, etc.).
/// table_identifier:
///     Fully-qualified table identifier as a list of strings, e.g. ``["ns", "table"]``.
/// source_batch:
///     A ``pa.RecordBatch`` or ``pa.Table`` containing the upsert source data.
///     Both types are accepted; if a ``pa.Table`` is passed it is automatically
///     converted via ``table.to_batches()[0]``.
/// join_columns:
///     Key column names for matching. If ``None``, uses ``identifier_field_ids``.
/// skip_unchanged:
///     CoW only. When ``True`` (default), skips rewriting files whose matched rows
///     have identical non-key column values.
/// write_mode:
///     ``"copy-on-write"`` (default) or ``"merge-on-read"``.
///
/// Returns
/// -------
/// UpsertFilesResult
///     Contains ``files_added``, ``files_removed`` (lists of ``DataFileInfo``),
///     plus ``rows_updated``, ``rows_inserted``, ``files_affected`` counters.
#[pyfunction]
#[pyo3(signature = (
    metadata_json,
    file_io_properties,
    metadata_location,
    table_identifier,
    source_batch,
    join_columns=None,
    skip_unchanged=true,
    write_mode="copy-on-write"
))]
#[allow(clippy::too_many_arguments)]
pub fn py_upsert(
    _py: Python<'_>,
    metadata_json: String,
    file_io_properties: HashMap<String, String>,
    metadata_location: String,
    table_identifier: Vec<String>,
    source_batch: &Bound<'_, PyAny>,
    join_columns: Option<Vec<String>>,
    skip_unchanged: bool,
    write_mode: &str,
) -> PyResult<PyUpsertFilesResult> {
    let batch = pyarrow_to_record_batch(source_batch)?;

    let write_mode_enum = UpsertWriteMode::try_from(write_mode).map_err(to_py_err)?;

    let config = UpsertConfig {
        join_columns: join_columns.unwrap_or_default(),
        skip_unchanged,
        write_mode: write_mode_enum,
        pruning_predicate: None,
    };

    let delta: UpsertFileDelta = runtime().block_on(async {
        // Derive the storage scheme from the metadata location URI prefix.
        // E.g. "s3://..." → "s3",  "gs://..." → "gs",  "/local/..." → "file".
        let scheme = metadata_location
            .split("://")
            .next()
            .filter(|s| s.len() > 1 && !s.contains('/'))
            .unwrap_or("file");

        let file_io = FileIOBuilder::new(scheme)
            .with_props(file_io_properties)
            .build()
            .map_err(to_py_err)?;

        let table_metadata: TableMetadata = serde_json::from_str(&metadata_json)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to parse table metadata: {e}")))?;

        let ident = TableIdent::from_strs(table_identifier)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid table identifier: {e}")))?;

        let table = Table::builder()
            .identifier(ident)
            .metadata(Arc::new(table_metadata))
            .metadata_location(metadata_location)
            .file_io(file_io)
            .build()
            .map_err(to_py_err)?;

        let batch = enrich_batch_with_field_ids(batch, table.metadata().current_schema())
            .map_err(to_py_err)?;

        iceberg::upsert::upsert_compute(&table, batch, config)
            .await
            .map_err(to_py_err)
    })?;

    Ok(PyUpsertFilesResult {
        files_added: delta.added_data_files.iter().map(data_file_to_py).collect(),
        files_removed: delta
            .deleted_data_files
            .iter()
            .map(data_file_to_py)
            .collect(),
        rows_updated: delta.stats.rows_updated,
        rows_inserted: delta.stats.rows_inserted,
        files_affected: delta.stats.files_affected,
    })
}

pub fn register_module(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let upsert_module = PyModule::new(py, "upsert")?;
    upsert_module.add_class::<PyDataFileInfo>()?;
    upsert_module.add_class::<PyUpsertFilesResult>()?;
    upsert_module.add_function(wrap_pyfunction!(py_upsert, &upsert_module)?)?;
    parent.add_submodule(&upsert_module)?;
    // Register as a top-level importable submodule so
    // `from pyiceberg_core.upsert import py_upsert` works.
    py.import("sys")?
        .getattr("modules")?
        .set_item("pyiceberg_core.upsert", &upsert_module)?;
    Ok(())
}

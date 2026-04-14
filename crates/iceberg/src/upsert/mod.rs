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

//! Upsert support: high-performance Copy-on-Write (CoW) and Merge-on-Read (MoR)
//! strategies built on Arrow columnar compute.

pub mod cow_executor;
pub mod matcher;
pub mod mor_executor;
pub mod planner;

use arrow_array::RecordBatch;

use crate::Result;
use crate::catalog::Catalog;
use crate::expr::Predicate;
use crate::spec::DataFile;
use crate::table::Table;

/// Write strategy for upsert operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpsertWriteMode {
    /// Copy-on-Write: rewrite affected data files with merged content.
    /// Best for read-heavy workloads or when updates are sparse.
    CopyOnWrite,
    /// Merge-on-Read: write equality-delete files and new data files without
    /// rewriting existing files.  Best for write-heavy workloads or dense updates.
    MergeOnRead,
}

impl Default for UpsertWriteMode {
    fn default() -> Self {
        Self::CopyOnWrite
    }
}

impl TryFrom<&str> for UpsertWriteMode {
    type Error = crate::Error;

    fn try_from(s: &str) -> std::result::Result<Self, Self::Error> {
        match s.to_lowercase().as_str() {
            "copy-on-write" | "cow" => Ok(Self::CopyOnWrite),
            "merge-on-read" | "mor" => Ok(Self::MergeOnRead),
            _ => Err(crate::Error::new(
                crate::ErrorKind::DataInvalid,
                format!(
                    "Unknown upsert write mode: '{s}'. Expected 'copy-on-write' or 'merge-on-read'."
                ),
            )),
        }
    }
}

/// Configuration for an upsert operation.
#[derive(Debug, Clone)]
pub struct UpsertConfig {
    /// Explicit join key column names.  If empty, the table's
    /// `schema.identifier_field_ids()` are used.
    pub join_columns: Vec<String>,

    /// When `true` (the default), CoW rewrites skip files where all matched
    /// rows have unchanged non-key columns.  MoR always writes new data + delete
    /// files regardless of whether values changed.
    pub skip_unchanged: bool,

    /// Write strategy.
    pub write_mode: UpsertWriteMode,

    /// Optional bounding-box predicate used to prune files during scanning.
    /// Built automatically from source key ranges when absent.
    pub pruning_predicate: Option<Predicate>,
}

impl Default for UpsertConfig {
    fn default() -> Self {
        Self {
            join_columns: Vec::new(),
            skip_unchanged: true,
            write_mode: UpsertWriteMode::CopyOnWrite,
            pruning_predicate: None,
        }
    }
}

/// Result of an upsert operation.
#[derive(Debug, Clone)]
pub struct UpsertResult {
    /// Number of target rows that were updated (matched and changed).
    pub rows_updated: u64,
    /// Number of source rows inserted as new data (never matched any target row).
    pub rows_inserted: u64,
    /// Number of data files rewritten (CoW) or equality-delete files created (MoR).
    pub files_affected: u64,
    /// Number of data files added (new inserts + all rewrites for CoW).
    pub files_added: u64,
    /// Number of data files deleted (CoW only).
    pub files_removed: u64,
}

/// File delta produced by [`upsert_compute`]: files written to storage + files to delete,
/// without a catalog commit.
///
/// The caller is responsible for committing this delta to the catalog using any
/// transaction mechanism (e.g., `OverwriteFilesAction` in Rust, or the PyIceberg
/// transaction API in Python).
#[derive(Debug)]
pub struct UpsertFileDelta {
    /// New data files written to storage (rewrites + inserts for CoW; new data files
    /// + equality-delete files for MoR).
    pub added_data_files: Vec<DataFile>,
    /// Original data files to mark as deleted in the catalog (CoW only; empty for MoR).
    pub deleted_data_files: Vec<DataFile>,
    /// Operation statistics.
    pub stats: UpsertResult,
}

/// Execute an upsert operation on `table` using `source` as the upsert data.
///
/// The operation uses the Arrow-based [`matcher::UpsertMatcher`] for vectorized
/// hash-joins and columnar change detection.  Files are scanned via the existing
/// [`TableScan`](crate::scan::TableScan) infrastructure and filtered using a
/// bounding-box predicate constructed from source key ranges.
///
/// # Arguments
/// * `table`     — the target Iceberg table
/// * `catalog`   — catalog used to commit the transaction (with retry on conflict)
/// * `source`    — a single Arrow `RecordBatch` containing the upsert data
/// * `config`    — operation configuration
///
/// # Example
/// ```ignore
/// let (table, result) = upsert(
///     &table,
///     &rest_catalog,
///     source_batch,
///     UpsertConfig::default(),
/// )
/// .await?;
/// ```
pub async fn upsert(
    table: &Table,
    catalog: &dyn Catalog,
    source: RecordBatch,
    config: UpsertConfig,
) -> Result<(Table, UpsertResult)> {
    // 1. Resolve key columns from schema / explicit config
    let schema = table.metadata().current_schema();
    let key_columns = planner::resolve_key_columns(schema, &config.join_columns)?;

    // 2. Validate source has no duplicate key values
    let key_indices: Vec<usize> = key_columns.iter().map(|k| k.schema_index).collect();
    planner::validate_source_keys(&source, &key_indices)?;

    // 3. Build (or use provided) pruning predicate (IN-list for ≤1000 keys, bounding-box
    //    otherwise).  Always covers the full source key range to correctly handle mixed
    //    update/insert workloads — the insert-optimized predicate was incorrect because it
    //    couldn't distinguish updates from inserts without pre-scanning the existing table.
    let predicate = match config.pruning_predicate {
        Some(p) => p,
        None => planner::build_optimal_predicate(&source, &key_columns)?.ok_or_else(|| {
            crate::Error::new(
                crate::ErrorKind::DataInvalid,
                "Source batch is empty; nothing to upsert",
            )
        })?,
    };

    // 4. Resolve non-key column indices for change detection
    let non_key_indices = planner::non_key_column_indices(schema, &key_columns)?;

    // 5. Dispatch to CoW or MoR executor
    match config.write_mode {
        UpsertWriteMode::CopyOnWrite => {
            cow_executor::execute(
                table,
                catalog,
                source,
                &key_columns,
                &non_key_indices,
                &predicate,
                config.skip_unchanged,
            )
            .await
        }
        UpsertWriteMode::MergeOnRead => {
            mor_executor::execute(
                table,
                catalog,
                source,
                &key_columns,
                &non_key_indices,
                &predicate,
            )
            .await
        }
    }
}

/// Compute the upsert file delta without committing to any catalog.
///
/// Unlike [`upsert`], this function does **not** interact with a catalog at all.
/// The caller must:
///   1. Pass a **fresh** `Table` loaded from the catalog immediately before calling
///      this function (so the scan sees all committed files).
///   2. Commit the returned [`UpsertFileDelta`] to the catalog using any transaction
///      mechanism appropriate for the caller's environment.
///
/// This is the building block for catalog-agnostic language bindings such as the
/// PyIceberg integration, where Python handles the catalog commit.
pub async fn upsert_compute(
    table: &Table,
    source: RecordBatch,
    config: UpsertConfig,
) -> Result<UpsertFileDelta> {
    let schema = table.metadata().current_schema();
    let key_columns = planner::resolve_key_columns(schema, &config.join_columns)?;
    let key_indices: Vec<usize> = key_columns.iter().map(|k| k.schema_index).collect();
    planner::validate_source_keys(&source, &key_indices)?;

    let predicate = match config.pruning_predicate {
        Some(p) => p,
        None => planner::build_optimal_predicate(&source, &key_columns)?.ok_or_else(|| {
            crate::Error::new(
                crate::ErrorKind::DataInvalid,
                "Source batch is empty; nothing to upsert",
            )
        })?,
    };

    let non_key_indices = planner::non_key_column_indices(schema, &key_columns)?;

    match config.write_mode {
        UpsertWriteMode::CopyOnWrite => {
            cow_executor::compute(
                table,
                source,
                &key_columns,
                &non_key_indices,
                &predicate,
                config.skip_unchanged,
            )
            .await
        }
        UpsertWriteMode::MergeOnRead => {
            mor_executor::compute(table, source, &key_columns, &non_key_indices, &predicate).await
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_array::{Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use futures::TryStreamExt;
    use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
    use parquet::file::properties::WriterProperties;
    use tempfile::TempDir;

    use super::*;
    use crate::catalog::memory::MEMORY_CATALOG_WAREHOUSE;
    use crate::catalog::{CatalogBuilder, MemoryCatalog};
    use crate::memory::MemoryCatalogBuilder;
    use crate::spec::{DataFileFormat, NestedField, PrimitiveType, Schema, Type};
    use crate::transaction::{ApplyTransactionAction, Transaction};
    use crate::writer::base_writer::data_file_writer::DataFileWriterBuilder;
    use crate::writer::file_writer::ParquetWriterBuilder;
    use crate::writer::file_writer::location_generator::{
        DefaultFileNameGenerator, DefaultLocationGenerator,
    };
    use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
    use crate::writer::{IcebergWriter, IcebergWriterBuilder};
    use crate::{Catalog, NamespaceIdent, TableCreation};

    // ── Schema helpers ────────────────────────────────────────────────────────

    /// Iceberg schema: id (Long, required, identifier), name (String, required),
    /// value (String, optional).
    fn test_iceberg_schema() -> Schema {
        Schema::builder()
            .with_schema_id(1)
            .with_identifier_field_ids(vec![1])
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::optional(3, "value", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap()
    }

    /// Arrow schema matching the Iceberg schema above, with Iceberg field IDs
    /// embedded in metadata so the parquet writer stores them in the file footer.
    fn test_arrow_schema() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
            Field::new("name", DataType::Utf8, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "2".to_string(),
            )])),
            Field::new("value", DataType::Utf8, true).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "3".to_string(),
            )])),
        ]))
    }

    fn make_batch(ids: Vec<i64>, names: Vec<&str>, values: Vec<Option<&str>>) -> RecordBatch {
        RecordBatch::try_new(test_arrow_schema(), vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(StringArray::from(names)),
            Arc::new(StringArray::from(values)),
        ])
        .unwrap()
    }

    // ── Catalog / table setup helpers ─────────────────────────────────────────

    /// Creates a MemoryCatalog backed by a real temp directory and returns the
    /// `TempDir` handle so the caller keeps it alive for the test duration.
    async fn setup_catalog() -> (MemoryCatalog, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let warehouse = temp_dir.path().to_str().unwrap().to_string();
        let catalog = MemoryCatalogBuilder::default()
            .load(
                "memory",
                HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse)]),
            )
            .await
            .unwrap();
        (catalog, temp_dir)
    }

    /// Creates a namespace + table inside `catalog` using `test_iceberg_schema()`.
    async fn setup_table(catalog: &impl Catalog) -> Table {
        let ns = NamespaceIdent::from_strs(["test_ns"]).unwrap();
        catalog.create_namespace(&ns, HashMap::new()).await.unwrap();

        catalog
            .create_table(
                &ns,
                TableCreation::builder()
                    .name("upsert_test".to_string())
                    .schema(test_iceberg_schema())
                    .build(),
            )
            .await
            .unwrap()
    }

    /// Writes `batch` as a real Parquet data file, commits it to `table` via
    /// `fast_append`, and returns the refreshed `Table`.
    async fn commit_batch(table: &Table, catalog: &dyn Catalog, batch: RecordBatch) -> Table {
        let file_io = table.file_io().clone();
        let metadata = table.metadata();

        let location_gen = DefaultLocationGenerator::new(metadata.clone()).unwrap();
        let file_name_gen = DefaultFileNameGenerator::new(
            "test-initial".to_string(),
            Some(uuid::Uuid::now_v7().to_string()),
            DataFileFormat::Parquet,
        );
        let parquet_builder = ParquetWriterBuilder::new(
            WriterProperties::default(),
            metadata.current_schema().clone(),
        );
        let rolling_builder = RollingFileWriterBuilder::new_with_default_file_size(
            parquet_builder,
            file_io,
            location_gen,
            file_name_gen,
        );
        let mut writer = DataFileWriterBuilder::new(rolling_builder)
            .build(None)
            .await
            .unwrap();
        writer.write(batch).await.unwrap();
        let data_files = writer.close().await.unwrap();

        let tx = Transaction::new(table);
        let action = tx.fast_append().add_data_files(data_files);
        let tx = action.apply(tx).unwrap();
        tx.commit(catalog).await.unwrap()
    }

    /// Scans all rows from `table` and returns them as `RecordBatch`es.
    async fn scan_all(table: &Table) -> Vec<RecordBatch> {
        table
            .scan()
            .build()
            .unwrap()
            .to_arrow()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap()
    }

    fn total_rows(batches: &[RecordBatch]) -> usize {
        batches.iter().map(|b| b.num_rows()).sum()
    }

    /// Extracts all `id` values from scanned batches, sorted ascending.
    fn sorted_ids(batches: &[RecordBatch]) -> Vec<i64> {
        let mut ids: Vec<i64> = batches
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .iter()
                    .flatten()
            })
            .collect();
        ids.sort_unstable();
        ids
    }

    /// Returns the `name` value for the row with the given `id`, if found.
    fn find_name(batches: &[RecordBatch], id: i64) -> Option<String> {
        for batch in batches {
            let id_col = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let name_col = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            for row in 0..batch.num_rows() {
                if id_col.value(row) == id {
                    return Some(name_col.value(row).to_string());
                }
            }
        }
        None
    }

    // ── CoW tests ─────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_cow_empty_table_all_inserts() {
        let (catalog, _tmp) = setup_catalog().await;
        let table = setup_table(&catalog).await;

        let source = make_batch(vec![1, 2, 3], vec!["alice", "bob", "charlie"], vec![
            Some("v1"),
            Some("v2"),
            Some("v3"),
        ]);
        let config = UpsertConfig {
            write_mode: UpsertWriteMode::CopyOnWrite,
            ..Default::default()
        };

        let (table, result) = upsert(&table, &catalog, source, config).await.unwrap();

        assert_eq!(result.rows_inserted, 3);
        assert_eq!(result.rows_updated, 0);
        assert_eq!(result.files_removed, 0);

        let batches = scan_all(&table).await;
        assert_eq!(total_rows(&batches), 3);
        assert_eq!(sorted_ids(&batches), vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn test_cow_no_key_overlap_inserts_into_existing_table() {
        let (catalog, _tmp) = setup_catalog().await;
        let table = setup_table(&catalog).await;

        // Write initial data: ids 1, 2, 3.
        let initial = make_batch(vec![1, 2, 3], vec!["alice", "bob", "charlie"], vec![
            Some("v1"),
            Some("v2"),
            Some("v3"),
        ]);
        let table = commit_batch(&table, &catalog, initial).await;

        // Upsert with completely new ids: 4, 5.
        let source = make_batch(vec![4, 5], vec!["diana", "eve"], vec![
            Some("v4"),
            Some("v5"),
        ]);
        let config = UpsertConfig {
            write_mode: UpsertWriteMode::CopyOnWrite,
            ..Default::default()
        };

        let (table, result) = upsert(&table, &catalog, source, config).await.unwrap();

        assert_eq!(result.rows_inserted, 2);
        assert_eq!(result.rows_updated, 0);
        // The existing file with ids [1,2,3] should NOT be rewritten — no key overlap.
        assert_eq!(
            result.files_affected, 0,
            "existing file must not be rewritten for a pure insert"
        );

        let batches = scan_all(&table).await;
        assert_eq!(total_rows(&batches), 5);
        assert_eq!(sorted_ids(&batches), vec![1, 2, 3, 4, 5]);
    }

    #[tokio::test]
    async fn test_cow_full_overlap_all_updates() {
        let (catalog, _tmp) = setup_catalog().await;
        let table = setup_table(&catalog).await;

        // Initial data.
        let initial = make_batch(vec![1, 2], vec!["old_alice", "old_bob"], vec![
            Some("old_v1"),
            Some("old_v2"),
        ]);
        let table = commit_batch(&table, &catalog, initial).await;

        // Upsert with same ids but different values.
        let source = make_batch(vec![1, 2], vec!["new_alice", "new_bob"], vec![
            Some("new_v1"),
            Some("new_v2"),
        ]);
        let config = UpsertConfig {
            write_mode: UpsertWriteMode::CopyOnWrite,
            skip_unchanged: true,
            ..Default::default()
        };

        let (table, result) = upsert(&table, &catalog, source, config).await.unwrap();

        assert_eq!(result.rows_updated, 2);
        assert_eq!(result.rows_inserted, 0);
        assert!(
            result.files_affected > 0,
            "existing file should have been rewritten"
        );

        let batches = scan_all(&table).await;
        assert_eq!(total_rows(&batches), 2);
        assert_eq!(find_name(&batches, 1).as_deref(), Some("new_alice"));
        assert_eq!(find_name(&batches, 2).as_deref(), Some("new_bob"));
    }

    #[tokio::test]
    async fn test_cow_skip_unchanged_does_not_rewrite() {
        let (catalog, _tmp) = setup_catalog().await;
        let table = setup_table(&catalog).await;

        // Initial data.
        let initial = make_batch(vec![1, 2], vec!["alice", "bob"], vec![
            Some("v1"),
            Some("v2"),
        ]);
        let table = commit_batch(&table, &catalog, initial).await;

        // Upsert with identical rows (no actual change).
        let source = make_batch(vec![1, 2], vec!["alice", "bob"], vec![
            Some("v1"),
            Some("v2"),
        ]);
        let config = UpsertConfig {
            write_mode: UpsertWriteMode::CopyOnWrite,
            skip_unchanged: true,
            ..Default::default()
        };

        let (table, result) = upsert(&table, &catalog, source, config).await.unwrap();

        assert_eq!(
            result.files_affected, 0,
            "no files should be rewritten when rows are unchanged"
        );
        assert_eq!(result.rows_updated, 0);
        assert_eq!(result.rows_inserted, 0);

        // Verify data integrity: the original rows must still be readable after a no-op upsert.
        // A broken early-return that corrupts catalog state would fail here.
        let batches = scan_all(&table).await;
        assert_eq!(total_rows(&batches), 2, "original rows must be preserved");
        assert_eq!(find_name(&batches, 1).as_deref(), Some("alice"));
        assert_eq!(find_name(&batches, 2).as_deref(), Some("bob"));
    }

    #[tokio::test]
    async fn test_cow_mixed_updates_and_inserts() {
        let (catalog, _tmp) = setup_catalog().await;
        let table = setup_table(&catalog).await;

        // Initial data: ids 1, 2, 3.
        let initial = make_batch(vec![1, 2, 3], vec!["alice", "bob", "charlie"], vec![
            Some("v1"),
            Some("v2"),
            Some("v3"),
        ]);
        let table = commit_batch(&table, &catalog, initial).await;

        // Upsert: ids 2 & 3 update, ids 4 & 5 insert.
        let source = make_batch(
            vec![2, 3, 4, 5],
            vec!["new_bob", "new_charlie", "diana", "eve"],
            vec![Some("nv2"), Some("nv3"), Some("v4"), Some("v5")],
        );
        let config = UpsertConfig {
            write_mode: UpsertWriteMode::CopyOnWrite,
            ..Default::default()
        };

        let (table, result) = upsert(&table, &catalog, source, config).await.unwrap();

        assert_eq!(result.rows_updated, 2);
        assert_eq!(result.rows_inserted, 2);

        let batches = scan_all(&table).await;
        assert_eq!(total_rows(&batches), 5);
        assert_eq!(sorted_ids(&batches), vec![1, 2, 3, 4, 5]);
        // id=1 unchanged.
        assert_eq!(find_name(&batches, 1).as_deref(), Some("alice"));
        // id=2 updated.
        assert_eq!(find_name(&batches, 2).as_deref(), Some("new_bob"));
    }

    // ── MoR tests ─────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_mor_all_inserts_no_delete_file_created() {
        let (catalog, _tmp) = setup_catalog().await;
        let table = setup_table(&catalog).await;

        // Initial data.
        let initial = make_batch(vec![1, 2, 3], vec!["alice", "bob", "charlie"], vec![
            Some("v1"),
            Some("v2"),
            Some("v3"),
        ]);
        let table = commit_batch(&table, &catalog, initial).await;

        // Upsert with completely new ids → no matches, so no equality-delete file.
        let source = make_batch(vec![4, 5], vec!["diana", "eve"], vec![
            Some("v4"),
            Some("v5"),
        ]);
        let config = UpsertConfig {
            write_mode: UpsertWriteMode::MergeOnRead,
            ..Default::default()
        };

        let (table, result) = upsert(&table, &catalog, source, config).await.unwrap();

        assert_eq!(result.rows_inserted, 2);
        assert_eq!(result.rows_updated, 0);
        // No equality-delete file should be created when there are no matches.
        assert_eq!(result.files_affected, 0);

        let batches = scan_all(&table).await;
        assert_eq!(total_rows(&batches), 5);
        assert_eq!(sorted_ids(&batches), vec![1, 2, 3, 4, 5]);
    }

    #[tokio::test]
    async fn test_mor_all_updates_creates_equality_delete_file() {
        let (catalog, _tmp) = setup_catalog().await;
        let table = setup_table(&catalog).await;

        // Initial data.
        let initial = make_batch(vec![1, 2], vec!["old_alice", "old_bob"], vec![
            Some("old_v1"),
            Some("old_v2"),
        ]);
        let table = commit_batch(&table, &catalog, initial).await;

        // Upsert all rows with changed values → equality-delete file expected.
        let source = make_batch(vec![1, 2], vec!["new_alice", "new_bob"], vec![
            Some("new_v1"),
            Some("new_v2"),
        ]);
        let config = UpsertConfig {
            write_mode: UpsertWriteMode::MergeOnRead,
            ..Default::default()
        };

        let (table, result) = upsert(&table, &catalog, source, config).await.unwrap();

        assert_eq!(result.rows_updated, 2);
        assert_eq!(result.rows_inserted, 0);
        assert!(
            result.files_affected >= 1,
            "an equality-delete file should have been created"
        );

        // Scan must apply equality deletes → final count stays 2, not 4.
        let batches = scan_all(&table).await;
        assert_eq!(
            total_rows(&batches),
            2,
            "equality deletes must be applied during scan"
        );
        assert_eq!(find_name(&batches, 1).as_deref(), Some("new_alice"));
        assert_eq!(find_name(&batches, 2).as_deref(), Some("new_bob"));
    }

    #[tokio::test]
    async fn test_mor_mixed_updates_and_inserts() {
        let (catalog, _tmp) = setup_catalog().await;
        let table = setup_table(&catalog).await;

        // Initial data: ids 1, 2, 3.
        let initial = make_batch(vec![1, 2, 3], vec!["alice", "bob", "charlie"], vec![
            Some("v1"),
            Some("v2"),
            Some("v3"),
        ]);
        let table = commit_batch(&table, &catalog, initial).await;

        // Upsert: ids 2 & 3 update, ids 4 & 5 insert.
        let source = make_batch(
            vec![2, 3, 4, 5],
            vec!["new_bob", "new_charlie", "diana", "eve"],
            vec![Some("nv2"), Some("nv3"), Some("v4"), Some("v5")],
        );
        let config = UpsertConfig {
            write_mode: UpsertWriteMode::MergeOnRead,
            ..Default::default()
        };

        let (table, result) = upsert(&table, &catalog, source, config).await.unwrap();

        assert_eq!(result.rows_updated, 2);
        assert_eq!(result.rows_inserted, 2);

        // Equality deletes for ids 2 & 3 plus the new data file → 5 final rows.
        let batches = scan_all(&table).await;
        assert_eq!(total_rows(&batches), 5);
        assert_eq!(sorted_ids(&batches), vec![1, 2, 3, 4, 5]);
        // id=1 untouched.
        assert_eq!(find_name(&batches, 1).as_deref(), Some("alice"));
        // id=2 should reflect the updated value.
        assert_eq!(find_name(&batches, 2).as_deref(), Some("new_bob"));
    }

    #[tokio::test]
    async fn test_mor_two_rounds_accumulate_correctly() {
        let (catalog, _tmp) = setup_catalog().await;
        let table = setup_table(&catalog).await;

        // Round 0: write 3 initial rows.
        let initial = make_batch(vec![1, 2, 3], vec!["a1", "a2", "a3"], vec![
            Some("v1"),
            Some("v2"),
            Some("v3"),
        ]);
        let table = commit_batch(&table, &catalog, initial).await;

        // Round 1: update id=1, insert id=4.
        let source1 = make_batch(vec![1, 4], vec!["b1", "a4"], vec![Some("nv1"), Some("v4")]);
        let config = UpsertConfig {
            write_mode: UpsertWriteMode::MergeOnRead,
            ..Default::default()
        };
        let (table, r1) = upsert(&table, &catalog, source1, config.clone())
            .await
            .unwrap();
        assert_eq!(r1.rows_updated, 1);
        assert_eq!(r1.rows_inserted, 1);

        // Round 2: update id=2 (which is in the original data file, not Round 1's new file),
        // and insert id=5.
        let source2 = make_batch(vec![2, 5], vec!["b2", "a5"], vec![Some("nv2"), Some("v5")]);
        let (table, r2) = upsert(&table, &catalog, source2, config).await.unwrap();
        assert_eq!(r2.rows_updated, 1);
        assert_eq!(r2.rows_inserted, 1);

        // Final state: ids 1 (updated), 2 (updated), 3 (original), 4 (inserted r1),
        // 5 (inserted r2) → 5 rows.
        let batches = scan_all(&table).await;
        assert_eq!(total_rows(&batches), 5);
        assert_eq!(sorted_ids(&batches), vec![1, 2, 3, 4, 5]);
        assert_eq!(find_name(&batches, 1).as_deref(), Some("b1"));
        assert_eq!(find_name(&batches, 2).as_deref(), Some("b2"));
        assert_eq!(find_name(&batches, 3).as_deref(), Some("a3")); // untouched
    }

    // ── CoW IN-predicate tests ────────────────────────────────────────────────

    #[tokio::test]
    async fn test_cow_in_predicate_small_batch_correctness() {
        // ≤1000 keys → triggers IN-list predicate path in build_optimal_predicate.
        let (catalog, _tmp) = setup_catalog().await;
        let table = setup_table(&catalog).await;

        // Write 10 initial rows.
        let initial_ids: Vec<i64> = (1..=10).collect();
        let initial_names: Vec<&str> = (1..=10).map(|_| "orig").collect();
        let initial_values: Vec<Option<&str>> = (1..=10).map(|_| Some("v")).collect();
        let initial = make_batch(initial_ids, initial_names, initial_values);
        let table = commit_batch(&table, &catalog, initial).await;

        // Upsert 6 rows: ids 6–11 (ids 6–10 update, id 11 inserts).
        let source_ids: Vec<i64> = (6..=11).collect();
        let source_names: Vec<&str> = vec!["upd6", "upd7", "upd8", "upd9", "upd10", "new11"];
        let source_values: Vec<Option<&str>> = (6..=11).map(|_| Some("nv")).collect();
        let source = make_batch(source_ids, source_names, source_values);
        let config = UpsertConfig {
            write_mode: UpsertWriteMode::CopyOnWrite,
            ..Default::default()
        };
        let (table, _result) = upsert(&table, &catalog, source, config).await.unwrap();

        let batches = scan_all(&table).await;
        assert_eq!(total_rows(&batches), 11); // 10 original + 1 insert
        assert_eq!(sorted_ids(&batches), (1..=11).collect::<Vec<i64>>());
        // Unchanged original rows.
        assert_eq!(find_name(&batches, 1).as_deref(), Some("orig"));
        assert_eq!(find_name(&batches, 5).as_deref(), Some("orig"));
        // Updated rows.
        assert_eq!(find_name(&batches, 6).as_deref(), Some("upd6"));
        assert_eq!(find_name(&batches, 10).as_deref(), Some("upd10"));
        // Inserted row.
        assert_eq!(find_name(&batches, 11).as_deref(), Some("new11"));
    }

    #[tokio::test]
    async fn test_cow_in_predicate_large_batch_fallback() {
        // >1000 keys → triggers bounding-box fallback in build_optimal_predicate.
        let (catalog, _tmp) = setup_catalog().await;
        let table = setup_table(&catalog).await;

        // Write 100 initial rows.
        let initial_ids: Vec<i64> = (1..=100).collect();
        let initial_names: Vec<&str> = (1..=100).map(|_| "orig").collect();
        let initial_values: Vec<Option<&str>> = (1..=100).map(|_| Some("v")).collect();
        let initial = make_batch(initial_ids, initial_names, initial_values);
        let table = commit_batch(&table, &catalog, initial).await;

        // Upsert 210 rows: ids 51–260 (ids 51–100 update, ids 101–260 insert).
        // This exceeds IN_PREDICATE_LIMIT=1000, so bounding-box fallback is used.
        let source_ids: Vec<i64> = (51..=260).collect();
        let source_names: Vec<&str> = source_ids.iter().map(|_| "upserted").collect();
        let source_values: Vec<Option<&str>> = source_ids.iter().map(|_| Some("nv")).collect();
        let source = make_batch(source_ids, source_names, source_values);
        let config = UpsertConfig {
            write_mode: UpsertWriteMode::CopyOnWrite,
            ..Default::default()
        };
        let (table, _result) = upsert(&table, &catalog, source, config).await.unwrap();

        let batches = scan_all(&table).await;
        assert_eq!(total_rows(&batches), 260); // 50 original + 210 upserted
        // Original rows below update range untouched.
        assert_eq!(find_name(&batches, 1).as_deref(), Some("orig"));
        assert_eq!(find_name(&batches, 50).as_deref(), Some("orig"));
        // Updated rows.
        assert_eq!(find_name(&batches, 51).as_deref(), Some("upserted"));
        assert_eq!(find_name(&batches, 100).as_deref(), Some("upserted"));
        // Inserted rows.
        assert_eq!(find_name(&batches, 101).as_deref(), Some("upserted"));
        assert_eq!(find_name(&batches, 260).as_deref(), Some("upserted"));
    }

    #[tokio::test]
    async fn test_cow_in_predicate_mixed_updates_inserts() {
        // Small batch: verifies IN-list predicate doesn't break normal CoW semantics.
        let (catalog, _tmp) = setup_catalog().await;
        let table = setup_table(&catalog).await;

        let initial = make_batch(vec![1, 2, 3], vec!["a1", "a2", "a3"], vec![
            Some("v1"),
            Some("v2"),
            Some("v3"),
        ]);
        let table = commit_batch(&table, &catalog, initial).await;

        // Update id=1 and id=3, insert id=4.
        let source = make_batch(vec![1, 3, 4], vec!["new1", "new3", "new4"], vec![
            Some("nv1"),
            Some("nv3"),
            Some("v4"),
        ]);
        let config = UpsertConfig {
            write_mode: UpsertWriteMode::CopyOnWrite,
            ..Default::default()
        };
        let (table, result) = upsert(&table, &catalog, source, config).await.unwrap();
        assert_eq!(result.rows_updated, 2);
        assert_eq!(result.rows_inserted, 1);

        let batches = scan_all(&table).await;
        assert_eq!(total_rows(&batches), 4);
        assert_eq!(sorted_ids(&batches), vec![1, 2, 3, 4]);
        assert_eq!(find_name(&batches, 1).as_deref(), Some("new1")); // updated
        assert_eq!(find_name(&batches, 2).as_deref(), Some("a2")); // unchanged
        assert_eq!(find_name(&batches, 3).as_deref(), Some("new3")); // updated
        assert_eq!(find_name(&batches, 4).as_deref(), Some("new4")); // inserted
    }
}

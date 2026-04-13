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

//! Merge-on-Read upsert executor.
//!
//! For each target data file that overlaps the source's key range:
//!  1. Read the file into Arrow `RecordBatch`es.
//!  2. Run the [`super::matcher::UpsertMatcher`] to find matches.
//!  3. Collect matched source key values (for the equality delete file).
//!
//! After all files have been processed:
//!  4. Write an equality-delete file containing all matched key values.
//!  5. Write a new data file containing ALL source rows (updates + inserts).
//!  6. Commit via `OverwriteFilesAction` (both data and delete files).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use arrow_array::{ArrayRef, RecordBatch, UInt32Array};
use arrow_select::take::take;
use futures::stream::TryStreamExt;
use parquet::file::properties::WriterProperties;
use tracing::{debug, info};

use super::matcher::UpsertMatcher;
use super::planner::KeyColumn;
use super::{UpsertFileDelta, UpsertResult};
use crate::arrow::arrow_schema_to_schema;
use crate::arrow::record_batch_partition_splitter::RecordBatchPartitionSplitter;
use crate::catalog::Catalog;
use crate::spec::{DataFile, DataFileFormat, PartitionKey, PartitionSpec, Struct};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::utils::{DEFAULT_UPSERT_DATA_LOAD_CONCURRENCY, load_data_files};
use crate::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use crate::writer::base_writer::equality_delete_writer::{
    EqualityDeleteFileWriterBuilder, EqualityDeleteWriterConfig,
};
use crate::writer::file_writer::ParquetWriterBuilder;
use crate::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use crate::writer::partitioning::PartitioningWriter;
use crate::writer::partitioning::fanout_writer::FanoutWriter;
use crate::writer::{IcebergWriter, IcebergWriterBuilder};
use crate::{Error, ErrorKind, Result};

const MOR_WRITER_NAME: &str = "upsert-mor";

/// Execute a Merge-on-Read upsert.
pub(super) async fn execute(
    table: &Table,
    catalog: &dyn Catalog,
    source: RecordBatch,
    key_columns: &[KeyColumn],
    non_key_column_indices: &[usize],
    predicate: &crate::expr::Predicate,
) -> Result<(Table, UpsertResult)> {
    let key_indices: Vec<usize> = key_columns.iter().map(|k| k.schema_index).collect();
    let file_io = table.file_io().clone();

    // Reload table from catalog so the scan sees all committed data files
    let table = catalog.load_table(table.identifier()).await?;

    // Get the default partition spec (None if the table is unpartitioned).
    let partition_spec = table.metadata().default_partition_spec();

    // Build matcher from source.
    let mut matcher = UpsertMatcher::new(&source, &key_indices, non_key_column_indices)?;

    // Track matched source indices keyed by the TARGET data file's partition.
    // Using the target partition (not the source row's partition) is critical for
    // correctness when the upsert changes the partition column value: the equality
    // delete must be placed in the OLD partition where the row to be removed lives.
    let mut matched_by_target_partition: HashMap<Struct, Vec<u32>> = HashMap::new();

    // Scan matching files.
    let scan = table.scan().with_filter(predicate.clone()).build()?;

    let scan_start = Instant::now();
    let file_scan_stream = scan.plan_files().await?;
    let file_tasks: Vec<_> = file_scan_stream.try_collect().await?;
    let scan_duration = scan_start.elapsed();

    info!(
        files_scanned = file_tasks.len(),
        scan_duration_ms = scan_duration.as_millis() as u64,
        source_rows = source.num_rows(),
        "MoR: scanned files matching predicate"
    );

    let read_start = Instant::now();

    // Phase 1: Read all matched files concurrently
    let data_file_reads = load_data_files(
        &file_io,
        file_tasks.clone(),
        DEFAULT_UPSERT_DATA_LOAD_CONCURRENCY,
    )
    .await?;

    // Phase 2: Sequential matcher processing (required - UpsertMatcher has mutable state)
    let mut files_read = 0;
    let mut total_rows_read = 0;
    let mut total_bytes_read: u64 = 0;

    for read_result in &data_file_reads {
        let target_partition = read_result
            .task
            .data_file
            .as_ref()
            .map(|df| df.partition().clone())
            .unwrap_or_else(Struct::empty);

        if let Some(file_size) = read_result.file_size {
            total_bytes_read += file_size;
            debug!(
                file_path = %read_result.task.data_file.as_ref().map(|df| &df.file_path).unwrap_or(&Default::default()),
                file_size_bytes = file_size,
                batches_read = read_result.batches.len(),
                rows_read = read_result.batches.iter().map(|b| b.num_rows() as u64).sum::<u64>(),
                "MoR: processed file"
            );
        }

        files_read += 1;
        total_rows_read += read_result
            .batches
            .iter()
            .map(|b| b.num_rows() as u64)
            .sum::<u64>();

        let partition_matches = matched_by_target_partition
            .entry(target_partition)
            .or_default();
        for batch in &read_result.batches {
            let result = matcher.match_batch(batch)?;
            for i in 0..result.matched_source_indices.len() {
                partition_matches.push(result.matched_source_indices.value(i));
            }
        }
    }
    let read_duration = read_start.elapsed();

    // Count globally unique matched source indices for stats.
    let rows_updated = {
        let unique: HashSet<u32> = matched_by_target_partition
            .values()
            .flat_map(|v| v.iter().copied())
            .collect();
        unique.len() as u64
    };
    let rows_inserted = matcher.source_len() as u64 - rows_updated;

    info!(
        files_read,
        total_files_scanned = file_tasks.len(),
        total_rows_read,
        total_bytes_read,
        read_duration_ms = read_duration.as_millis() as u64,
        scan_duration_ms = scan_duration.as_millis() as u64,
        "MoR: completed file reading phase"
    );

    // Write equality-delete files, one per target partition (or a single global file
    // for unpartitioned tables). A source row may match in multiple target files within
    // the same partition; we deduplicate per partition before writing.
    let mut added_delete_files: Vec<DataFile> = Vec::new();
    let write_start = Instant::now();
    for (target_partition, indices) in matched_by_target_partition {
        if indices.is_empty() {
            continue;
        }
        // Per-partition deduplication.
        let unique_indices: Vec<u32> = {
            let mut seen = HashSet::with_capacity(indices.len());
            indices.into_iter().filter(|&i| seen.insert(i)).collect()
        };
        let eq_delete_batch = build_equality_delete_batch(&source, key_columns, &unique_indices)?;
        let partition_key = if partition_spec.is_unpartitioned() {
            None
        } else {
            Some(PartitionKey::new(
                partition_spec.as_ref().clone(),
                table.metadata().current_schema().clone(),
                target_partition,
            ))
        };
        let eq_files =
            write_equality_delete_file(&table, eq_delete_batch, key_columns, partition_key).await?;
        added_delete_files.extend(eq_files);
    }

    // Write new data file: ALL source rows (matched updates + inserts).
    // These ARE partitioned using the table's partition spec.
    let added_data_files = write_data_file(&table, source.clone(), Some(partition_spec)).await?;
    let data_files_len = added_data_files.len();
    let data_file_size = added_data_files
        .iter()
        .map(|f| f.file_size_in_bytes)
        .sum::<u64>();

    // Commit via OverwriteFilesAction — it auto-classifies files by DataContentType.
    let commit_start = Instant::now();
    let tx = Transaction::new(&table);

    // Compute next sequence number from the already-reloaded table metadata.
    let current_last_seq = table.metadata().last_sequence_number();
    let next_seq = current_last_seq + 1;

    let overwrite_action = tx
        .overwrite_files()
        .add_data_files(added_data_files)
        .add_data_files(added_delete_files.clone()) // equality deletes
        .set_new_data_file_sequence_number(next_seq);

    let tx = overwrite_action.apply(tx)?;
    let committed_table = tx.commit(catalog).await?;
    let commit_duration = commit_start.elapsed();

    let total_duration = scan_start.elapsed();
    info!(
        total_duration_ms = total_duration.as_millis() as u64,
        scan_duration_ms = scan_duration.as_millis() as u64,
        read_duration_ms = read_duration.as_millis() as u64,
        write_duration_ms = write_start.elapsed().as_millis() as u64,
        commit_duration_ms = commit_duration.as_millis() as u64,
        rows_updated,
        rows_inserted,
        files_affected = added_delete_files.len() as u64,
        files_added = (added_delete_files.len() + data_files_len) as u64,
        files_removed = 0_u64,
        data_file_size_bytes = data_file_size,
        "MoR: upsert round complete"
    );

    Ok((committed_table, UpsertResult {
        rows_updated,
        rows_inserted,
        files_affected: added_delete_files.len() as u64,
        files_added: (added_delete_files.len() + 1) as u64,
        files_removed: 0,
    }))
}

/// Compute the MoR file delta without committing to any catalog.
///
/// The caller must pass a fresh, up-to-date `Table`. This function performs
/// the same scan-match-write work as [`execute`] but skips the catalog reload
/// and transaction commit, returning the raw file delta instead.
pub(super) async fn compute(
    table: &Table,
    source: RecordBatch,
    key_columns: &[KeyColumn],
    non_key_column_indices: &[usize],
    predicate: &crate::expr::Predicate,
) -> Result<UpsertFileDelta> {
    let key_indices: Vec<usize> = key_columns.iter().map(|k| k.schema_index).collect();
    let file_io = table.file_io().clone();

    // Get the default partition spec (None if the table is unpartitioned).
    let partition_spec = table.metadata().default_partition_spec();

    let mut matcher = UpsertMatcher::new(&source, &key_indices, non_key_column_indices)?;
    let mut matched_by_target_partition: HashMap<Struct, Vec<u32>> = HashMap::new();

    let scan = table.scan().with_filter(predicate.clone()).build()?;
    let file_scan_stream = scan.plan_files().await?;
    let file_tasks: Vec<_> = file_scan_stream.try_collect().await?;

    let data_file_reads = load_data_files(
        &file_io,
        file_tasks.clone(),
        DEFAULT_UPSERT_DATA_LOAD_CONCURRENCY,
    )
    .await?;

    for read_result in &data_file_reads {
        let target_partition = read_result
            .task
            .data_file
            .as_ref()
            .map(|df| df.partition().clone())
            .unwrap_or_else(Struct::empty);
        let partition_matches = matched_by_target_partition
            .entry(target_partition)
            .or_default();
        for batch in &read_result.batches {
            let result = matcher.match_batch(batch)?;
            for i in 0..result.matched_source_indices.len() {
                partition_matches.push(result.matched_source_indices.value(i));
            }
        }
    }

    let rows_updated = {
        let unique: HashSet<u32> = matched_by_target_partition
            .values()
            .flat_map(|v| v.iter().copied())
            .collect();
        unique.len() as u64
    };
    let rows_inserted = matcher.source_len() as u64 - rows_updated;

    let mut added_delete_files: Vec<DataFile> = Vec::new();
    for (target_partition, indices) in matched_by_target_partition {
        if indices.is_empty() {
            continue;
        }
        let unique_indices: Vec<u32> = {
            let mut seen = HashSet::with_capacity(indices.len());
            indices.into_iter().filter(|&i| seen.insert(i)).collect()
        };
        let eq_delete_batch = build_equality_delete_batch(&source, key_columns, &unique_indices)?;
        let partition_key = if partition_spec.is_unpartitioned() {
            None
        } else {
            Some(PartitionKey::new(
                partition_spec.as_ref().clone(),
                table.metadata().current_schema().clone(),
                target_partition,
            ))
        };
        let eq_files =
            write_equality_delete_file(table, eq_delete_batch, key_columns, partition_key).await?;
        added_delete_files.extend(eq_files);
    }

    // Data files ARE partitioned using the table's partition spec.
    let added_data_files = write_data_file(table, source.clone(), Some(partition_spec)).await?;

    // Combine: new data files first, then equality-delete files.
    let mut all_added = added_data_files;
    all_added.extend(added_delete_files.iter().cloned());
    let files_added = all_added.len() as u64;

    Ok(UpsertFileDelta {
        added_data_files: all_added,
        deleted_data_files: vec![],
        stats: UpsertResult {
            rows_updated,
            rows_inserted,
            files_affected: added_delete_files.len() as u64,
            files_added,
            files_removed: 0,
        },
    })
}

/// Build a RecordBatch containing the key columns for the equality-delete file.
/// Takes only the rows at `matched_indices` from the source batch.
fn build_equality_delete_batch(
    source: &RecordBatch,
    key_columns: &[KeyColumn],
    matched_indices: &[u32],
) -> Result<RecordBatch> {
    let indices_array = UInt32Array::from(matched_indices.to_vec());

    let key_arrays: Result<Vec<ArrayRef>> = key_columns
        .iter()
        .map(|k| {
            let col = source.column(k.schema_index);
            take(col.as_ref(), &indices_array, None).map_err(|e| {
                Error::new(
                    ErrorKind::Unexpected,
                    format!("Failed to take key column '{}': {e}", k.name),
                )
            })
        })
        .collect();

    let key_arrays = key_arrays?;

    // Build the equality-delete schema (key columns only, same names/types).
    let fields: Vec<arrow_schema::Field> = key_columns
        .iter()
        .map(|k| {
            arrow_schema::Field::new(
                k.name.as_str(),
                source.schema().field(k.schema_index).data_type().clone(),
                source.schema().field(k.schema_index).is_nullable(),
            )
        })
        .collect();

    let schema = arrow_schema::Schema::new(fields);

    RecordBatch::try_new(Arc::new(schema), key_arrays).map_err(|e| {
        Error::new(
            ErrorKind::Unexpected,
            format!("Failed to build equality-delete batch: {e}"),
        )
    })
}

/// Write a single equality-delete file and return its DataFile.
///
/// `partition_key` should be `None` for global (unpartitioned) equality deletes and
/// `Some(pk)` when writing a partition-scoped equality delete file.  The caller is
/// responsible for splitting the batch by partition before calling this function.
async fn write_equality_delete_file(
    table: &Table,
    batch: RecordBatch,
    key_columns: &[KeyColumn],
    partition_key: Option<PartitionKey>,
) -> Result<Vec<DataFile>> {
    let file_io = table.file_io().clone();
    let metadata = table.metadata();
    let location_generator = DefaultLocationGenerator::new(metadata.clone()).map_err(|e| {
        Error::new(
            ErrorKind::Unexpected,
            format!("Failed to create location generator: {e}"),
        )
    })?;

    let file_name_generator = DefaultFileNameGenerator::new(
        format!("{MOR_WRITER_NAME}-eq-delete"),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );

    // Build equality delete config first to get the projected schema.
    let eq_ids: Vec<i32> = key_columns.iter().map(|k| k.field_id).collect();
    let config = EqualityDeleteWriterConfig::new(eq_ids, metadata.current_schema().clone())?;

    // Convert the projected Arrow schema (key columns only) to an Iceberg schema
    // so the ParquetWriter can use it for the file footer schema.
    let eq_schema: crate::spec::SchemaRef = Arc::new(arrow_schema_to_schema(
        config.projected_arrow_schema_ref().as_ref(),
    )?);

    let parquet_writer_builder =
        ParquetWriterBuilder::new(WriterProperties::default(), eq_schema.clone());

    let rolling_writer_builder = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_writer_builder,
        file_io.clone(),
        location_generator,
        file_name_generator,
    );

    let eq_writer_builder = EqualityDeleteFileWriterBuilder::new(rolling_writer_builder, config);

    let mut eq_writer = eq_writer_builder.build(partition_key).await.map_err(|e| {
        Error::new(
            ErrorKind::Unexpected,
            format!("Failed to build equality delete writer: {e}"),
        )
    })?;

    eq_writer.write(batch).await.map_err(|e| {
        Error::new(
            ErrorKind::Unexpected,
            format!("Failed to write equality delete batch: {e}"),
        )
    })?;

    eq_writer.close().await.map_err(|e| {
        Error::new(
            ErrorKind::Unexpected,
            format!("Failed to close equality delete writer: {e}"),
        )
    })
}

/// Write a single data file from a RecordBatch.
async fn write_data_file(
    table: &Table,
    batch: RecordBatch,
    partition_spec: Option<&PartitionSpec>,
) -> Result<Vec<DataFile>> {
    let file_io = table.file_io().clone();
    let metadata = table.metadata();
    let schema = metadata.current_schema().clone();
    let location_generator = DefaultLocationGenerator::new(metadata.clone()).map_err(|e| {
        Error::new(
            ErrorKind::Unexpected,
            format!("Failed to create location generator: {e}"),
        )
    })?;

    let file_name_generator = DefaultFileNameGenerator::new(
        MOR_WRITER_NAME.to_string(),
        Some(uuid::Uuid::now_v7().to_string()),
        DataFileFormat::Parquet,
    );

    let parquet_writer_builder =
        ParquetWriterBuilder::new(WriterProperties::default(), schema.clone());

    let rolling_writer_builder = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_writer_builder,
        file_io.clone(),
        location_generator,
        file_name_generator,
    );

    let data_file_writer_builder = DataFileWriterBuilder::new(rolling_writer_builder);

    // Use FanoutWriter when partitioned to write to multiple partition files.
    if let Some(spec) = partition_spec
        && !spec.is_unpartitioned()
    {
        let splitter = RecordBatchPartitionSplitter::try_new_with_computed_values(
            schema.clone(),
            Arc::new(spec.clone()),
        )?;

        let mut fanout_writer = FanoutWriter::new(data_file_writer_builder);

        let partitioned = splitter.split(&batch)?;
        for (partition_key, partitioned_batch) in partitioned {
            fanout_writer
                .write(partition_key, partitioned_batch)
                .await?;
        }

        return fanout_writer.close().await;
    }

    let mut writer = data_file_writer_builder.build(None).await.map_err(|e| {
        Error::new(
            ErrorKind::Unexpected,
            format!("Failed to build data file writer: {e}"),
        )
    })?;

    writer
        .write(batch)
        .await
        .map_err(|e| Error::new(ErrorKind::Unexpected, format!("Failed to write batch: {e}")))?;

    let files = writer.close().await.map_err(|e| {
        Error::new(
            ErrorKind::Unexpected,
            format!("Failed to close data file writer: {e}"),
        )
    })?;

    Ok(files)
}

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

//! Copy-on-Write upsert executor.
//!
//! For each target data file that overlaps the source's key range:
//!  1. Read the file into Arrow `RecordBatch`es.
//!  2. Run the [`super::matcher::UpsertMatcher`] to classify rows.
//!  3. If any matched rows have changed non-key columns, rewrite the file:
//!     new contents = unmatched-target-rows + unchanged-target-rows + matched-source-rows.
//!  4. After all files, write unmatched source rows (inserts) as new data files.
//!  5. Commit via `OverwriteFilesAction` (added files + removed old files).

use std::time::Instant;

use arrow_array::{BooleanArray, RecordBatch};
use arrow_select::filter::filter_record_batch;
use futures::stream::TryStreamExt;
use parquet::file::properties::WriterProperties;
use tracing::{debug, info};

use super::UpsertResult;
use super::matcher::UpsertMatcher;
use super::planner::KeyColumn;
use crate::catalog::Catalog;
use crate::spec::{DataFile, DataFileFormat};
use crate::table::Table;
use crate::transaction::{ApplyTransactionAction, Transaction};
use crate::utils::{DEFAULT_UPSERT_DATA_LOAD_CONCURRENCY, load_data_files};
use crate::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use crate::writer::file_writer::ParquetWriterBuilder;
use crate::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use crate::writer::{IcebergWriter, IcebergWriterBuilder};
use crate::{Error, ErrorKind, Result};

const UPSERT_WRITER_NAME: &str = "upsert";

/// Execute a Copy-on-Write upsert.
pub(super) async fn execute(
    table: &Table,
    catalog: &dyn Catalog,
    source: RecordBatch,
    key_columns: &[KeyColumn],
    non_key_column_indices: &[usize],
    predicate: &crate::expr::Predicate,
    skip_unchanged: bool,
) -> Result<(Table, UpsertResult)> {
    let key_indices: Vec<usize> = key_columns.iter().map(|k| k.schema_index).collect();
    let file_io = table.file_io().clone();

    // Reload table from catalog so the scan sees all committed data files,
    // including those added by previous upsert rounds.
    let table = catalog.load_table(table.identifier()).await?;

    // Build the matcher once from the source.
    let mut matcher = UpsertMatcher::new(&source, &key_indices, non_key_column_indices)?;

    // Track files to delete (original data files that were rewritten) and
    // files to add (rewritten files + insert files).
    let mut deleted_data_files: Vec<DataFile> = Vec::new();
    let mut added_data_files: Vec<DataFile> = Vec::new();
    let mut rows_updated: u64 = 0;
    let mut files_rewritten: u64 = 0;

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
        "CoW: scanned files matching predicate"
    );

    let read_start = Instant::now();

    // Phase 1: Read all matched files concurrently
    let data_file_reads = load_data_files(
        &file_io,
        file_tasks.clone(),
        DEFAULT_UPSERT_DATA_LOAD_CONCURRENCY,
    )
    .await?;

    // Phase 2: Sequential matcher processing and write (required - UpsertMatcher has mutable state)
    let mut files_read = 0;
    let mut total_rows_read = 0;
    let mut total_bytes_read: u64 = 0;

    for read_result in &data_file_reads {
        if let Some(file_size) = read_result.file_size {
            total_bytes_read += file_size;
            debug!(
                file_path = %read_result.task.data_file.as_ref().map(|df| &df.file_path).unwrap_or(&Default::default()),
                file_size_bytes = file_size,
                batches_read = read_result.batches.len(),
                rows_read = read_result.batches.iter().map(|b| b.num_rows() as u64).sum::<u64>(),
                "CoW: processed file"
            );
        }

        files_read += 1;
        total_rows_read += read_result
            .batches
            .iter()
            .map(|b| b.num_rows() as u64)
            .sum::<u64>();

        let mut has_changes = false;

        // Accumulated batches for the rewritten file.
        let mut rewrite_batches: Vec<RecordBatch> = Vec::new();

        for batch in &read_result.batches {
            let result = matcher.match_batch(batch)?;

            let n_matched = result.matched_target_indices.len();

            if n_matched == 0 {
                // No matches: keep entire batch unchanged.
                rewrite_batches.push(batch.clone());
                continue;
            }

            // Build the "changed" mask.
            let changed_mask: BooleanArray = if skip_unchanged {
                result.changed_mask
            } else {
                BooleanArray::from(vec![true; n_matched])
            };

            let has_changed = changed_mask.true_count();

            // Unchanged matched pairs: take from target.
            if has_changed < n_matched {
                // Build mask: true where matched AND unchanged.
                let unchanged_in_target = {
                    let mut m = vec![false; batch.num_rows()];
                    for i in 0..n_matched {
                        if !changed_mask.value(i) {
                            m[result.matched_target_indices.value(i) as usize] = true;
                        }
                    }
                    BooleanArray::from(m)
                };
                let kept = filter_record_batch(batch, &unchanged_in_target)?;
                rewrite_batches.push(kept);
            }

            // Changed pairs: take updated values from source.
            if has_changed > 0 {
                has_changes = true;
                let changed_in_source = {
                    let mut m = vec![false; matcher.source_len()];
                    for i in 0..n_matched {
                        if changed_mask.value(i) {
                            m[result.matched_source_indices.value(i) as usize] = true;
                        }
                    }
                    BooleanArray::from(m)
                };
                let updated = filter_record_batch(matcher.source(), &changed_in_source)?;
                rewrite_batches.push(updated);
                rows_updated += has_changed as u64;
            }

            // Unmatched target rows: keep as-is.
            if result.unmatched_target_mask.true_count() > 0 {
                let unmatched = filter_record_batch(batch, &result.unmatched_target_mask)?;
                rewrite_batches.push(unmatched);
            }
        }

        // Only rewrite if there are actual changes.
        if !has_changes {
            // No matches or no changes: skip file entirely.
            continue;
        }

        // Rewrite the file with merged content.
        let new_data_files = write_data_files(&table, rewrite_batches).await?;
        added_data_files.extend(new_data_files);

        // Mark the original file as deleted.
        if let Some(data_file) = &read_result.task.data_file {
            deleted_data_files.push(data_file.clone());
        }
        files_rewritten += 1;
    }
    let read_duration = read_start.elapsed();

    info!(
        files_read,
        total_files_scanned = file_tasks.len(),
        total_rows_read,
        total_bytes_read,
        read_duration_ms = read_duration.as_millis() as u64,
        scan_duration_ms = scan_duration.as_millis() as u64,
        "CoW: completed file reading phase"
    );

    // Write unmatched source rows (inserts).
    let write_start = Instant::now();
    let unmatched_mask = matcher.unmatched_source_mask();
    if unmatched_mask.true_count() > 0 {
        let insert_batch = filter_record_batch(matcher.source(), &unmatched_mask)?;
        let new_inserts = write_data_files(&table, vec![insert_batch]).await?;
        added_data_files.extend(new_inserts);
    }

    // Compute rows_inserted from the unmatched source count.
    let rows_inserted = unmatched_mask.true_count() as u64;

    // Commit via OverwriteFilesAction.
    let commit_start = Instant::now();
    let tx = Transaction::new(&table);
    let overwrite_action = tx
        .overwrite_files()
        .add_data_files(added_data_files.clone())
        .delete_files(deleted_data_files.clone());

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
        files_affected = files_rewritten,
        files_added = added_data_files.len() as u64,
        files_removed = deleted_data_files.len() as u64,
        "CoW: upsert round complete"
    );

    Ok((committed_table, UpsertResult {
        rows_updated,
        rows_inserted,
        files_affected: files_rewritten,
        files_added: added_data_files.len() as u64,
        files_removed: deleted_data_files.len() as u64,
    }))
}

/// Write a sequence of RecordBatches as new Iceberg data files, returning
/// the resulting `Vec<DataFile>`.
async fn write_data_files(table: &Table, batches: Vec<RecordBatch>) -> Result<Vec<DataFile>> {
    if batches.is_empty() {
        return Ok(Vec::new());
    }

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
        UPSERT_WRITER_NAME.to_string(),
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

    let writer: DataFileWriterBuilder<
        ParquetWriterBuilder,
        DefaultLocationGenerator,
        DefaultFileNameGenerator,
    > = DataFileWriterBuilder::new(rolling_writer_builder);

    let mut writer = writer.build(None).await.map_err(|e| {
        Error::new(
            ErrorKind::Unexpected,
            format!("Failed to build data file writer: {e}"),
        )
    })?;

    for batch in batches {
        writer.write(batch).await.map_err(|e| {
            Error::new(ErrorKind::Unexpected, format!("Failed to write batch: {e}"))
        })?;
    }

    let data_files = writer.close().await.map_err(|e| {
        Error::new(
            ErrorKind::Unexpected,
            format!("Failed to close data file writer: {e}"),
        )
    })?;

    Ok(data_files)
}

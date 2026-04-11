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

    // 3. Build (or use provided) bounding-box pruning predicate
    let predicate = match config.pruning_predicate {
        Some(p) => p,
        None => planner::build_pruning_predicate(&source, &key_columns)?.ok_or_else(|| {
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

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

//! Arrow-based hash matching and columnar change detection for upsert.

use std::collections::HashMap;

use arrow_arith::boolean::{and, and_not, or};
use arrow_array::{Array, ArrayRef, BooleanArray, RecordBatch, UInt32Array};
use arrow_ord::cmp::neq;
use arrow_row::{OwnedRow, RowConverter, SortField};
use arrow_select::take::take;

use crate::{Error, ErrorKind, Result};

/// Result of matching one target [`RecordBatch`] against the source data.
#[derive(Debug)]
pub struct MatchResult {
    /// Boolean mask over the target batch: `true` = this target row has no matching source key
    /// and should be kept as-is (Copy-on-Write) or ignored (MoR).
    pub unmatched_target_mask: BooleanArray,

    /// Indices into the **source** batch for each matched pair (parallel to
    /// `matched_target_indices`).
    pub matched_source_indices: UInt32Array,

    /// Indices into the **target** batch for each matched pair (parallel to
    /// `matched_source_indices`).
    pub matched_target_indices: UInt32Array,

    /// Per-matched-pair boolean: `true` if at least one non-key column differs between
    /// the source and target row (accounting for null changes).
    /// Parallel to `matched_source_indices` / `matched_target_indices`.
    pub changed_mask: BooleanArray,
}

/// Arrow hash-join matcher for upsert operations.
///
/// Built once from the source [`RecordBatch`], then reused across all target files/batches.
/// Tracks which source rows have been matched so that unmatched rows can be written as inserts.
pub struct UpsertMatcher {
    row_converter: RowConverter,
    /// Map from (serialised) key row → 0-based row index in the source batch.
    source_key_index: HashMap<OwnedRow, usize>,
    /// `true` for each source row that has been matched by at least one target row.
    source_matched: Vec<bool>,
    /// The full source batch (needed for columnar non-key comparison).
    source: RecordBatch,
    key_column_indices: Vec<usize>,
    non_key_column_indices: Vec<usize>,
}

impl UpsertMatcher {
    /// Build a matcher from the source batch.
    ///
    /// `key_column_indices`: 0-based column positions of the join key columns in the source schema.
    /// `non_key_column_indices`: 0-based column positions of all remaining columns.
    pub fn new(
        source: &RecordBatch,
        key_column_indices: &[usize],
        non_key_column_indices: &[usize],
    ) -> Result<Self> {
        let sort_fields: Vec<SortField> = key_column_indices
            .iter()
            .map(|&i| SortField::new(source.schema().field(i).data_type().clone()))
            .collect();

        let row_converter = RowConverter::new(sort_fields)
            .map_err(|e| Error::new(ErrorKind::Unexpected, format!("RowConverter error: {e}")))?;

        let key_arrays: Vec<ArrayRef> = key_column_indices
            .iter()
            .map(|&i| source.column(i).clone())
            .collect();

        let source_rows = row_converter.convert_columns(&key_arrays).map_err(|e| {
            Error::new(
                ErrorKind::Unexpected,
                format!("RowConverter convert error: {e}"),
            )
        })?;

        let mut source_key_index = HashMap::with_capacity(source.num_rows());
        for (row_idx, row) in source_rows.iter().enumerate() {
            source_key_index.insert(row.owned(), row_idx);
        }

        Ok(UpsertMatcher {
            row_converter,
            source_key_index,
            source_matched: vec![false; source.num_rows()],
            source: source.clone(),
            key_column_indices: key_column_indices.to_vec(),
            non_key_column_indices: non_key_column_indices.to_vec(),
        })
    }

    /// Match a target batch against the source.
    ///
    /// Updates internal tracking of which source rows have been matched.
    /// May be called repeatedly for multiple target batches.
    pub fn match_batch(&mut self, target: &RecordBatch) -> Result<MatchResult> {
        let key_arrays: Vec<ArrayRef> = self
            .key_column_indices
            .iter()
            .map(|&i| target.column(i).clone())
            .collect();

        let target_rows = self
            .row_converter
            .convert_columns(&key_arrays)
            .map_err(|e| {
                Error::new(
                    ErrorKind::Unexpected,
                    format!("RowConverter convert error: {e}"),
                )
            })?;

        let n_target = target.num_rows();
        let mut unmatched_target = vec![true; n_target];
        let mut matched_source_idx: Vec<u32> = Vec::new();
        let mut matched_target_idx: Vec<u32> = Vec::new();

        for (target_row_idx, target_row) in target_rows.iter().enumerate() {
            if let Some(&source_row_idx) = self.source_key_index.get(&target_row.owned()) {
                unmatched_target[target_row_idx] = false;
                matched_source_idx.push(source_row_idx as u32);
                matched_target_idx.push(target_row_idx as u32);
                self.source_matched[source_row_idx] = true;
            }
        }

        let n_matched = matched_source_idx.len();

        let changed_mask = if n_matched == 0 || self.non_key_column_indices.is_empty() {
            BooleanArray::from(vec![false; n_matched])
        } else {
            let source_take_indices = UInt32Array::from(matched_source_idx.clone());
            let target_take_indices = UInt32Array::from(matched_target_idx.clone());

            detect_changes(
                &self.source,
                target,
                &self.non_key_column_indices,
                &source_take_indices,
                &target_take_indices,
            )?
        };

        Ok(MatchResult {
            unmatched_target_mask: BooleanArray::from(unmatched_target),
            matched_source_indices: UInt32Array::from(matched_source_idx),
            matched_target_indices: UInt32Array::from(matched_target_idx),
            changed_mask,
        })
    }

    /// Returns a boolean mask over the source batch: `true` = this source row was never
    /// matched by any target row and should be written as a new insert.
    pub fn unmatched_source_mask(&self) -> BooleanArray {
        BooleanArray::from(
            self.source_matched
                .iter()
                .map(|&matched| !matched)
                .collect::<Vec<bool>>(),
        )
    }

    /// Return the number of source rows (for sizing output allocations).
    pub fn source_len(&self) -> usize {
        self.source.num_rows()
    }

    /// Reference to the underlying source batch.
    pub fn source(&self) -> &RecordBatch {
        &self.source
    }
}

/// Build an element-wise boolean array indicating whether the row changed.
///
/// A row is "changed" if any non-key column differs between source and target, where "differs"
/// accounts for null transitions: `null → value`, `value → null`, or `value_a → value_b`.
fn detect_changes(
    source: &RecordBatch,
    target: &RecordBatch,
    non_key_column_indices: &[usize],
    source_indices: &UInt32Array,
    target_indices: &UInt32Array,
) -> Result<BooleanArray> {
    let mut any_changed: Option<BooleanArray> = None;

    for &col_idx in non_key_column_indices {
        let src_col = take_column(source.column(col_idx), source_indices)?;
        let tgt_col = take_column(target.column(col_idx), target_indices)?;

        let col_changed = column_changed(&src_col, &tgt_col)?;

        any_changed = Some(match any_changed {
            None => col_changed,
            Some(prev) => or(&prev, &col_changed)
                .map_err(|e| Error::new(ErrorKind::Unexpected, format!("Boolean OR error: {e}")))?,
        });
    }

    // Safe: we checked non_key_column_indices is non-empty and n_matched > 0 before calling.
    Ok(any_changed.expect("at least one non-key column"))
}

/// Take a subset of rows from an array using integer indices.
fn take_column(col: &dyn Array, indices: &UInt32Array) -> Result<ArrayRef> {
    take(col, indices, None)
        .map_err(|e| Error::new(ErrorKind::Unexpected, format!("Arrow take error: {e}")))
}

/// Compute a boolean array indicating whether each element differs between `src` and `tgt`,
/// accounting for null transitions.
///
/// Changed = (null status changed) OR (both non-null AND values differ)
fn column_changed(src: &ArrayRef, tgt: &ArrayRef) -> Result<BooleanArray> {
    // Null-status change: XOR via (src_null AND NOT tgt_null) OR (NOT src_null AND tgt_null).
    let src_null = is_null_mask(src.as_ref());
    let tgt_null = is_null_mask(tgt.as_ref());
    // xor(a, b) = and_not(a, b) OR and_not(b, a)
    let null_changed = {
        let left = and_not(&src_null, &tgt_null).map_err(|e| {
            Error::new(ErrorKind::Unexpected, format!("Boolean AND_NOT error: {e}"))
        })?;
        let right = and_not(&tgt_null, &src_null).map_err(|e| {
            Error::new(ErrorKind::Unexpected, format!("Boolean AND_NOT error: {e}"))
        })?;
        or(&left, &right)
            .map_err(|e| Error::new(ErrorKind::Unexpected, format!("Boolean OR error: {e}")))?
    };

    // Value change for non-null rows.  `neq` returns null when either input is null;
    // `and` with `both_non_null` converts those nulls to false (Kleene: null AND false = false).
    let both_non_null = and(&not_mask(&src_null), &not_mask(&tgt_null))
        .map_err(|e| Error::new(ErrorKind::Unexpected, format!("Boolean AND error: {e}")))?;

    // `neq` takes &dyn Datum; &ArrayRef (= &Arc<dyn Array>) implements Datum.
    let val_neq = neq(src, tgt).map_err(|e| {
        Error::new(
            ErrorKind::Unexpected,
            format!("Column neq comparison error: {e}"),
        )
    })?;

    // Suppress the null result from `neq` by masking with `both_non_null`.
    let val_changed = and(&val_neq, &both_non_null)
        .map_err(|e| Error::new(ErrorKind::Unexpected, format!("Boolean AND error: {e}")))?;

    or(&null_changed, &val_changed)
        .map_err(|e| Error::new(ErrorKind::Unexpected, format!("Boolean OR error: {e}")))
}

/// Build a boolean array: `true` where `arr[i]` is null.
fn is_null_mask(arr: &dyn Array) -> BooleanArray {
    match arr.logical_nulls() {
        None => BooleanArray::from(vec![false; arr.len()]),
        Some(nulls) => BooleanArray::from_iter((0..arr.len()).map(|i| Some(nulls.is_null(i)))),
    }
}

/// Negate a null-free [`BooleanArray`] element-wise (used on masks that are always valid).
fn not_mask(mask: &BooleanArray) -> BooleanArray {
    arrow_arith::boolean::not(mask).expect("not() on a null-free BooleanArray cannot fail")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Int32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};

    use super::*;
    use crate::upsert::planner::{non_key_column_indices, resolve_key_columns};

    fn make_schema() -> crate::spec::Schema {
        use crate::spec::{NestedField, PrimitiveType, Schema, Type};
        Schema::builder()
            .with_schema_id(1)
            .with_identifier_field_ids(vec![1])
            .with_fields(vec![
                Arc::new(NestedField::required(
                    1,
                    "id",
                    Type::Primitive(PrimitiveType::Int),
                )),
                Arc::new(NestedField::optional(
                    2,
                    "name",
                    Type::Primitive(PrimitiveType::String),
                )),
                Arc::new(NestedField::optional(
                    3,
                    "value",
                    Type::Primitive(PrimitiveType::Int),
                )),
            ])
            .build()
            .unwrap()
    }

    fn arrow_schema() -> ArrowSchema {
        ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("value", DataType::Int32, true),
        ])
    }

    fn make_batch(
        ids: Vec<i32>,
        names: Vec<Option<&str>>,
        values: Vec<Option<i32>>,
    ) -> RecordBatch {
        let schema = Arc::new(arrow_schema());
        RecordBatch::try_new(schema, vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(names)),
            Arc::new(Int32Array::from(values)),
        ])
        .unwrap()
    }

    fn indices(schema: &crate::spec::Schema) -> (Vec<usize>, Vec<usize>) {
        let keys = resolve_key_columns(schema, &[]).unwrap();
        let key_idx: Vec<usize> = keys.iter().map(|k| k.schema_index).collect();
        let non_key_idx = non_key_column_indices(schema, &keys).unwrap();
        (key_idx, non_key_idx)
    }

    #[test]
    fn test_no_matches() {
        let schema = make_schema();
        let source = make_batch(vec![1, 2], vec![Some("a"), Some("b")], vec![
            Some(10),
            Some(20),
        ]);
        let target = make_batch(vec![3, 4], vec![Some("c"), Some("d")], vec![
            Some(30),
            Some(40),
        ]);

        let (key_idx, non_key_idx) = indices(&schema);
        let mut matcher = UpsertMatcher::new(&source, &key_idx, &non_key_idx).unwrap();
        let result = matcher.match_batch(&target).unwrap();

        assert_eq!(result.unmatched_target_mask.len(), 2);
        assert!(result.unmatched_target_mask.value(0)); // target row 0 unmatched
        assert!(result.unmatched_target_mask.value(1)); // target row 1 unmatched
        assert_eq!(result.matched_source_indices.len(), 0);

        let unmatched_src = matcher.unmatched_source_mask();
        assert!(unmatched_src.value(0));
        assert!(unmatched_src.value(1));
    }

    #[test]
    fn test_all_matches_changed() {
        let schema = make_schema();
        let source = make_batch(vec![1, 2], vec![Some("new_a"), Some("new_b")], vec![
            Some(100),
            Some(200),
        ]);
        let target = make_batch(vec![1, 2], vec![Some("old_a"), Some("old_b")], vec![
            Some(10),
            Some(20),
        ]);

        let (key_idx, non_key_idx) = indices(&schema);
        let mut matcher = UpsertMatcher::new(&source, &key_idx, &non_key_idx).unwrap();
        let result = matcher.match_batch(&target).unwrap();

        // All target rows matched → unmatched mask all false
        assert!(!result.unmatched_target_mask.value(0));
        assert!(!result.unmatched_target_mask.value(1));

        // Both pairs changed
        assert_eq!(result.changed_mask.len(), 2);
        assert!(result.changed_mask.value(0));
        assert!(result.changed_mask.value(1));

        // Source rows all matched → unmatched mask all false
        let unmatched_src = matcher.unmatched_source_mask();
        assert!(!unmatched_src.value(0));
        assert!(!unmatched_src.value(1));
    }

    #[test]
    fn test_all_matches_unchanged() {
        let schema = make_schema();
        let source = make_batch(vec![1, 2], vec![Some("a"), Some("b")], vec![
            Some(10),
            Some(20),
        ]);
        let target = make_batch(vec![1, 2], vec![Some("a"), Some("b")], vec![
            Some(10),
            Some(20),
        ]);

        let (key_idx, non_key_idx) = indices(&schema);
        let mut matcher = UpsertMatcher::new(&source, &key_idx, &non_key_idx).unwrap();
        let result = matcher.match_batch(&target).unwrap();

        assert_eq!(result.changed_mask.len(), 2);
        assert!(!result.changed_mask.value(0));
        assert!(!result.changed_mask.value(1));
    }

    #[test]
    fn test_partial_match_mixed_change() {
        let schema = make_schema();
        // source: id=1 (new name), id=3 (new insert)
        let source = make_batch(vec![1, 3], vec![Some("new_a"), Some("c")], vec![
            Some(99),
            Some(30),
        ]);
        // target: id=1 (old name), id=2 (no source match)
        let target = make_batch(vec![1, 2], vec![Some("old_a"), Some("b")], vec![
            Some(10),
            Some(20),
        ]);

        let (key_idx, non_key_idx) = indices(&schema);
        let mut matcher = UpsertMatcher::new(&source, &key_idx, &non_key_idx).unwrap();
        let result = matcher.match_batch(&target).unwrap();

        // target row 0 (id=1) matched, target row 1 (id=2) unmatched
        assert!(!result.unmatched_target_mask.value(0));
        assert!(result.unmatched_target_mask.value(1));

        assert_eq!(result.matched_source_indices.len(), 1);
        assert_eq!(result.matched_source_indices.value(0), 0); // source row 0 (id=1)
        assert_eq!(result.matched_target_indices.value(0), 0); // target row 0 (id=1)

        // The matched pair has changed values
        assert!(result.changed_mask.value(0));

        // source row 0 (id=1) matched; source row 1 (id=3) unmatched → insert
        let unmatched_src = matcher.unmatched_source_mask();
        assert!(!unmatched_src.value(0));
        assert!(unmatched_src.value(1));
    }

    #[test]
    fn test_null_transition_detected_as_change() {
        let schema = make_schema();
        // source: id=1 with null name
        let source = make_batch(vec![1], vec![None], vec![Some(10)]);
        // target: id=1 with non-null name
        let target = make_batch(vec![1], vec![Some("a")], vec![Some(10)]);

        let (key_idx, non_key_idx) = indices(&schema);
        let mut matcher = UpsertMatcher::new(&source, &key_idx, &non_key_idx).unwrap();
        let result = matcher.match_batch(&target).unwrap();

        assert_eq!(result.changed_mask.len(), 1);
        assert!(
            result.changed_mask.value(0),
            "null→value should be detected as change"
        );
    }

    #[test]
    fn test_both_null_not_changed() {
        let schema = make_schema();
        // source and target both have null name
        let source = make_batch(vec![1], vec![None], vec![Some(10)]);
        let target = make_batch(vec![1], vec![None], vec![Some(10)]);

        let (key_idx, non_key_idx) = indices(&schema);
        let mut matcher = UpsertMatcher::new(&source, &key_idx, &non_key_idx).unwrap();
        let result = matcher.match_batch(&target).unwrap();

        assert_eq!(result.changed_mask.len(), 1);
        assert!(
            !result.changed_mask.value(0),
            "null→null should not be a change"
        );
    }

    #[test]
    fn test_multiple_target_batches() {
        let schema = make_schema();
        let source = make_batch(vec![1, 2, 3], vec![Some("a"), Some("b"), Some("c")], vec![
            Some(10),
            Some(20),
            Some(30),
        ]);
        let target1 = make_batch(vec![1], vec![Some("old")], vec![Some(99)]);
        let target2 = make_batch(vec![2], vec![Some("b")], vec![Some(20)]);

        let (key_idx, non_key_idx) = indices(&schema);
        let mut matcher = UpsertMatcher::new(&source, &key_idx, &non_key_idx).unwrap();

        let r1 = matcher.match_batch(&target1).unwrap();
        assert_eq!(r1.matched_source_indices.len(), 1);
        assert!(r1.changed_mask.value(0)); // id=1: name changed old→a, value changed 99→10

        let r2 = matcher.match_batch(&target2).unwrap();
        assert_eq!(r2.matched_source_indices.len(), 1);
        assert!(!r2.changed_mask.value(0)); // id=2: no change

        // Source row 2 (id=3) never matched → insert
        let unmatched_src = matcher.unmatched_source_mask();
        assert!(!unmatched_src.value(0)); // id=1 matched
        assert!(!unmatched_src.value(1)); // id=2 matched
        assert!(unmatched_src.value(2)); // id=3 unmatched → insert
    }
}

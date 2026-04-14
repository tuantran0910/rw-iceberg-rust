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

//! Key resolution, source validation, and pruning predicate construction for upsert.

/// Maximum number of distinct values in an IN-list predicate before falling back to a
/// bounding-box predicate.
///
/// This matches `IN_PREDICATE_LIMIT` in `InclusiveMetricsEvaluator`: when an IN predicate
/// has more than 1000 literals the evaluator short-circuits to `ROWS_MIGHT_MATCH` for every
/// file, providing zero pruning benefit.
///
/// Increased from 200 to 1000 to improve file pruning for upsert workloads with larger
/// batches. IN-list predicates provide much tighter pruning than bounding-box.
const IN_PREDICATE_LIMIT: usize = 1000;

use std::collections::HashSet;

use arrow_arith::aggregate::{max_string, min_string};
use arrow_array::types::*;
use arrow_array::{Array, PrimitiveArray, RecordBatch, StringArray};
use arrow_row::{RowConverter, SortField};
use arrow_schema::DataType;
use tracing::debug;

use crate::expr::{Predicate, Reference};
use crate::spec::{Datum, PrimitiveType, Schema};
use crate::{Error, ErrorKind, Result};

/// A resolved key column: its Iceberg field-id, the name in the table schema,
/// and its 0-based index in an Arrow [`RecordBatch`] produced from that schema.
#[derive(Debug, Clone)]
pub struct KeyColumn {
    /// Iceberg field-id.
    pub field_id: i32,
    /// Column name as it appears in the table schema.
    pub name: String,
    /// 0-based column index in a full-schema RecordBatch.
    pub schema_index: usize,
}

/// Resolve the join key columns.
///
/// If `explicit_columns` is non-empty, each name is looked up in `schema`.
/// Otherwise the schema's `identifier_field_ids()` are used.
///
/// Returns an error if:
/// - no key columns can be determined,
/// - any named column is absent from the schema,
/// - any key column has a floating-point or nested type (not valid for equality keys).
pub fn resolve_key_columns(schema: &Schema, explicit_columns: &[String]) -> Result<Vec<KeyColumn>> {
    let field_ids: Vec<i32> = if explicit_columns.is_empty() {
        schema.identifier_field_ids().collect()
    } else {
        explicit_columns
            .iter()
            .map(|name| {
                schema.field_id_by_name(name).ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!("Join column '{name}' not found in table schema"),
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?
    };

    if field_ids.is_empty() {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            "No join key columns found: either set identifier-field-ids on the table schema \
             or pass explicit join_columns",
        ));
    }

    let arrow_schema: arrow_schema::Schema = schema.try_into().map_err(|e| {
        Error::new(
            ErrorKind::Unexpected,
            format!("Failed to convert Iceberg schema to Arrow schema: {e}"),
        )
    })?;

    field_ids
        .into_iter()
        .map(|field_id| {
            let name = schema.name_by_field_id(field_id).ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("Identifier field-id {field_id} has no name in schema"),
                )
            })?;

            // Validate key type is primitive (not float, not nested)
            let field = schema.field_by_id(field_id).ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("Field id {field_id} not found in schema"),
                )
            })?;
            match field.field_type.as_ref() {
                crate::spec::Type::Primitive(PrimitiveType::Float)
                | crate::spec::Type::Primitive(PrimitiveType::Double) => {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Join column '{name}' has float type which is not valid for equality \
                             matching (per Iceberg spec)"
                        ),
                    ));
                }
                crate::spec::Type::Struct(_)
                | crate::spec::Type::List(_)
                | crate::spec::Type::Map(_) => {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Join column '{name}' is a nested type which is not valid for \
                             equality matching"
                        ),
                    ));
                }
                _ => {}
            }

            let schema_index = arrow_schema.index_of(name).map_err(|_| {
                Error::new(
                    ErrorKind::Unexpected,
                    format!("Column '{name}' not found in Arrow schema"),
                )
            })?;

            Ok(KeyColumn {
                field_id,
                name: name.to_string(),
                schema_index,
            })
        })
        .collect()
}

/// Derive the 0-based column indices for non-key columns in a full-schema RecordBatch.
pub fn non_key_column_indices(schema: &Schema, key_columns: &[KeyColumn]) -> Result<Vec<usize>> {
    let key_indices: HashSet<usize> = key_columns.iter().map(|k| k.schema_index).collect();

    let arrow_schema: arrow_schema::Schema = schema.try_into().map_err(|e| {
        Error::new(
            ErrorKind::Unexpected,
            format!("Failed to convert Iceberg schema to Arrow schema: {e}"),
        )
    })?;

    Ok((0..arrow_schema.fields().len())
        .filter(|i| !key_indices.contains(i))
        .collect())
}

/// Validate that the source batch has no duplicate key values.
///
/// Uses [`RowConverter`] to hash all key columns together and inserts each
/// row into a [`HashSet`].  Returns an error on the first duplicate found.
pub fn validate_source_keys(source: &RecordBatch, key_column_indices: &[usize]) -> Result<()> {
    let key_arrays: Vec<_> = key_column_indices
        .iter()
        .map(|&i| source.column(i).clone())
        .collect();

    let sort_fields: Vec<SortField> = key_column_indices
        .iter()
        .map(|&i| SortField::new(source.schema().field(i).data_type().clone()))
        .collect();

    let converter = RowConverter::new(sort_fields)
        .map_err(|e| Error::new(ErrorKind::Unexpected, format!("RowConverter error: {e}")))?;

    let rows = converter.convert_columns(&key_arrays).map_err(|e| {
        Error::new(
            ErrorKind::Unexpected,
            format!("RowConverter convert error: {e}"),
        )
    })?;

    let mut seen = HashSet::with_capacity(rows.num_rows());
    for row in rows.iter() {
        if !seen.insert(row.owned()) {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "Source data contains duplicate key values; upsert requires unique source keys",
            ));
        }
    }

    Ok(())
}

/// Build an IN-list predicate for a single column.
///
/// Extracts every distinct non-null value and returns `col IN (v1, v2, …)`.
/// Returns `None` when:
/// - the column type is unsupported (no `Datum` conversion available),
/// - the number of distinct values exceeds [`IN_PREDICATE_LIMIT`], or
/// - the column contains only nulls.
fn build_in_list_column_predicate(col_name: &str, col: &dyn Array) -> Result<Option<Predicate>> {
    macro_rules! int_in_list {
        ($array_type:ty, $datum_fn:expr) => {{
            let arr = col
                .as_any()
                .downcast_ref::<PrimitiveArray<$array_type>>()
                .unwrap();
            let mut datums: HashSet<Datum> = HashSet::new();
            for i in 0..arr.len() {
                if arr.is_valid(i) {
                    datums.insert($datum_fn(arr.value(i)));
                    if datums.len() > IN_PREDICATE_LIMIT {
                        return Ok(None);
                    }
                }
            }
            if datums.is_empty() {
                return Ok(None); // all-null column
            }
            Ok(Some(Reference::new(col_name).is_in(datums)))
        }};
    }

    match col.data_type() {
        DataType::Int8 => int_in_list!(Int8Type, |v: i8| Datum::int(v as i32)),
        DataType::Int16 => int_in_list!(Int16Type, |v: i16| Datum::int(v as i32)),
        DataType::Int32 => int_in_list!(Int32Type, |v: i32| Datum::int(v)),
        DataType::Int64 => int_in_list!(Int64Type, |v: i64| Datum::long(v)),
        DataType::UInt8 => int_in_list!(UInt8Type, |v: u8| Datum::int(v as i32)),
        DataType::UInt16 => int_in_list!(UInt16Type, |v: u16| Datum::int(v as i32)),
        DataType::UInt32 => int_in_list!(UInt32Type, |v: u32| Datum::long(v as i64)),
        DataType::UInt64 => int_in_list!(UInt64Type, |v: u64| Datum::long(v as i64)),
        DataType::Date32 => int_in_list!(Date32Type, |v: i32| Datum::date(v)),
        DataType::Utf8 | DataType::LargeUtf8 => {
            let arr = match col.as_any().downcast_ref::<StringArray>() {
                Some(a) => a,
                None => return Ok(None),
            };
            let mut datums: HashSet<Datum> = HashSet::new();
            for i in 0..arr.len() {
                if arr.is_valid(i) {
                    datums.insert(Datum::string(arr.value(i)));
                    if datums.len() > IN_PREDICATE_LIMIT {
                        return Ok(None);
                    }
                }
            }
            if datums.is_empty() {
                return Ok(None);
            }
            Ok(Some(Reference::new(col_name).is_in(datums)))
        }
        _ => Ok(None), // unsupported type — no pruning for this column
    }
}

/// Primary predicate builder for upsert: uses IN-list for ≤[`IN_PREDICATE_LIMIT`] distinct
/// key values per column, falling back to bounding-box for larger key sets.
///
/// An exact IN list lets the [`InclusiveMetricsEvaluator`] skip files whose value ranges
/// do not intersect any of the listed keys (not just the min/max bounding box).
/// Row-group filtering benefits similarly.
///
/// Returns `None` if the source batch is empty.
pub fn build_optimal_predicate(
    source: &RecordBatch,
    key_columns: &[KeyColumn],
) -> Result<Option<Predicate>> {
    if source.num_rows() == 0 {
        return Ok(None);
    }

    let mut predicate: Option<Predicate> = None;

    for key_col in key_columns {
        let col = source.column(key_col.schema_index);

        // Try IN-list first; if None (too many values or unsupported type) fall back to bounds.
        let col_pred = match build_in_list_column_predicate(key_col.name.as_str(), col.as_ref())? {
            Some(p) => Some(p),
            None => build_column_predicate(key_col.name.as_str(), col.as_ref())?,
        };

        if let Some(p) = col_pred {
            predicate = Some(match predicate {
                None => p,
                Some(existing) => existing.and(p),
            });
        }
    }

    if let Some(ref p) = predicate {
        debug!(predicate = %p, key_columns = key_columns.iter().map(|k| k.name.as_str()).collect::<Vec<_>>().join(","), source_rows = source.num_rows(), "Built optimal pruning predicate for upsert");
    }

    Ok(predicate)
}

/// Build a `col >= min AND col <= max` predicate for a single column.
/// Returns `None` for unsupported types (which simply means no pruning for that column).
fn build_column_predicate(col_name: &str, col: &dyn Array) -> Result<Option<Predicate>> {
    use arrow_arith::aggregate::{max, min};

    let reference = Reference::new(col_name);

    macro_rules! int_bounds {
        ($array_type:ty, $datum_fn:expr) => {{
            let arr = col
                .as_any()
                .downcast_ref::<PrimitiveArray<$array_type>>()
                .unwrap();
            let lo = min(arr);
            let hi = max(arr);
            match (lo, hi) {
                (Some(lo), Some(hi)) => {
                    let p = reference
                        .clone()
                        .greater_than_or_equal_to($datum_fn(lo))
                        .and(reference.clone().less_than_or_equal_to($datum_fn(hi)));
                    Some(p)
                }
                _ => None, // all-null column — no pruning
            }
        }};
    }

    Ok(match col.data_type() {
        DataType::Int8 => int_bounds!(Int8Type, |v: i8| Datum::int(v as i32)),
        DataType::Int16 => int_bounds!(Int16Type, |v: i16| Datum::int(v as i32)),
        DataType::Int32 => int_bounds!(Int32Type, |v: i32| Datum::int(v)),
        DataType::Int64 => int_bounds!(Int64Type, |v: i64| Datum::long(v)),
        DataType::UInt8 => int_bounds!(UInt8Type, |v: u8| Datum::int(v as i32)),
        DataType::UInt16 => int_bounds!(UInt16Type, |v: u16| Datum::int(v as i32)),
        DataType::UInt32 => int_bounds!(UInt32Type, |v: u32| Datum::long(v as i64)),
        DataType::UInt64 => int_bounds!(UInt64Type, |v: u64| Datum::long(v as i64)),
        DataType::Date32 => int_bounds!(Date32Type, |v: i32| Datum::date(v)),
        DataType::Utf8 | DataType::LargeUtf8 => {
            let arr = col.as_any().downcast_ref::<StringArray>();
            if let Some(arr) = arr {
                let lo = min_string(arr);
                let hi = max_string(arr);
                match (lo, hi) {
                    (Some(lo), Some(hi)) => {
                        let p = reference
                            .clone()
                            .greater_than_or_equal_to(Datum::string(lo))
                            .and(reference.clone().less_than_or_equal_to(Datum::string(hi)));
                        Some(p)
                    }
                    _ => None,
                }
            } else {
                None
            }
        }
        // For other types (timestamps, decimals, binary, etc.) skip pruning for now.
        // The scan will still correctly apply deletes and equality; we just won't prune files.
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Int32Array, Int64Array, StringArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};

    use super::*;
    use crate::spec::{NestedField, PrimitiveType, Schema, Type};

    fn test_schema() -> Schema {
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
                    Type::Primitive(PrimitiveType::Long),
                )),
            ])
            .build()
            .unwrap()
    }

    fn make_batch(
        ids: Vec<i32>,
        names: Vec<Option<&str>>,
        values: Vec<Option<i64>>,
    ) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("value", DataType::Int64, true),
        ]));
        RecordBatch::try_new(schema, vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(names)),
            Arc::new(Int64Array::from(values)),
        ])
        .unwrap()
    }

    #[test]
    fn test_resolve_key_columns_from_identifier_fields() {
        let schema = test_schema();
        let keys = resolve_key_columns(&schema, &[]).unwrap();
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].name, "id");
        assert_eq!(keys[0].field_id, 1);
    }

    #[test]
    fn test_resolve_key_columns_explicit() {
        let schema = test_schema();
        let keys = resolve_key_columns(&schema, &["name".to_string()]).unwrap();
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].name, "name");
    }

    #[test]
    fn test_resolve_key_columns_unknown_column() {
        let schema = test_schema();
        let result = resolve_key_columns(&schema, &["nonexistent".to_string()]);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_source_keys_no_duplicates() {
        let batch = make_batch(vec![1, 2, 3], vec![None, None, None], vec![
            None, None, None,
        ]);
        assert!(validate_source_keys(&batch, &[0]).is_ok());
    }

    #[test]
    fn test_validate_source_keys_with_duplicates() {
        let batch = make_batch(vec![1, 2, 1], vec![None, None, None], vec![
            None, None, None,
        ]);
        let result = validate_source_keys(&batch, &[0]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("duplicate key"));
    }
}

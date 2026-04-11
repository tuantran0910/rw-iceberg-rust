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

//! Upsert example using BigLake metastore.
//!
//! This example demonstrates:
//!  1. Creating a table with initial records
//!  2. Running a Copy-on-Write upsert (10K upsert records into 100K existing records)
//!  3. Measuring the time taken for the upsert operation
//!
//! # Usage
//!
//! ```bash
//! # Set up GCP credentials for BigLake
//! export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
//!
//! # Run the example
//! cargo run --example upsert-biglake
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use arrow_array::{Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use gcloud_auth::project::Config;
use gcloud_auth::token::DefaultTokenSourceProvider;
use iceberg::expr::{BinaryExpression, Predicate, PredicateOperator, Reference};
use iceberg::spec::{Datum, NestedField, PrimitiveType, Schema, Type};
use iceberg::upsert::{UpsertConfig, UpsertWriteMode, upsert};
use iceberg::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use iceberg::writer::file_writer::ParquetWriterBuilder;
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent};
use iceberg_catalog_rest::{
    REST_CATALOG_PROP_URI, REST_CATALOG_PROP_WAREHOUSE, RestCatalogBuilder,
};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use parquet::file::properties::WriterProperties;
use rand::Rng;
use rand::seq::SliceRandom;
use token_source::TokenSourceProvider;
use tracing_subscriber::fmt::format::FmtSpan;
use uuid::Uuid;

// ===== Configuration (same as rest_catalog_biglake.rs) =====

static GCP_PROJECT_ID: &str = "rainbow-data-production-483609";
static BIGLAKE_URI: &str = "https://biglake.googleapis.com/iceberg/v1/restcatalog";
static WAREHOUSE: &str = "bq://projects/rainbow-data-production-483609";
static GCS_BUCKET_LOCATION: &str = "gs://rainbow-data-production-iceberg/test_iceberg_upsert";
static BQ_CONNECTION: &str =
    "projects/rainbow-data-production-483609/locations/us/connections/iceberg_conn";
static USER_PROJECT_HEADER: &str = "x-goog-user-project";

static NAMESPACE_NAME: &str = "test_iceberg_upsert";
static TABLE_NAME: &str = "upsert_demo_v5";

// ===== Benchmark configuration =====

const INITIAL_RECORDS: usize = 1_000;
const UPSERT_BATCH_SIZE: usize = 100;
const UPSERT_ROUNDS: usize = 2;
// How many of the upsert records have keys that already exist in the initial data
const UPSERT_MATCH_RATIO: f32 = 0.9; // 90% update, 10% insert

// ============================================================================
// Table setup helpers
// ============================================================================

fn build_table_schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::optional(3, "value", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .with_schema_id(0)
        .with_identifier_field_ids(vec![1]) // id is the join key
        .build()
        .unwrap()
}

fn arrow_schema() -> ArrowSchema {
    ArrowSchema::new(vec![
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
    ])
}

// ============================================================================
// Data generation
// ============================================================================

fn random_string(len: usize) -> String {
    rand::thread_rng()
        .sample_iter(rand::distributions::Alphanumeric)
        .take(len)
        .map(char::from)
        .collect()
}

fn generate_initial_batch(n: usize, start_id: i64) -> RecordBatch {
    let ids: Vec<i64> = (start_id..start_id + n as i64).collect();

    let names: Vec<String> = (0..n)
        .map(|_| random_string(rand::thread_rng().gen_range(8..20)))
        .collect();
    let values: Vec<String> = (0..n)
        .map(|_| random_string(rand::thread_rng().gen_range(16..64)))
        .collect();

    let schema = Arc::new(arrow_schema());
    RecordBatch::try_new(schema, vec![
        Arc::new(Int64Array::from(ids)),
        Arc::new(StringArray::from(names)),
        Arc::new(StringArray::from(values)),
    ])
    .unwrap()
}

fn generate_upsert_batch(n: usize, existing_ids: &[i64], match_ratio: f32) -> RecordBatch {
    let n_existing = (n as f32 * match_ratio) as usize;
    let n_insert = n - n_existing;

    // Upsert records: half update existing keys, half are new inserts
    let mut ids: Vec<i64> = Vec::with_capacity(n);
    ids.extend(existing_ids.iter().take(n_existing).copied());
    // New insert IDs start after existing IDs
    let max_existing = existing_ids.iter().max().copied().unwrap_or(0);
    for i in 0..n_insert {
        ids.push(max_existing + 1 + i as i64);
    }
    // Shuffle so updates and inserts are mixed
    ids.shuffle(&mut rand::thread_rng());

    let names: Vec<String> = (0..n)
        .map(|_| random_string(rand::thread_rng().gen_range(8..20)))
        .collect();
    let values: Vec<String> = (0..n)
        .map(|_| random_string(rand::thread_rng().gen_range(16..64)))
        .collect();

    let schema = Arc::new(arrow_schema());
    RecordBatch::try_new(schema, vec![
        Arc::new(Int64Array::from(ids)),
        Arc::new(StringArray::from(names)),
        Arc::new(StringArray::from(values)),
    ])
    .unwrap()
}

// ============================================================================
// GCS Authentication helpers
// ============================================================================

/// Retrieve a GCS access token using Google Application Default Credentials (ADC).
/// This token can be used for both:
/// 1. BigLake REST API authentication (via "token" prop)
/// 2. GCS storage authentication (via "gcs.oauth2.token" prop)
async fn get_gcs_token() -> anyhow::Result<String> {
    let scopes = vec!["https://www.googleapis.com/auth/cloud-platform"];
    let config = Config::default().with_scopes(&scopes);
    let provider = DefaultTokenSourceProvider::new(config)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to initialize Google ADC: {e}"))?;
    let token = provider
        .token_source()
        .token()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to get GCS access token: {e}"))?;
    // Strip "Bearer " prefix if present
    let access_token = token.strip_prefix("Bearer ").unwrap_or(&token).to_string();
    Ok(access_token)
}

// ============================================================================
// Data writing helpers
// ============================================================================

async fn write_initial_data(
    table: &iceberg::table::Table,
    batches: Vec<RecordBatch>,
) -> anyhow::Result<Vec<iceberg::spec::DataFile>> {
    let file_io = table.file_io().clone();
    let metadata = table.metadata();
    let schema = metadata.current_schema().clone();

    let location_generator = DefaultLocationGenerator::new(metadata.clone())?;

    let mut all_files: Vec<iceberg::spec::DataFile> = Vec::new();

    // Write batches to separate files (no rolling - each batch = one file)
    // This creates ~100 small files (batch_size=10K, so 2M/10K = 200 files)
    // To get ~100 files, we group every 2 batches together
    let batches_per_file = 2;
    for (file_idx, chunk) in batches.chunks(batches_per_file).enumerate() {
        let file_name_generator = DefaultFileNameGenerator::new(
            format!("init-{file_idx:03}"),
            Some(Uuid::now_v7().to_string()),
            iceberg::spec::DataFileFormat::Parquet,
        );

        let parquet_writer_builder =
            ParquetWriterBuilder::new(WriterProperties::default(), schema.clone());

        let rolling_writer_builder = RollingFileWriterBuilder::new_with_default_file_size(
            parquet_writer_builder,
            file_io.clone(),
            location_generator.clone(),
            file_name_generator,
        );

        let writer: DataFileWriterBuilder<
            ParquetWriterBuilder,
            DefaultLocationGenerator,
            DefaultFileNameGenerator,
        > = DataFileWriterBuilder::new(rolling_writer_builder);

        let mut writer = writer.build(None).await?;

        for batch in chunk {
            writer.write(batch.clone()).await?;
        }

        let files = writer.close().await?;
        all_files.extend(files);
    }

    Ok(all_files)
}

async fn commit_initial_data(
    table: &iceberg::table::Table,
    catalog: &dyn Catalog,
    data_files: Vec<iceberg::spec::DataFile>,
) -> anyhow::Result<iceberg::table::Table> {
    use iceberg::transaction::{ApplyTransactionAction, Transaction};

    let tx = Transaction::new(table);
    let append_action = tx.fast_append().add_data_files(data_files);
    let tx = append_action.apply(tx)?;
    let table = tx.commit(catalog).await?;
    Ok(table)
}

// ============================================================================
// Record lookup helper
// ============================================================================

/// Read a single row by `id` from the table, applying all delete files (MoR).
/// Returns `Some((name, value))` if found, `None` if no row with that id exists.
async fn read_record_by_id(
    table: &iceberg::table::Table,
    id: i64,
) -> anyhow::Result<Option<(String, Option<String>)>> {
    use futures::TryStreamExt;

    let filter = Predicate::Binary(BinaryExpression::new(
        PredicateOperator::Eq,
        Reference::new("id"),
        Datum::long(id),
    ));

    let scan_stream = table.scan().with_filter(filter).build()?.to_arrow().await?;
    let batches: Vec<RecordBatch> = scan_stream.try_collect().await?;

    for batch in &batches {
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
        let value_col = batch
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        for i in 0..batch.num_rows() {
            if id_col.value(i) == id {
                let name = name_col.value(i).to_string();
                let value = if value_col.is_null(i) {
                    None
                } else {
                    Some(value_col.value(i).to_string())
                };
                return Ok(Some((name, value)));
            }
        }
    }

    Ok(None)
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Toggle tracing based on ICEBERG_TRACE environment variable
    // Set to "debug" for verbose logging, "info" for normal, "warn" for minimal
    let trace_level = std::env::var("ICEBERG_TRACE").unwrap_or_else(|_| "info".to_string());
    let tracing_level = match trace_level.as_str() {
        "debug" => tracing::Level::DEBUG,
        "info" => tracing::Level::INFO,
        "warn" => tracing::Level::WARN,
        "error" => tracing::Level::ERROR,
        _ => tracing::Level::INFO,
    };

    tracing_subscriber::fmt()
        .with_max_level(tracing_level)
        .with_span_events(FmtSpan::CLOSE)
        .try_init()
        .ok();

    println!("==========================================");
    println!("BigLake Upsert Benchmark Example");
    println!("==========================================");
    println!();
    println!("Configuration:");
    println!("  Initial records:     {INITIAL_RECORDS}");
    println!("  Upsert batch size:   {UPSERT_BATCH_SIZE}");
    println!("  Upsert rounds:       {UPSERT_ROUNDS}");
    println!("  Match ratio:         {:.0}%", UPSERT_MATCH_RATIO * 100.0);
    println!();

    // ----- Step 1: Get GCS token via ADC and connect to catalog -----
    println!("[1/5] Getting GCS token via ADC...");
    let start = Instant::now();
    let gcs_token = get_gcs_token().await?;
    println!("  ✓ Token retrieved in {:?}", start.elapsed());

    println!("[2/5] Connecting to BigLake catalog...");
    let start = Instant::now();
    let mut props = HashMap::new();
    props.insert(REST_CATALOG_PROP_URI.to_string(), BIGLAKE_URI.to_string());
    props.insert(
        REST_CATALOG_PROP_WAREHOUSE.to_string(),
        WAREHOUSE.to_string(),
    );
    // Use direct token for REST API auth instead of google-auth=true
    // This token will also be passed to FileIO via gcs.oauth2.token
    props.insert("token".to_string(), gcs_token.clone());
    props.insert("gcs.oauth2.token".to_string(), gcs_token.clone());
    props.insert(
        format!("header.{USER_PROJECT_HEADER}"),
        GCP_PROJECT_ID.to_string(),
    );

    let catalog = RestCatalogBuilder::default()
        .load("biglake", props)
        .await
        .unwrap();
    println!("  ✓ Connected in {:?}", start.elapsed());
    println!();

    let namespace_ident = NamespaceIdent::from_vec(vec![NAMESPACE_NAME.to_string()])?;
    let table_ident = TableIdent::new(namespace_ident.clone(), TABLE_NAME.to_string());

    // ----- Step 3: Create or recreate table -----
    println!("[3/5] Setting up table '{NAMESPACE_NAME}.{TABLE_NAME}'...");
    let start = Instant::now();

    // Drop table if it exists
    if catalog.table_exists(&table_ident).await? {
        catalog.drop_table(&table_ident).await?;
        println!("  Dropped existing table.");
    }

    // Create namespace if needed
    if !catalog.namespace_exists(&namespace_ident).await? {
        catalog
            .create_namespace(
                &namespace_ident,
                HashMap::from([("location".to_owned(), GCS_BUCKET_LOCATION.to_owned())]),
            )
            .await?;
    }

    let table_schema = build_table_schema();
    let table_creation = TableCreation::builder()
        .name(table_ident.name.clone())
        .schema(table_schema.clone())
        .properties(HashMap::from([(
            "bq_connection".to_string(),
            BQ_CONNECTION.to_string(),
        )]))
        .build();

    let table = catalog
        .create_table(&namespace_ident, table_creation)
        .await?;
    println!("  ✓ Table created in {:?}", start.elapsed());
    println!();

    // ----- Step 4: Write initial 100K records -----
    println!("[4/5] Writing {INITIAL_RECORDS} initial records...");
    let start = Instant::now();

    // Split into batches of 10,000 for writing
    let batch_size = 10_000;
    let mut all_batches: Vec<RecordBatch> = Vec::new();
    let existing_ids: Vec<i64> = (0..INITIAL_RECORDS as i64).collect();

    for chunk in (0..INITIAL_RECORDS).step_by(batch_size) {
        let end = (chunk + batch_size).min(INITIAL_RECORDS);
        let batch = generate_initial_batch(end - chunk, chunk as i64);
        all_batches.push(batch);
    }

    let data_files = write_initial_data(&table, all_batches).await?;
    let mut table = commit_initial_data(&table, &catalog, data_files).await?;
    println!(
        "  ✓ {INITIAL_RECORDS} records written in {:?}",
        start.elapsed()
    );
    println!();

    // ----- Step 5: Run multiple upsert rounds (Merge-on-Read) -----
    println!(
        "[5/5] Running {UPSERT_ROUNDS} Merge-on-Read upsert rounds ({UPSERT_BATCH_SIZE} records each)..."
    );
    println!();

    let config = UpsertConfig {
        join_columns: vec!["id".to_string()], // explicit join key
        write_mode: UpsertWriteMode::MergeOnRead,
        skip_unchanged: true,
        pruning_predicate: None,
    };

    // Track cumulative stats
    let mut total_rows_updated: u64 = 0;
    let mut total_rows_inserted: u64 = 0;
    let mut total_files_rewritten: u64 = 0;
    let mut total_files_added: u64 = 0;
    let mut total_files_removed: u64 = 0;
    let mut total_upsert_time = std::time::Duration::ZERO;

    // Current known IDs for generating upsert batches
    // Start with initial IDs, will expand with each round's inserts
    let mut known_ids: Vec<i64> = existing_ids.clone();
    let mut max_id = known_ids.iter().max().copied().unwrap_or(0);

    // Pick a probe ID that is guaranteed to be updated in every round.
    // generate_upsert_batch takes the first n_existing IDs from known_ids as update targets,
    // where n_existing = UPSERT_BATCH_SIZE * UPSERT_MATCH_RATIO = 90. ID 0 is always included.
    let probe_id = existing_ids[0];
    let record_before = read_record_by_id(&table, probe_id)
        .await?
        .expect("probe record must exist in initial data");
    println!(
        "  Probe record id={probe_id} BEFORE upsert: name='{}', value='{}'",
        record_before.0,
        record_before.1.as_deref().unwrap_or("null")
    );
    println!();

    for round in 1..=UPSERT_ROUNDS {
        println!("  ----- Round {round}/{UPSERT_ROUNDS} -----",);

        // Generate upsert batch: 50% update existing, 50% insert new
        let upsert_batch = generate_upsert_batch(UPSERT_BATCH_SIZE, &known_ids, UPSERT_MATCH_RATIO);

        // 90% update, 10% insert

        let start = Instant::now();
        let (updated_table, result) = upsert(&table, &catalog, upsert_batch, config.clone()).await?;
        table = updated_table;
        let elapsed = start.elapsed();
        total_upsert_time += elapsed;

        println!(
            "    ✓ Round {round} complete in {elapsed:?} ({} updates, {} inserts)",
            result.rows_updated, result.rows_inserted
        );

        total_rows_updated += result.rows_updated;
        total_rows_inserted += result.rows_inserted;
        total_files_rewritten += result.files_affected;
        total_files_added += result.files_added;
        total_files_removed += result.files_removed;

        // Add the new insert IDs to known_ids for the next round
        // The new IDs start after max_id and go up
        let new_insert_count = result.rows_inserted as usize;
        let new_ids: Vec<i64> = ((max_id + 1)..=(max_id + new_insert_count as i64)).collect();
        known_ids.extend(new_ids);
        max_id += new_insert_count as i64;

        println!();
    }

    println!("========== CUMULATIVE RESULTS ==========");
    println!("  Total records updated:  {total_rows_updated}");
    println!("  Total records inserted: {total_rows_inserted}");
    println!("  Total files rewritten:  {total_files_rewritten}");
    println!("  Total files added:      {total_files_added}");
    println!("  Total files removed:    {total_files_removed}");
    println!();
    println!("  ⏱  Total upsert time:  {total_upsert_time:?}");
    let total_rows_processed = total_rows_updated + total_rows_inserted;
    println!(
        "  ⚡ Throughput:   {:.2} rows/sec",
        total_rows_processed as f64 / total_upsert_time.as_secs_f64()
    );
    println!();

    // Calculate expected final record count
    // Initial + all inserts from upserts (updates don't change count)
    let expected_final_count = INITIAL_RECORDS + total_rows_inserted as usize;

    // ----- Step 6: Validate upsert results -----
    println!("[6/6] Validating upsert results...");

    // 6a. Count total records — equality deletes are applied automatically by the ArrowReader.
    use futures::TryStreamExt;
    let scan_stream = table.scan().build()?.to_arrow().await?;
    let batches: Vec<RecordBatch> = scan_stream.try_collect().await?;
    let total_count: usize = batches.iter().map(|b| b.num_rows()).sum();

    println!("  Total records in table (via scan): {total_count}");
    println!(
        "  Expected: {expected_final_count} (initial {INITIAL_RECORDS} + {total_rows_inserted} inserts)"
    );
    assert_eq!(
        total_count, expected_final_count,
        "record count mismatch: scan={total_count}, expected={expected_final_count}"
    );
    println!("  ✓ Record count matches expected");

    // 6b. Compare probe record before vs after upsert.
    let record_after = read_record_by_id(&table, probe_id)
        .await?
        .expect("probe record must still exist after upsert");

    println!("\n  Before/after comparison for probe record id={probe_id}:");
    println!(
        "    Before: name='{}', value='{}'",
        record_before.0,
        record_before.1.as_deref().unwrap_or("null")
    );
    println!(
        "    After:  name='{}', value='{}'",
        record_after.0,
        record_after.1.as_deref().unwrap_or("null")
    );

    assert_ne!(
        record_before, record_after,
        "probe record should have been updated by upsert (name/value should differ)"
    );
    println!("  ✓ Record values changed as expected");
    println!("==========================================");

    Ok(())
}

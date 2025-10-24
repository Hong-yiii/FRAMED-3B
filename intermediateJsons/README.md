# Intermediate JSONs Directory

This directory stores the output of each step in the photo curation pipeline for analysis and debugging.

## Directory Structure

```
intermediateJsons/
├── ingest/           # Ingest service outputs
├── preprocess/       # Preprocessing service outputs (future)
├── features/         # Feature extraction outputs
├── scoring/          # Scoring service outputs
├── clustering/       # Clustering service outputs
├── ranking/          # Ranking service outputs
├── optimizer/        # Optimization service outputs
├── exporter/         # Final export outputs
├── analyze_pipeline.py   # Analysis script
└── README.md
```

## File Naming Convention

Each output file follows the pattern: `{batch_id}_{step}_output.json`

Examples:
- `batch_2025-01-15_14-30-25_ingest_output.json`
- `batch_2025-01-15_14-30-25_features_output.json`
- `batch_2025-01-15_14-30-25_scoring_output.json`

## Usage

### Running the Analysis
```bash
cd intermediateJsons/
python analyze_pipeline.py
```

### Manual Inspection
```bash
# View ingest results
cat intermediateJsons/ingest/batch_xxx_ingest_output.json | jq '.photo_index | length'

# View scoring results
cat intermediateJsons/scoring/batch_xxx_scoring_output.json | jq '.scores[0]'

# View final curated list
cat intermediateJsons/exporter/batch_xxx_exporter_output.json | jq '.items | length'
```

## Analysis Features

The `analyze_pipeline.py` script provides:

- **Pipeline Completion Status**: Which steps completed successfully
- **Data Flow Analysis**: Record counts and data quality for each step
- **Performance Metrics**:
  - Scoring: Average scores, score ranges
  - Ranking: Number of pairwise judgments, token estimates
  - Optimizer: Coverage percentages across diversity axes
- **Issue Detection**: Missing outputs or data quality problems
- **Batch Comparison**: Compare performance across multiple batches

## Example Output

```
📊 PIPELINE ANALYSIS SUMMARY REPORT
================================================================================

🎯 Batch: batch_2025-01-15_14-30-25
   Steps Completed: 8/8
   Pipeline: ingest → features → scoring → clustering → ranking → optimizer → exporter

   🔄 INGEST:
      Records: 15
      Photos with EXIF: 12

   🔄 SCORING:
      Records: 15, Dropped: 0
      Avg Score: 0.72 (0.65 - 0.78)

   🔄 RANKING:
      Records: 8
      Pairs Judged: 12, Est. Tokens: 8,500

   🔄 OPTIMIZER:
      Records: 15
      Coverage: 83.2%
```

## Integration with Services

Each service automatically saves its output to the appropriate subdirectory:

```python
# In each service.process() method:
os.makedirs("intermediateJsons/features", exist_ok=True)
with open(f"intermediateJsons/features/{batch_id}_features_output.json", 'w') as f:
    json.dump(result, f, indent=2)
```

## Benefits

1. **Debugging**: Easily inspect what each step produced
2. **Performance Analysis**: Track how well each step performs
3. **Data Quality Monitoring**: Identify issues in the pipeline
4. **Experimentation**: Compare different parameter settings
5. **Audit Trail**: Complete record of processing decisions

## Cleanup

The intermediate JSONs are preserved for analysis but can be cleaned up periodically:

```bash
# Remove old batches (older than 7 days)
find intermediateJsons/ -name "*.json" -mtime +7 -delete

# Or remove specific batch
rm -rf intermediateJsons/*/*batch_old_batch_id*
```

# Scoring Service Documentation

## Overview

The Scoring Service computes quality scores for photos using extracted features. It serves as the critical quality assessment layer that converts raw feature data into normalized, comparable scores across multiple aesthetic and technical dimensions. The service implements both hard quality gates and nuanced scoring algorithms to ensure only high-quality photos advance through the curation pipeline.

## Key Responsibilities

1. **Technical Quality Assessment:** Evaluate capture quality and processing artifacts
2. **Aesthetic Scoring:** Assess visual appeal and composition
3. **Vibe Analysis:** Measure emotional impact and atmosphere
4. **Typography Evaluation:** Analyze text space and layout potential
5. **Composition Analysis:** Assess rule-of-thirds and balance
6. **Quality Gatekeeping:** Filter out technically inadequate photos

## Scoring Dimensions

### 1. Technical Quality (Q_tech)
**Purpose:** Assess fundamental image capture quality
**Components:**
- **Sharpness:** Focus quality and edge clarity (40% weight)
- **Exposure:** Brightness and contrast balance (40% weight)
- **Noise:** Sensor noise and grain levels (20% weight)
**Formula:** `0.4 × sharpness + 0.4 × exposure + 0.2 × (1 - noise)`
**Range:** 0.0 (poor quality) to 1.0 (excellent quality)

### 2. Aesthetic Score
**Purpose:** Evaluate visual appeal and artistic quality
**Formula:** `0.6 + 0.3 × (Q_tech + neg_space_ratio)`
**Influences:** Technical quality, composition balance, visual harmony
**Range:** 0.0 to 1.0

### 3. Vibe Score
**Purpose:** Assess emotional impact and atmosphere
**Formula:** `0.5 + 0.4 × neg_space_ratio`
**Influences:** Negative space, mood, storytelling potential
**Range:** 0.0 to 1.0

### 4. Typography Score
**Purpose:** Evaluate space for text overlay and headlines
**Formula:** `neg_space_ratio × 0.8 + 0.2`
**Influences:** Negative space ratio, composition layout
**Range:** 0.0 to 1.0

### 5. Composition Score
**Purpose:** Assess classical composition principles
**Formula:** `0.6 + 0.3 × Q_tech`
**Influences:** Technical quality, rule-of-thirds alignment
**Range:** 0.0 to 1.0

### 6. Total Preliminary Score
**Purpose:** Overall quality assessment before LLM ranking
**Formula:** `0.7 + 0.2 × Q_tech`
**Use:** Initial filtering and baseline ranking
**Range:** 0.0 to 1.0

## Quality Gates

### Technical Quality Gate
- **Threshold:** Q_tech > 0.3
- **Purpose:** Eliminate technically inadequate photos
- **Rationale:** Photos below this threshold have fundamental quality issues
- **Action:** Moved to `dropped_for_tech` list

### Scoring Bounds
- **Minimum Score:** All scores clamped to [0.0, 1.0]
- **Normalization:** Automatic scaling to prevent outliers
- **Consistency:** Same scoring scale across all dimensions

## Processing Workflow

### 1. Input Processing
- Receives feature-rich artifacts from features service
- Validates feature completeness and data types
- Handles missing features with default values

### 2. Score Computation
- Extract relevant features from feature dictionary
- Apply scoring formulas for each dimension
- Normalize and bound all scores to [0,1] range

### 3. Quality Gate Application
- Evaluate technical quality against threshold
- Separate passing and failing photos
- Log reasons for dropped photos

### 4. Output Generation
- Structure scores with photo IDs
- Include audit trail of scoring decisions
- Generate summary statistics

## Input Format

```json
{
  "batch_id": "batch_20250920_143703",
  "artifacts": [
    {
      "photo_id": "a6d54a6157e85c3fce236acca72d7f67...",
      "features": {
        "tech": {
          "sharpness": 0.85,
          "exposure": 0.72,
          "noise": 0.15,
          "horizon_deg": -0.8
        },
        "saliency": {
          "neg_space_ratio": 0.34
        }
      }
    }
  ]
}
```

## Output Format

```json
{
  "batch_id": "batch_20250920_143703",
  "scores": [
    {
      "photo_id": "a6d54a6157e85c3fce236acca72d7f67...",
      "Q_tech": 0.82,
      "Aesthetic": 0.73,
      "Vibe": 0.64,
      "Typography": 0.47,
      "Composition": 0.75,
      "Total_prelim": 0.76
    },
    {
      "photo_id": "b8c92d4e6f1a3h5j...",
      "Q_tech": 0.15,
      "Aesthetic": 0.45,
      "Vibe": 0.54,
      "Typography": 0.38,
      "Composition": 0.55,
      "Total_prelim": 0.63
    }
  ],
  "dropped_for_tech": [
    "c9d38e5f7g2b4i6k...",
    "d0e49f6g8h3c5j7l..."
  ]
}
```

## Scoring Algorithm Details

### Technical Quality Computation
```python
def compute_technical_quality(tech_features):
    sharpness = tech_features.get("sharpness", 0.5)
    exposure = tech_features.get("exposure", 0.5)
    noise = tech_features.get("noise", 0.5)

    q_tech = (0.4 * sharpness +
              0.4 * exposure +
              0.2 * (1 - noise))

    return max(0.0, min(1.0, q_tech))
```

### Aesthetic Score Computation
```python
def compute_aesthetic_score(q_tech, saliency_features):
    neg_space = saliency_features.get("neg_space_ratio", 0.3)
    aesthetic = 0.6 + 0.3 * (q_tech + neg_space)
    return max(0.0, min(1.0, aesthetic))
```

### Score Normalization
- **Method:** Clamp to [0.0, 1.0] range
- **Purpose:** Prevent outliers and ensure consistency
- **Implementation:** `max(0.0, min(1.0, score))`

## Caching Strategy

### Cache Key Generation
- **Algorithm:** MD5 hash of `{photo_id}_scores`
- **Purpose:** Avoid recomputation for same features
- **Scope:** Per-photo scoring results

### Cache Structure
```json
{
  "photo_id": "a6d54a6157e85c3fce236acca72d7f67...",
  "scores": {...},
  "features_hash": "md5_of_input_features",
  "timestamp": "2025-09-20T14:37:03Z"
}
```

## Error Handling

### Missing Features
- **Default Values:** Use reasonable defaults for missing features
- **Logging:** Warn about missing feature data
- **Graceful Degradation:** Continue processing with partial data

### Invalid Data Types
- **Type Checking:** Validate feature data types
- **Fallback Values:** Use defaults for invalid data
- **Error Logging:** Record data quality issues

### Computation Errors
- **Exception Handling:** Catch and log scoring errors
- **Fallback Scores:** Return neutral scores on failure
- **Recovery:** Continue processing other photos

## Performance Characteristics

### Processing Speed
- **Per-Photo Time:** 0.01-0.05 seconds
- **Batch Scaling:** Linear with photo count
- **Memory Usage:** Minimal (feature data only)

### Optimization Features
- **Caching:** 90%+ hit rate for repeated processing
- **Batch Processing:** Efficient vectorized operations
- **Memory Efficient:** No image loading required

## Dependencies

- **Core Libraries:** Built-in Python (no external dependencies)
- **Data Structures:** JSON for caching and serialization
- **Hashing:** hashlib for cache key generation

## Logging and Monitoring

### Log Files
- **Service Logs:** `intermediateJsons/scoring/scoring_service.log`
- **Batch Outputs:** `intermediateJsons/scoring/{batch_id}_scoring_output.json`

### Performance Metrics
- **Processing Time:** Per-batch and total timing
- **Cache Performance:** Hit rates and cache size
- **Quality Distribution:** Score histograms and statistics
- **Drop Rates:** Technical gate failure percentages

### Quality Metrics
- **Score Distributions:** Mean, median, standard deviation
- **Gate Statistics:** Photos dropped vs passed
- **Feature Coverage:** Percentage of photos with complete features

## Configuration Options

### Quality Thresholds
```python
TECH_QUALITY_THRESHOLD = 0.3  # Minimum Q_tech to pass
SCORE_BOUNDS = (0.0, 1.0)     # Min/max score range
```

### Scoring Weights
```python
SCORING_WEIGHTS = {
    "q_tech": {
        "sharpness": 0.4,
        "exposure": 0.4,
        "noise": 0.2
    }
}
```

## Integration Points

### Upstream Services
- **Features Service:** Primary input source with extracted features
- **Fallback Input:** Can process basic feature data for testing

### Downstream Services
- **Clustering Service:** Uses scores for quality-based grouping
- **Ranking Service:** Provides baseline scores for LLM comparison
- **Optimizer Service:** Leverages all scoring dimensions

## Score Interpretation

### Technical Quality Ranges
- **0.0-0.3:** Unusable (dropped automatically)
- **0.3-0.6:** Poor quality, major issues
- **0.6-0.8:** Good quality, minor issues
- **0.8-1.0:** Excellent quality, professional grade

### Aesthetic Score Ranges
- **0.0-0.4:** Low visual appeal
- **0.4-0.7:** Moderate appeal
- **0.7-1.0:** High visual appeal

### Use in Curation
- **Filtering:** Technical gate removes inadequate photos
- **Ranking:** Preliminary ordering before LLM assessment
- **Diversity:** Scoring dimensions inform selection algorithms

## Best Practices

### Score Calibration
1. **Ground Truth:** Validate scores against human judgments
2. **Consistency:** Ensure reproducible results across runs
3. **Balance:** Weight scoring dimensions appropriately
4. **Transparency:** Document scoring formulas and rationale

### Quality Assurance
1. **Monitoring:** Track score distributions over time
2. **Validation:** Cross-check scores with feature data
3. **Updates:** Plan for algorithm refinements
4. **Audit:** Maintain scoring decision logs

### Performance Optimization
1. **Caching Strategy:** Maximize cache effectiveness
2. **Batch Processing:** Optimize for large photo collections
3. **Memory Management:** Efficient data structures
4. **Parallelization:** Support concurrent scoring

## Use Cases

### Quality Filtering
- **Input:** Large batch with mixed quality photos
- **Processing:** Apply technical quality gate
- **Output:** High-quality subset for curation

### Preliminary Ranking
- **Input:** Feature-extracted photo collection
- **Processing:** Compute all scoring dimensions
- **Output:** Quality-ranked photos for human review

### Curation Pipeline Integration
- **Input:** Features from automated extraction
- **Processing:** Multi-dimensional quality assessment
- **Output:** Structured scores for optimization algorithms

## Future Enhancements

### Advanced Scoring
- **Machine Learning:** Train scoring models on human preferences
- **Theme-Specific:** Customize scoring for different content types
- **Context-Aware:** Consider collection context in scoring

### Feature Integration
- **Additional Metrics:** Incorporate new feature types
- **Dynamic Weights:** Adjust scoring based on use case
- **Personalization:** User-specific scoring preferences

### Quality Improvements
- **Calibration:** Regular validation against ground truth
- **A/B Testing:** Compare different scoring algorithms
- **Feedback Loop:** Learn from curation decisions

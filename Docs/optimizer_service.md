# Optimizer Service Documentation

## Overview

The Optimizer Service performs intelligent photo selection with diversity optimization. It serves as the final curation layer that balances quality, diversity, and thematic requirements to produce the optimal photo subset. Currently implemented with a simple cluster winner selection approach that can be enhanced with sophisticated submodular optimization algorithms.

## Key Responsibilities

1. **Photo Selection:** Choose optimal photos from cluster winners
2. **Diversity Optimization:** Ensure variety across multiple dimensions
3. **Role Assignment:** Classify photos by layout function (opener, anchor, etc.)
4. **Coverage Analysis:** Track representation across categories
5. **Marginal Gain Calculation:** Measure selection quality improvements
6. **Constraint Satisfaction:** Meet thematic and layout requirements

## Selection Strategy

### Current Implementation (Simple Selection)
**Algorithm:** Cluster winner collection with size limits
- **Input:** Cluster winners from ranking service
- **Selection:** Top cluster heroes + alternates
- **Limit:** Maximum 80 photos for manageability
- **Purpose:** Basic winner collection

### Planned Implementation (Submodular Optimization)
**Algorithm:** Facility location or saturation functions
- **Objective:** Maximize diversity subject to quality constraints
- **Constraints:** Theme requirements, layout quotas
- **Optimization:** Greedy selection with marginal gains
- **Advantage:** Mathematically optimal diversity

## Processing Workflow

### 1. Input Processing
- Receives cluster winners from ranking service
- Extracts photo IDs and ranking information
- Validates input data and cluster structure

### 2. Selection Algorithm
- Apply optimization strategy to select photos
- Balance quality and diversity objectives
- Respect thematic and layout constraints

### 3. Role Classification
- Assign layout roles to selected photos
- Calculate role probabilities and confidence
- Prepare metadata for layout service integration

### 4. Diversity Analysis
- Compute coverage across multiple dimensions
- Calculate marginal gains for selection decisions
- Generate optimization statistics and metrics

## Input Format

```json
{
  "batch_id": "batch_20250920_143703",
  "cluster_winners": [
    {
      "cluster_id": "m_0001",
      "hero": "a6d54a6157e85c3fce236acca72d7f67...",
      "alternates": ["b8c92d4e6f1a3h5j...", "c9d38e5f7g2b4i6k..."]
    },
    {
      "cluster_id": "m_0002",
      "hero": "d0e49f6g8h3c5j7l...",
      "alternates": []
    }
  ]
}
```

## Output Format

```json
{
  "batch_id": "batch_20250920_143703",
  "selected_ids": [
    "a6d54a6157e85c3fce236acca72d7f67...",
    "b8c92d4e6f1a3h5j...",
    "d0e49f6g8h3c5j7l..."
  ],
  "roles": [
    {
      "photo_id": "a6d54a6157e85c3fce236acca72d7f67...",
      "role": "opener",
      "prob": 0.78
    },
    {
      "photo_id": "b8c92d4e6f1a3h5j...",
      "role": "body",
      "prob": 0.65
    }
  ],
  "coverage": {
    "scene_type": 0.85,
    "palette_cluster": 0.78,
    "time_of_day": 0.82,
    "location_cluster": 0.75,
    "people_count": 0.80,
    "orientation": 0.95
  },
  "marginal_gains": {
    "a6d54a6157e85c3fce236acca72d7f67...": 0.043,
    "b8c92d4e6f1a3h5j...": 0.038,
    "d0e49f6g8h3c5j7l...": 0.052
  }
}
```

## Optimization Algorithm Details

### Current Simple Selection
```python
def select_photos(cluster_winners):
    selected_ids = []

    # Collect all candidates
    for winner in cluster_winners:
        selected_ids.append(winner["hero"])
        selected_ids.extend(winner["alternates"])

    # Apply size limit
    num_select = min(80, len(selected_ids))
    return selected_ids[:num_select]
```

### Future Submodular Optimization
**Objective Function:**
```
maximize: ∑ quality(photo) + λ × diversity(coverage)
subject to: |selection| ≤ max_size
           coverage ≥ theme_requirements
```

**Greedy Algorithm:**
1. Start with empty selection
2. At each step, add photo with highest marginal gain
3. Update coverage and quality metrics
4. Repeat until constraints satisfied

## Coverage Dimensions

### Scene Type Coverage
- **Categories:** urban, nature, portrait, architecture, etc.
- **Measurement:** Percentage of scene types represented
- **Importance:** Ensures variety in subject matter

### Palette Cluster Coverage
- **Categories:** warm, cool, monochromatic, complementary, etc.
- **Measurement:** Color harmony diversity
- **Importance:** Visual cohesion and interest

### Time of Day Coverage
- **Categories:** dawn, morning, noon, afternoon, dusk, night
- **Measurement:** Temporal distribution
- **Importance:** Story progression and mood variety

### People Count Coverage
- **Categories:** solo, couple, small group, large group, none
- **Measurement:** Social context diversity
- **Importance:** Relationship and scale variety

### Orientation Coverage
- **Categories:** landscape, portrait, square
- **Measurement:** Aspect ratio distribution
- **Importance:** Layout flexibility

## Role Classification

### Layout Roles
- **Opener:** First photo, sets tone and context
- **Anchor:** Final photo, provides conclusion
- **Hero:** Main focal photos, highest visual impact
- **Body:** Supporting photos, maintain flow
- **Connector:** Transitional photos between sections

### Role Assignment Criteria
- **Quality:** Higher quality photos get prominent roles
- **Diversity:** Spread different types across roles
- **Flow:** Consider visual progression
- **Theme:** Match thematic requirements

## Marginal Gain Calculation

### Definition
- **Marginal Gain:** Improvement in objective function from adding photo
- **Formula:** `gain = quality(photo) + λ × coverage_improvement`
- **Purpose:** Quantify selection value

### Usage
- **Optimization:** Guide greedy selection algorithm
- **Audit:** Explain selection decisions
- **Sensitivity:** Assess robustness of selections

## Caching Strategy

### Cache Key Generation
- **Algorithm:** MD5 hash of `{batch_id}_optimization`
- **Purpose:** Cache optimization results for same inputs
- **Scope:** Per-batch optimization results

### Cache Structure
```json
{
  "batch_id": "batch_20250920_143703",
  "selected_ids": [...],
  "roles": [...],
  "coverage": {...},
  "marginal_gains": {...},
  "algorithm": "simple_selection_v1",
  "timestamp": "2025-09-20T14:37:03Z"
}
```

## Performance Characteristics

### Current Implementation
- **Processing Time:** O(n) linear time
- **Memory Usage:** O(n) for photo storage
- **Scalability:** Handles hundreds of photos efficiently

### Future Optimization
- **Complexity:** O(n²) for full optimization (can be approximated)
- **Approximation:** Greedy algorithms with theoretical guarantees
- **Scalability:** Handle thousands of photos with approximation

## Error Handling

### Missing Cluster Winners
- **Default Behavior:** Return empty selection
- **Logging:** Warn about incomplete input data
- **Graceful Degradation:** Continue with available data

### Optimization Failures
- **Fallback:** Return simple selection without optimization
- **Recovery:** Continue processing with degraded optimization
- **Monitoring:** Track optimization success rates

### Constraint Violations
- **Validation:** Check coverage and size constraints
- **Correction:** Adjust selection to meet requirements
- **Reporting:** Log constraint satisfaction status

## Dependencies

### Current Implementation
- **Core Libraries:** Built-in Python (json, hashlib, random)
- **Data Structures:** Lists and dictionaries
- **No External Dependencies:** Pure Python implementation

### Future Implementation
- **NumPy:** Matrix operations for optimization
- **scipy:** Optimization algorithms
- **scikit-learn:** Diversity metrics and clustering

## Logging and Monitoring

### Log Files
- **Service Logs:** `intermediateJsons/optimizer/optimizer_service.log`
- **Batch Outputs:** `intermediateJsons/optimizer/{batch_id}_optimizer_output.json`

### Performance Metrics
- **Selection Statistics:** Number selected, roles assigned
- **Coverage Metrics:** Average and per-dimension coverage
- **Processing Time:** Optimization algorithm performance
- **Cache Performance:** Hit rates and storage efficiency

### Optimization Metrics
- **Marginal Gains:** Distribution of selection values
- **Role Distribution:** Balance across layout roles
- **Quality Retention:** Average quality of selected photos
- **Diversity Achievement:** Coverage goal attainment

## Configuration Options

### Selection Limits
```python
MAX_SELECTION_SIZE = 80      # Maximum photos to select
MIN_SELECTION_SIZE = 10      # Minimum photos to select
```

### Optimization Parameters
```python
OPTIMIZATION_CONFIG = {
    "diversity_weight": 0.3,    # λ in objective function
    "coverage_threshold": 0.7,  # Minimum coverage required
    "quality_weight": 0.7,      # Quality importance
}
```

## Integration Points

### Upstream Services
- **Ranking Service:** Primary input with cluster winners
- **Clustering Service:** Provides cluster structure
- **Theme Spec:** Optional theme constraints and requirements

### Downstream Services
- **Exporter Service:** Receives final selection for output
- **Layout Service:** Uses roles for intelligent page layout
- **Human Review:** Presents optimized selection for approval

## Quality Assessment

### Selection Quality Metrics
- **Quality Retention:** Average score of selected photos
- **Diversity Score:** Coverage-weighted diversity measure
- **Role Balance:** Distribution across layout functions
- **Marginal Efficiency:** Gains per photo added

### Comparative Analysis
- **Baseline:** Random selection performance
- **Upper Bound:** Optimal (exhaustive) selection
- **Efficiency:** Approximation quality vs. computation time

## Best Practices

### Optimization Strategy
1. **Objective Balance:** Weight quality and diversity appropriately
2. **Constraint Handling:** Meet hard requirements first
3. **Approximation Quality:** Choose algorithms with good guarantees
4. **Scalability:** Handle large photo collections efficiently

### Quality Assurance
1. **Ground Truth:** Validate selections against human preferences
2. **Consistency:** Ensure reproducible results across runs
3. **Monitoring:** Track optimization metrics over time
4. **Feedback Loop:** Learn from curation decisions

### Performance Optimization
1. **Incremental Processing:** Handle new photos efficiently
2. **Approximation Methods:** Use fast approximation algorithms
3. **Caching Strategy:** Maximize cache effectiveness
4. **Memory Management:** Efficient data structures

## Use Cases

### Magazine Layout
- **Input:** Large photo collection for article
- **Optimization:** Select diverse, high-quality subset
- **Output:** Photos with layout roles assigned

### Portfolio Curation
- **Input:** Photographer's work collection
- **Optimization:** Maximize visual diversity and impact
- **Output:** Optimal portfolio subset

### Event Coverage
- **Input:** Event photography with many similar shots
- **Optimization:** Balance moment coverage and quality
- **Output:** Comprehensive yet concise event story

## Future Enhancements

### Advanced Algorithms
- **Machine Learning:** Learn user preferences for optimization
- **Multi-Objective:** Optimize multiple criteria simultaneously
- **Contextual:** Consider layout and storytelling constraints

### Integration Features
- **Real-time:** Interactive optimization with user feedback
- **Personalization:** User-specific optimization preferences
- **Thematic:** Theme-aware selection and role assignment

### Quality Improvements
- **A/B Testing:** Compare different optimization strategies
- **User Studies:** Validate optimization against human judgment
- **Continuous Learning:** Improve algorithms based on outcomes

# Clustering Service Documentation

## Overview

The Clustering Service groups near-duplicate and similar photos into moment clusters. It serves as a critical deduplication layer that prevents redundant photos from cluttering the curation process while preserving the best representatives from each moment or scene. Currently implemented with a simple quality-based clustering approach that can be enhanced with perceptual hashing for true near-duplicate detection.

## Key Responsibilities

1. **Near-Duplicate Detection:** Identify photos of the same moment/scene
2. **Moment Clustering:** Group photos by temporal and spatial similarity
3. **Quality-Based Selection:** Prefer higher quality photos within clusters
4. **Deduplication:** Reduce redundant photos while preserving variety
5. **Cluster Management:** Assign cluster IDs and track member relationships

## Clustering Strategy

### Current Implementation (Quality-Based)
**Algorithm:** Simple quality range grouping
- **High Quality:** Total_prelim ≥ 0.7
- **Medium Quality:** Total_prelim 0.5-0.7
- **Low Quality:** Total_prelim 0.3-0.5
- **Purpose:** Basic quality stratification
- **Limitation:** Not true moment clustering

### Planned Implementation (Perceptual Hashing)
**Algorithm:** pHash-based similarity clustering
- **Similarity Metric:** Hamming distance between perceptual hashes
- **Threshold:** Configurable similarity threshold
- **Clustering:** DBSCAN or HDBSCAN for density-based grouping
- **Advantage:** True near-duplicate detection

## Processing Workflow

### 1. Input Processing
- Receives scored photos from scoring service
- Extracts photo IDs and quality scores
- Validates input data completeness

### 2. Clustering Algorithm
- Apply clustering strategy to group similar photos
- Generate unique cluster IDs
- Track cluster membership and statistics

### 3. Quality Optimization
- Within each cluster, identify highest quality photos
- Prepare cluster winners for ranking service
- Maintain cluster metadata and statistics

### 4. Output Generation
- Structure clusters with IDs and member lists
- Include clustering metadata and statistics
- Generate audit trail for cluster decisions

## Input Format

```json
{
  "batch_id": "batch_20250920_143703",
  "scores": [
    {
      "photo_id": "a6d54a6157e85c3fce236acca72d7f67...",
      "Q_tech": 0.82,
      "Total_prelim": 0.76
    },
    {
      "photo_id": "b8c92d4e6f1a3h5j...",
      "Q_tech": 0.65,
      "Total_prelim": 0.68
    }
  ]
}
```

## Output Format

```json
{
  "batch_id": "batch_20250920_143703",
  "clusters": [
    {
      "cluster_id": "m_0001",
      "members": [
        "a6d54a6157e85c3fce236acca72d7f67...",
        "b8c92d4e6f1a3h5j...",
        "c9d38e5f7g2b4i6k..."
      ]
    },
    {
      "cluster_id": "m_0002",
      "members": [
        "d0e49f6g8h3c5j7l...",
        "e1f5a7h9i4d6k8m..."
      ]
    }
  ]
}
```

## Clustering Algorithm Details

### Current Quality-Based Clustering
```python
def _create_clusters(scores):
    clusters = []
    cluster_id = 1

    # Define quality ranges
    quality_ranges = [(0.7, 1.0), (0.5, 0.7), (0.3, 0.5)]

    for min_q, max_q in quality_ranges:
        cluster_photos = [
            score["photo_id"] for score in scores
            if min_q <= score.get("Total_prelim", 0.5) < max_q
        ]

        if cluster_photos:
            clusters.append({
                "cluster_id": "02d",
                "members": cluster_photos
            })
            cluster_id += 1

    return clusters
```

### Future pHash-Based Clustering
**Similarity Function:**
```python
def hamming_distance(hash1, hash2):
    return bin(int(hash1, 16) ^ int(hash2, 16)).count('1')

def are_similar(photo1, photo2, threshold=5):
    distance = hamming_distance(photo1['phash'], photo2['phash'])
    return distance <= threshold
```

**Clustering Process:**
1. Extract pHash from features
2. Calculate pairwise similarities
3. Apply DBSCAN clustering
4. Assign cluster IDs and members

## Cluster ID Convention

### Format
- **Pattern:** `m_XXXX` (moment cluster)
- **Example:** `m_0001`, `m_0002`, etc.
- **Padding:** Zero-padded to 4 digits
- **Namespace:** `m_` prefix for moment clusters

### Future Extensions
- **Scene Clusters:** `s_XXXX` for scene-based grouping
- **Time Clusters:** `t_XXXX` for temporal grouping
- **Quality Clusters:** `q_XXXX` for quality-based grouping

## Caching Strategy

### Cache Key Generation
- **Algorithm:** MD5 hash of `{batch_id}_clusters`
- **Purpose:** Cache clustering results for same photo batches
- **Scope:** Per-batch clustering results

### Cache Structure
```json
{
  "batch_id": "batch_20250920_143703",
  "clusters": [...],
  "algorithm": "quality_based_v1",
  "timestamp": "2025-09-20T14:37:03Z",
  "stats": {
    "total_clusters": 5,
    "total_photos": 127,
    "avg_cluster_size": 25.4
  }
}
```

### Cache Invalidation
- **Photo Changes:** New photos or score changes
- **Algorithm Updates:** Clustering method improvements
- **Parameter Changes:** Similarity thresholds or quality ranges

## Performance Characteristics

### Current Implementation
- **Processing Time:** O(n) linear time
- **Memory Usage:** O(n) for photo storage
- **Scalability:** Handles thousands of photos efficiently

### Future pHash Implementation
- **Complexity:** O(n²) for pairwise comparisons (can be optimized)
- **Memory:** O(n) for hash storage
- **Optimization:** Approximate nearest neighbor search

## Error Handling

### Missing Scores
- **Default Values:** Use neutral scores for missing data
- **Logging:** Warn about incomplete scoring data
- **Graceful Handling:** Continue clustering with available data

### Empty Clusters
- **Filtering:** Remove clusters with no members
- **Validation:** Ensure cluster integrity
- **Logging:** Report clustering statistics

### Algorithm Failures
- **Fallback:** Return single-member clusters
- **Recovery:** Continue processing with degraded clustering
- **Monitoring:** Track clustering success rates

## Dependencies

### Current Implementation
- **Core Libraries:** Built-in Python (json, hashlib)
- **Data Structures:** Lists and dictionaries
- **No External Dependencies:** Pure Python implementation

### Future Implementation
- **scikit-learn:** DBSCAN/HDBSCAN clustering
- **NumPy:** Distance calculations and matrix operations
- **scipy:** Spatial algorithms for optimization

## Logging and Monitoring

### Log Files
- **Service Logs:** `intermediateJsons/clustering/clustering_service.log`
- **Batch Outputs:** `intermediateJsons/clustering/{batch_id}_clustering_output.json`

### Performance Metrics
- **Cluster Statistics:** Size distribution, member counts
- **Processing Time:** Clustering algorithm performance
- **Cache Performance:** Hit rates and storage efficiency
- **Quality Distribution:** Photos per quality range

### Clustering Metrics
- **Total Clusters:** Number of clusters created
- **Average Cluster Size:** Mean photos per cluster
- **Largest/Smallest Clusters:** Size distribution analysis
- **Singleton Clusters:** Clusters with single photos

## Configuration Options

### Quality Ranges
```python
QUALITY_RANGES = [
    (0.7, 1.0),  # High quality
    (0.5, 0.7),  # Medium quality
    (0.3, 0.5),  # Low quality
]
```

### Future pHash Configuration
```python
PHASH_CONFIG = {
    "similarity_threshold": 5,    # Hamming distance threshold
    "min_cluster_size": 2,        # Minimum cluster members
    "algorithm": "dbscan",        # Clustering algorithm
    "eps": 3.0,                   # DBSCAN epsilon parameter
}
```

## Integration Points

### Upstream Services
- **Scoring Service:** Primary input with quality scores
- **Features Service:** Provides pHash for similarity clustering

### Downstream Services
- **Ranking Service:** Receives cluster winners for LLM ranking
- **Optimizer Service:** Uses cluster structure for diversity optimization
- **Human Review:** Presents cluster representatives for selection

## Cluster Quality Assessment

### Current Metrics
- **Cluster Size:** Number of photos per cluster
- **Quality Distribution:** Score ranges within clusters
- **Representative Selection:** Highest scoring photo per cluster

### Future Enhancements
- **Cluster Coherence:** Measure cluster similarity tightness
- **Quality Variance:** Assess quality spread within clusters
- **Representative Quality:** Best photo selection algorithms

## Best Practices

### Clustering Strategy
1. **Similarity Thresholds:** Tune for appropriate cluster granularity
2. **Quality Preservation:** Prefer higher quality photos as representatives
3. **Balance:** Avoid too many singletons or huge clusters
4. **Validation:** Cross-check clusters with human judgment

### Performance Optimization
1. **Incremental Processing:** Handle new photos efficiently
2. **Approximate Methods:** Use ANN for large-scale clustering
3. **Caching Strategy:** Maximize cache effectiveness
4. **Memory Management:** Efficient data structures for large batches

### Quality Assurance
1. **Ground Truth:** Validate clusters against known duplicates
2. **Consistency:** Ensure reproducible clustering results
3. **Monitoring:** Track cluster quality metrics over time
4. **Feedback Loop:** Learn from curation decisions

## Use Cases

### Event Photography
- **Input:** Multiple shots of same moments at events
- **Clustering:** Group near-duplicate photos
- **Output:** One representative per moment

### Product Photography
- **Input:** Multiple angles of same products
- **Clustering:** Group similar product shots
- **Output:** Diverse product representations

### Travel Photography
- **Input:** Multiple photos of same locations
- **Clustering:** Group location-based clusters
- **Output:** Best photo per location/moment

## Future Enhancements

### Advanced Algorithms
- **Deep Learning:** CNN-based similarity detection
- **Temporal Clustering:** Time-based grouping
- **Semantic Clustering:** Content-based grouping beyond pHash

### Integration Features
- **Multi-Modal:** Combine visual, temporal, and metadata similarity
- **Hierarchical:** Multi-level clustering (scenes → moments → duplicates)
- **Interactive:** Human-guided cluster refinement

### Quality Improvements
- **Personalization:** User-specific similarity preferences
- **Context Awareness:** Event-type specific clustering
- **Quality Learning:** ML-based cluster quality assessment

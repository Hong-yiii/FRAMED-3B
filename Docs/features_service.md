# Features Service Documentation

## Overview

The Features Service extracts rich visual and technical features from processed photos. It serves as the analytical core of the curation pipeline, converting raw pixel data into structured feature vectors that enable intelligent photo assessment and comparison. Currently implemented with mock data generators that simulate real feature extraction algorithms.

## Key Responsibilities

1. **Visual Embeddings:** Generate semantic similarity vectors using CLIP models
2. **Perceptual Hashing:** Create content-based hashes for duplicate detection
3. **Technical Quality:** Assess image sharpness, exposure, and noise levels
4. **Saliency Analysis:** Detect visual attention regions and negative space
5. **Face Detection:** Identify and analyze facial features and landmarks
6. **Color Analysis:** Extract color palettes and clustering information

## Processing Workflow

### 1. Input Processing
- Receives standardized photo artifacts from preprocess service
- Handles both artifact and photo_index input formats
- Validates image file existence and accessibility

### 2. Feature Extraction Pipeline
- **Image Loading:** Open standardized images with PIL
- **Basic Properties:** Extract dimensions, aspect ratios, orientation
- **Feature Computation:** Apply various analysis algorithms
- **Quality Assessment:** Evaluate technical image characteristics

### 3. Caching and Optimization
- Content-based caching using photo IDs
- Avoid recomputation of expensive features
- Incremental processing for large batches

### 4. Output Generation
- Structured feature dictionaries
- File path references for large data (embeddings, heatmaps)
- Comprehensive metadata and processing information

## Feature Types

### 1. Embeddings
**Purpose:** Semantic similarity and content understanding
- **CLIP L/14:** 768-dimensional vector from OpenAI CLIP model
- **Use Cases:** Content-based similarity, theme matching
- **Storage:** NumPy arrays in `./data/emb/` directory
- **Format:** `{photo_id}_clip.npy`

### 2. Perceptual Hashes (pHash)
**Purpose:** Near-duplicate detection and similarity comparison
- **Algorithm:** Discrete Cosine Transform-based hashing
- **Output:** 16-character hexadecimal string
- **Use Cases:** Moment clustering, duplicate removal
- **Properties:** Hamming distance correlates with visual similarity

### 3. Technical Quality Metrics
**Purpose:** Assess image capture quality and processing artifacts
- **Sharpness:** Edge clarity and focus quality (0-1 scale)
- **Exposure:** Brightness and contrast balance (0-1 scale)
- **Noise:** Image sensor noise and grain levels (0-1 scale)
- **Horizon Detection:** Camera tilt in degrees (±5° typical)

### 4. Saliency Analysis
**Purpose:** Understand visual attention and composition
- **Heatmap:** Visual attention probability map
- **Negative Space Ratio:** Proportion of low-interest areas (0-1)
- **Storage:** PNG heatmap files in `./data/sal/` directory
- **Use Cases:** Composition analysis, typography space detection

### 5. Face Detection
**Purpose:** Identify human subjects and facial characteristics
- **Count:** Number of detected faces (0-10+)
- **Landmarks:** Facial feature detection quality boolean
- **Use Cases:** People count categorization, portrait identification
- **Privacy:** No facial recognition or identification data stored

### 6. Color Palette Analysis
**Purpose:** Extract dominant colors and color harmony
- **LAB Centroids:** Color cluster centers in LAB color space
- **Cluster ID:** Palette category identifier (`pal_01` to `pal_10`)
- **Use Cases:** Theme matching, visual diversity assessment
- **Format:** Array of [L, a, b] color coordinates

## Input Format

### From Preprocess Service (Primary)
```json
{
  "batch_id": "batch_20250920_143703",
  "artifacts": [
    {
      "photo_id": "a6d54a6157e85c3fce236acca72d7f67...",
      "original_uri": "./data/input/photo1.jpg",
      "std_uri": "./data/rankingInput/a6d54a6157e85c3fce236acca72d7f67_1024.jpg",
      "processing_metadata": {...}
    }
  ]
}
```

### Direct Photo Index (Fallback)
```json
{
  "batch_id": "batch_20250920_143703",
  "photo_index": [
    {
      "photo_id": "a6d54a6157e85c3fce236acca72d7f67...",
      "uri": "./data/rankingInput/photo.jpg"
    }
  ]
}
```

## Output Format

```json
{
  "batch_id": "batch_20250920_143703",
  "artifacts": [
    {
      "photo_id": "a6d54a6157e85c3fce236acca72d7f67...",
      "original_uri": "./data/input/photo1.jpg",
      "std_uri": "./data/rankingInput/a6d54a6157e85c3fce236acca72d7f67_1024.jpg",
      "features": {
        "embeddings": {
          "clip_L14": "./data/emb/a6d54a6157e85c3fce236acca72d7f67_clip.npy"
        },
        "hashes": {
          "phash": "4f2a8b9c1e5d7f3a"
        },
        "tech": {
          "sharpness": 0.85,
          "exposure": 0.72,
          "noise": 0.15,
          "horizon_deg": -0.8
        },
        "saliency": {
          "heatmap_uri": "./data/sal/a6d54a6157e85c3fce236acca72d7f67.png",
          "neg_space_ratio": 0.34
        },
        "faces": {
          "count": 2,
          "landmarks_ok": true
        },
        "palette": {
          "lab_centroids": [
            [65.2, 12.8, -8.5],
            [42.1, -15.3, 22.7]
          ],
          "cluster_id": "pal_03"
        }
      }
    }
  ]
}
```

## Directory Structure

```
/data/
├── rankingInput/     # Input: Standardized images from preprocess
├── emb/             # Output: CLIP embedding NumPy arrays
├── sal/             # Output: Saliency heatmap PNG files
├── cache/features/  # Cache: Feature extraction results
/intermediateJsons/features/  # Service outputs and logs
```

## Caching Strategy

### Cache Key Generation
- **Algorithm:** MD5 hash of `{photo_id}_features`
- **Purpose:** Deterministic, content-addressed caching
- **Collision Resistance:** 128-bit hash space

### Cache Structure
```json
{
  "cache_key": "md5_hash",
  "timestamp": "2025-09-20T14:37:03Z",
  "features": {...},
  "version": "1.0"
}
```

### Cache Invalidation
- **Manual:** Delete cache files to force recomputation
- **Automatic:** Version-based invalidation for algorithm updates
- **Selective:** Per-photo cache clearing for updates

## Error Handling

### Image Processing Errors
- **Corrupt Files:** Return minimal feature set with defaults
- **Unsupported Formats:** Should be handled by preprocess service
- **Memory Issues:** Graceful degradation to basic features

### Feature Extraction Failures
- **Algorithm Errors:** Fallback to statistical defaults
- **Missing Dependencies:** Mock implementations for development
- **Timeout Issues:** Partial feature completion with warnings

## Performance Characteristics

### Processing Times
- **Basic Features:** 0.1-0.5 seconds per image
- **Advanced Features:** 1-5 seconds per image (with real implementations)
- **Batch Scaling:** Linear with image count, parallelizable

### Memory Usage
- **Per-Image Memory:** 50-200MB depending on image size
- **Caching Benefits:** 90%+ reduction in repeated processing
- **Optimization:** Streaming processing for large batches

## Dependencies

### Current Implementation (Mock)
- **PIL/Pillow:** Image loading and basic properties
- **NumPy:** Random data generation and array operations
- **Hashlib:** MD5 hashing for cache keys and pHash simulation

### Planned Real Implementation
- **CLIP:** OpenAI CLIP model for embeddings
- **OpenCV:** Computer vision algorithms
- **scikit-image:** Advanced image processing
- **face-recognition:** Facial detection and analysis
- **colorthief/palette-extraction:** Color analysis algorithms

## Logging and Monitoring

### Log Files
- **Service Logs:** `intermediateJsons/features/features_service.log`
- **Batch Outputs:** `intermediateJsons/features/{batch_id}_features_output.json`

### Performance Metrics
- **Cache Hit Rate:** Percentage of cached vs computed features
- **Processing Time:** Per-image and total batch times
- **Feature Coverage:** Success rates for each feature type
- **Error Rates:** Failed extractions by category

## Configuration Options

### Feature Selection
```python
ENABLED_FEATURES = {
    "embeddings": True,
    "hashes": True,
    "technical": True,
    "saliency": True,
    "faces": True,
    "palette": True
}
```

### Quality Thresholds
- **Minimum Confidence:** Skip low-confidence feature extractions
- **Processing Timeout:** Maximum time per image (30 seconds)
- **Memory Limits:** Per-image memory constraints

## Integration Points

### Upstream Services
- **Preprocess Service:** Primary input source with standardized images
- **Direct Processing:** Can consume raw photo index for testing

### Downstream Services
- **Scoring Service:** Consumes all feature types for quality assessment
- **Clustering Service:** Uses pHash for duplicate detection
- **Ranking Service:** Leverages embeddings and technical features

## Development Status

### Current Implementation
- ✅ **Mock Framework:** Complete feature extraction skeleton
- ✅ **Caching System:** Full cache implementation with MD5 keys
- ✅ **Error Handling:** Robust error recovery and fallback
- ✅ **Data Structures:** Complete JSON schema and file formats

### Planned Enhancements
- 🔄 **Real CLIP Integration:** Replace mock embeddings with actual CLIP model
- 🔄 **OpenCV Pipeline:** Implement real computer vision algorithms
- 🔄 **Face Detection:** Add proper facial analysis
- 🔄 **Saliency Maps:** Generate actual attention heatmaps

## Best Practices

### Feature Engineering
1. **Normalization:** Ensure all features are properly scaled
2. **Robustness:** Handle edge cases and unusual images
3. **Versioning:** Track feature extraction algorithm versions
4. **Validation:** Verify feature quality and consistency

### Performance Optimization
1. **Caching Strategy:** Maximize cache hit rates
2. **Batch Processing:** Process images in optimal batch sizes
3. **Parallelization:** Support concurrent feature extraction
4. **Resource Management:** Monitor memory and CPU usage

### Quality Assurance
1. **Ground Truth:** Validate features against human judgments
2. **Consistency:** Ensure reproducible results across runs
3. **Monitoring:** Track feature quality metrics over time
4. **Updates:** Plan for algorithm improvements and retraining

## Use Cases

### Content-Based Similarity
- **Input:** Large photo collections
- **Features:** CLIP embeddings + pHash
- **Output:** Semantic similarity clusters

### Quality Assessment
- **Input:** Mixed quality photo batch
- **Features:** Technical quality metrics
- **Output:** Quality rankings and filtering

### Visual Analysis
- **Input:** Professional photo shoots
- **Features:** Saliency + color palette + faces
- **Output:** Composition analysis and curation insights

### Duplicate Detection
- **Input:** Event photography with multiple shots
- **Features:** Perceptual hashing
- **Output:** Near-duplicate groups for selection

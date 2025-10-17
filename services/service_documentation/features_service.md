---

# OpenCLIP Features Service (Framed Pipeline Integration)

**Goal:** Extract rich visual and technical features from photos using **OpenCLIP** and computer vision techniques. This service is integrated into the curation pipeline and processes batches of photos.

**Input:** Preprocessed photo artifacts from the preprocess service
**Output:** Feature-enriched artifacts with CLIP labels and technical quality metrics

## Pipeline Integration

This service integrates with the existing orchestrator pipeline:
- **Triggered by:** `preprocess.completed` event
- **Publishes:** `features.completed` event
- **Optimized for:** MacBook Pro with MPS (Metal Performance Shaders)

### Input Format

The service accepts two input formats:

**1. From Preprocess Service (Preferred):**
```json
{
  "batch_id": "batch_2025-01-15_test",
  "artifacts": [
    {
      "photo_id": "abed1315713f44e8c76ba97152ec25d788a02f36ec64b7858b5d00c7fb08e9ea",
      "original_uri": "./data/input/30ED3B7D-090E-485E-A3B7-A3A04F816B2E.jpg",
      "ranking_uri": "./data/rankingInput/abed1315713f44e8c76ba97152ec25d788a02f36ec64b7858b5d00c7fb08e9ea.jpg",
      "std_uri": "./data/rankingInput/abed1315713f44e8c76ba97152ec25d788a02f36ec64b7858b5d00c7fb08e9ea_1024.jpg"
    }
  ]
}
```

**2. Direct from Ingest Service (Fallback):**
```json
{
  "batch_id": "batch_20251016_132943",
  "photo_index": [
    {
      "photo_id": "f399832e651bcecc870e4daac76c389f117bd34f2543acd0e596d561128dad47",
      "original_uri": "./data/input/07C58076-AE31-41F3-878E-0B5837667F80.jpg",
      "ranking_uri": "./data/rankingInput/f399832e651bcecc870e4daac76c389f117bd34f2543acd0e596d561128dad47.jpg",
      "exif": {
        "camera": "Apple iPhone 13 Pro Max",
        "lens": "Unknown",
        "iso": 50,
        "aperture": "f/1.5",
        "shutter_speed": "1/5587",
        "focal_length": "5mm",
        "datetime": "2024:12:23 12:14:29",
        "gps": null
      },
      "format": ".jpg"
    }
  ]
}
```

### Output Format
```json
{
  "batch_id": "batch_2025-01-15_test",
  "artifacts": [
    {
      "photo_id": "abed1315713f44e8c76ba97152ec25d788a02f36ec64b7858b5d00c7fb08e9ea",
      "std_uri": "./data/rankingInput/abed1315713f44e8c76ba97152ec25d788a02f36ec64b7858b5d00c7fb08e9ea_1024.jpg",
      "original_uri": "./data/input/30ED3B7D-090E-485E-A3B7-A3A04F816B2E.jpg",
      "ranking_uri": "./data/rankingInput/abed1315713f44e8c76ba97152ec25d788a02f36ec64b7858b5d00c7fb08e9ea.jpg",
      "exif": {
        "camera": "Apple iPhone 11 Pro Max",
        "iso": 125,
        "aperture": "f/2.4",
        "shutter_speed": "1/100"
      },
      "features": {
        "tech": {
          "sharpness": 0.5824664430670188,
          "exposure": 0.7642725303263522,
          "noise": 0.15978705996824694,
          "horizon_deg": 0.6372531198368598,
          "iso_noise_factor": 0.039,
          "aperture_f_number": 2.4,
          "shutter_speed_seconds": 0.01,
          "camera_type": "Apple iPhone 11 Pro Max",
          "camera_tier": "smartphone"
        },
        "clip_labels": [
          "photography",
          "landscape",
          "nature",
          "outdoor",
          "scenic"
        ]
      }
    }
  ]
}
```

## Implementation Details

* **Framework:** Python 3.13+, integrated with existing pipeline
* **Model lib:** **open_clip_torch**
* **Primary checkpoint:** `hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K` (good accuracy, efficient)
* **Device:** MPS (Metal Performance Shaders) on MacBook Pro, CUDA fallback, then CPU
* **Precision:** FP16 on GPU/MPS for memory efficiency
* **Image processing:** Uses standardized images from preprocess service
* **Labels:** read from `config/labels.json` (50 photography-focused labels)
* **Prompt templates:** 6 templates per label; averaged text embeddings
* **Return:** top 5 CLIP labels with probabilities and cosine scores
* **Cache:** label embeddings cached to disk; technical features cached per photo
* **Thread-safety:** model loaded once on service initialization
* **Logging:** structured logs for batch processing, feature extraction timing
* **Error handling:** graceful fallback to default labels on processing errors

## Core Components

### CLIPClassifier
- **Model Loading:** Automatic device detection (MPS > CUDA > CPU)
- **Label Embeddings:** Pre-computed and cached text embeddings for all labels
- **Image Classification:** Zero-shot classification with temperature-scaled probabilities
- **Caching:** Persistent embedding cache with model/config validation

### TechnicalQualityAnalyzer
- **Sharpness:** Laplacian variance-based sharpness measurement
- **Exposure:** Histogram analysis for over/under-exposure detection
- **Noise:** Gaussian blur difference for noise estimation
- **Horizon:** Hough line detection for horizon tilt analysis
- **EXIF Enhancement:** Additional insights from camera metadata

### 5. EXIF-Based Insights
- **ISO Noise Factor:** Predicted noise level based on ISO setting
- **Aperture Analysis:** F-number for depth of field context
- **Shutter Speed:** Motion blur prediction from exposure time
- **Camera Tier:** Classification (smartphone/consumer/professional)
- **Camera Type:** Specific camera model information

### FeaturesService (Main Interface)
- **Pipeline Integration:** Processes batch artifacts from preprocess service
- **Caching Strategy:** Per-photo feature caching with hash-based keys
- **Error Handling:** Graceful degradation with fallback features
- **Logging:** Comprehensive logging for debugging and monitoring

## File Structure

```
services/
  features_service.py          # Main service implementation
config/
  labels.json                  # CLIP classification labels
  templates.json              # Text prompt templates
data/cache/features/
  label_embeddings.pt         # Cached CLIP text embeddings
  {hash}.json                 # Per-photo feature cache
```

### Dependencies (added to pixi.toml)

```toml
# OpenCLIP and ML dependencies
pytorch = ">=2.1.0,<3"
torchvision = ">=0.16.0,<1"
open-clip-torch = ">=2.26.1,<3"
# Image processing and quality assessment
opencv = ">=4.8.0,<5"
scikit-image = ">=0.22.0,<1"
# Technical quality metrics
pyiqa = ">=0.1.7,<1"
```

### `config/labels.json`

```json
{
  "labels": [
    "photography", "landscape", "nature", "outdoor", "scenic",
    "portrait", "street photography", "architecture", "urban", "cityscape",
    "mountain", "forest", "ocean", "beach", "sunset", "sunrise", "night",
    "indoor", "people", "animal", "flower", "tree", "building", "sky",
    "cloud", "water", "road", "bridge", "park", "garden", "travel",
    "vacation", "food", "restaurant", "art", "museum", "concert",
    "festival", "sports", "action", "macro", "close-up", "wide shot",
    "panorama", "black and white", "colorful", "vintage", "modern",
    "abstract", "minimalist"
  ]
}
```

### `config/templates.json`

```json
{
  "templates": [
    "a photo of {}",
    "a photograph showing {}",
    "an image of {}",
    "a picture featuring {}",
    "a {} scene",
    "a {} photograph"
  ]
}
```

## Technical Implementation

### 1. Model Loading & Device Selection
- **Device Priority:** MPS (MacBook Pro) > CUDA > CPU
- **Model:** `hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K`
- **Precision:** FP16 on GPU/MPS for memory efficiency
- **Initialization:** Model loaded once on service startup

### 2. Label Embedding Cache
- **Text Encoding:** Each label processed with 6 prompt templates
- **Embedding Generation:** Text embeddings L2-normalized and averaged per label
- **Caching:** Embeddings cached to `label_embeddings.pt` with validation
- **Invalidation:** Cache rebuilt when labels/templates/model changes

### 3. Image Classification Pipeline
- **Preprocessing:** Uses model-specific transforms from OpenCLIP
- **Encoding:** Image embeddings L2-normalized for cosine similarity
- **Scoring:** Temperature-scaled softmax for probability distribution
- **Output:** Top-5 labels with probabilities and raw cosine scores

### 4. Technical Quality Analysis
- **Sharpness:** Laplacian variance with normalization
- **Exposure:** Histogram analysis for clipping detection
- **Noise:** Gaussian blur difference method
- **Horizon:** Hough line detection for tilt measurement

### 5. Caching Strategy
- **Feature Cache:** Per-photo hash-based caching in JSON format
- **Cache Key:** MD5 hash of `{photo_id}_features`
- **Invalidation:** Manual cache cleanup (no automatic expiry)

### 6. Error Handling
- **Graceful Degradation:** Default features on processing errors
- **Logging:** Comprehensive error logging with context
- **Fallback Labels:** Photography-focused default labels

## Usage in Pipeline

### Service Integration
The FeaturesService integrates seamlessly with the existing orchestrator:

```python
# In services/__init__.py
from .features_service import FeaturesService

class RealServices:
    def __init__(self):
        self.features_service = FeaturesService()
    
    def process_features_service(self, input_data):
        return self.features_service.process(input_data)
```

### Running the Pipeline
```bash
# Install dependencies
pixi install

# Place photos in data/input/
# Run the full pipeline
python orchestrator.py
```

## Performance Characteristics

### MacBook Pro Optimization
- **MPS Acceleration:** Leverages Metal Performance Shaders for GPU acceleration
- **Memory Efficiency:** FP16 precision reduces memory usage by ~50%
- **Batch Processing:** Optimized for single-image processing with low latency

### Expected Performance
- **Model Loading:** ~10-15 seconds on first run
- **Label Embedding Cache:** ~5-10 seconds for 50 labels
- **Per-Image Processing:** ~200-500ms per image (including technical analysis)
- **Cache Hits:** ~10-50ms per image (cached features)

### Caching Benefits
- **Label Embeddings:** Cached across service restarts
- **Feature Cache:** Per-photo caching avoids recomputation
- **Incremental Processing:** Only new photos processed in subsequent runs

## Monitoring and Debugging

### Log Output
```
INFO:features_service:Using MPS (Metal Performance Shaders) for acceleration
INFO:features_service:Loading OpenCLIP model: hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K
INFO:features_service:Loaded 50 labels and 6 templates
INFO:features_service:Built and cached embeddings for 50 labels
INFO:features_service:Extracting features for abed1315...
INFO:features_service:✓ Features extracted for abed1315: ['landscape', 'nature', 'outdoor']...
```

### Cache Management
```bash
# Clear feature cache
rm -rf data/cache/features/*.json

# Clear label embedding cache
rm -f data/cache/features/label_embeddings.pt
```

---

**The service is now fully integrated with the pipeline and optimized for MacBook Pro. Ready for testing with real photos!**

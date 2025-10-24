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
- **Data Preservation:** Maintains ALL metadata from ingest and preprocess services

### Data Flow
```
Ingest Service → Preprocess Service → Features Service
     ↓                    ↓                   ↓
  [exif, format]    [+ processing_metadata]  [+ features]
```

The Features Service receives artifacts from the Preprocess Service that contain:
- **From Ingest:** `exif`, `format`, `original_uri`, `ranking_uri`, `photo_id`
- **From Preprocess:** `std_uri`, `processing_metadata`
- **Adds:** `features` (CLIP labels + technical quality metrics)

### Input Format

The service accepts two input formats:

**1. From Preprocess Service (Preferred):**
```json
{
  "batch_id": "batch_20251017_105804",
  "artifacts": [
    {
      "photo_id": "48f6cf1e6e10e5367e229aabc3bffcf1c3bb3adac74365665fa9d6d1d1ed0541",
      "original_uri": "./data/input/992DDF60-302C-4AB3-A794-EE9D0DDC56AA.jpg",
      "ranking_uri": "./data/rankingInput/48f6cf1e6e10e5367e229aabc3bffcf1c3bb3adac74365665fa9d6d1d1ed0541.jpg",
      "std_uri": "./data/rankingInput/48f6cf1e6e10e5367e229aabc3bffcf1c3bb3adac74365665fa9d6d1d1ed0541_1024.jpg",
      "exif": {
        "camera": "Apple iPhone 11 Pro Max",
        "lens": "Unknown",
        "iso": 64,
        "aperture": "f/1.8",
        "shutter_speed": "1/60",
        "focal_length": "4mm",
        "datetime": "2024:12:25 11:25:57",
        "gps": null
      },
      "format": ".jpg",
      "processing_metadata": {
        "original_size": [3840, 2160],
        "standardized_size": [2048, 1152],
        "processing_method": "quality_preserved"
      }
    }
  ]
}
```



### Output Format
```json
{
  "batch_id": "batch_20251017_105804",
  "artifacts": [
    {
      "photo_id": "48f6cf1e6e10e5367e229aabc3bffcf1c3bb3adac74365665fa9d6d1d1ed0541",
      "original_uri": "./data/input/992DDF60-302C-4AB3-A794-EE9D0DDC56AA.jpg",
      "ranking_uri": "./data/rankingInput/48f6cf1e6e10e5367e229aabc3bffcf1c3bb3adac74365665fa9d6d1d1ed0541.jpg",
      "std_uri": "./data/rankingInput/48f6cf1e6e10e5367e229aabc3bffcf1c3bb3adac74365665fa9d6d1d1ed0541_1024.jpg",
      "exif": {
        "camera": "Apple iPhone 11 Pro Max",
        "lens": "Unknown",
        "iso": 64,
        "aperture": "f/1.8",
        "shutter_speed": "1/60",
        "focal_length": "4mm",
        "datetime": "2024:12:25 11:25:57",
        "gps": null
      },
      "format": ".jpg",
      "processing_metadata": {
        "original_size": [3840, 2160],
        "standardized_size": [2048, 1152],
        "processing_method": "quality_preserved"
      },
      "features": {
        "tech": {
          "sharpness": 1.0,
          "exposure": 0.7250478958473853,
          "noise": 0.09551166822581503,
          "clip_iqa": 0.8234567890123456,
          "brisque": 0.6789012345678901
        },
        "clip_labels": [
          {
            "label": "wide shot",
            "confidence": 0.956,
            "cosine_score": 0.845
          },
          {
            "label": "colorful",
            "confidence": 0.823,
            "cosine_score": 0.781
          },
          {
            "label": "close-up",
            "confidence": 0.712,
            "cosine_score": 0.658
          },
          {
            "label": "panorama",
            "confidence": 0.634,
            "cosine_score": 0.542
          },
          {
            "label": "indoor",
            "confidence": 0.521,
            "cosine_score": 0.421
          }
        ]
      }
    }
  ]
}
```

## CLIP Labels with Confidence Scores

The `clip_labels` field contains the top-5 CLIP classification results, each with three metrics:

- **label** (string): The photography-focused classification label
- **confidence** (0-1 float): Temperature-scaled probability from softmax, representing the model's confidence in this label
- **cosine_score** (0-1 float): Raw cosine similarity score between the image embedding and label embedding (for reference/advanced use)

### Confidence Score Interpretation

- **confidence > 0.8**: Very confident match (e.g., clearly a landscape photo labeled "landscape")
- **confidence 0.6-0.8**: Good match (e.g., strong indication of the label being present)
- **confidence 0.4-0.6**: Moderate match (e.g., the label applies but with other strong contenders)
- **confidence < 0.4**: Weak match (fallback/exploratory label)

The labels are ordered by confidence (descending), so the first label is the model's top prediction.

## Implementation Details

* **Framework:** Python 3.13+, integrated with existing pipeline
* **Model lib:** **open_clip_torch**
* **Primary checkpoint:** `hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K` (good accuracy, efficient)
* **Device:** MPS (Metal Performance Shaders) on MacBook Pro, CUDA fallback, then CPU
* **Precision:** FP16 on GPU/MPS for memory efficiency
* **Image processing:** Uses standardized images from preprocess service
* **Labels:** read from `config/labels.json` (50 photography-focused labels)
* **Prompt templates:** 6 templates per label; averaged text embeddings
* **Return:** top 5 CLIP labels with confidence scores (softmax probability) and cosine similarity scores
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

### TechnicalQualityAnalyzer (v2 - Optimized + IQA + Performance Tuned)
- **Sharpness:** Tenengrad method using Sobel gradient energy with tanh squashing (calibrated with K=2000.0)
- **Exposure:** Percentile-based analysis combining midtone proximity, dynamic range, and clipping penalty
- **Noise:** Wavelet-based sigma estimation using scikit-image (calibrated with σ_max=0.08)
- **CLIP-IQA:** Deep learning-based image quality assessment using PIQ library (336px bicubic resize, [0,1] input)
- **BRISQUE:** Blind/Referenceless Image Spatial Quality Evaluator normalized to [0,1] (PIQ library)
- **Device Auto-Detection:** Automatically selects MPS → CUDA → CPU for optimal performance
- **Optimized I/O:** Single grayscale image load shared across traditional metrics (sharpness, exposure, noise)


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
# Image Quality Assessment
piq = ">=0.8.0,<1"
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
- **Scoring:** Temperature-scaled softmax for probability distribution (confidence)
- **Output:** Top-5 labels with confidence (softmax probability) and raw cosine scores

### 4. Technical Quality Analysis (v2 + IQA + Performance Optimized)
- **Sharpness:** Tenengrad method (Sobel gradients) with tanh(K=2000.0) squashing for [0,1] mapping
- **Exposure:** Percentile analysis (p1,p50,p99) combining midtone proximity, dynamic range, and clipping penalty
- **Noise:** Wavelet sigma estimation via scikit-image.restoration.estimate_sigma with σ_max=0.08 calibration
- **CLIP-IQA:** Deep learning quality assessment using PIQ library (336px bicubic resize, [0,1] input range)
- **BRISQUE:** No-reference quality assessment using PIQ library, normalized to [0,1] where 1=better
- **Parallel Execution:** 3 workers (traditional metrics bundle + 2 IQA models) using ThreadPoolExecutor
- **Optimized I/O:** Single grayscale preview (max_side=512, INTER_AREA) shared across traditional metrics
- **Device Auto-Detection:** Automatic MPS → CUDA → CPU selection for optimal performance
- **Static Method Optimization:** No unnecessary instance creation in metric calculations
- **Preprocessing Fixes:** CLIPIQA uses proper [0,1] input format (not CLIP-normalized)

### 5. Caching Strategy (Enhanced)
- **Feature Cache:** Per-photo hash-based caching in JSON format with version metadata
- **Cache Key:** MD5 hash of `{photo_id}_features`
- **Version Tracking:** Cache includes model versions, PIQ version, and configuration hashes
- **Auto-Invalidation:** Cache automatically invalidated when models or configurations change
- **Cache Format:** `{"version": {...}, "features": {...}}`

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

### Expected Performance (Optimized)
- **Model Loading:** ~10-15 seconds on first run (CLIP + IQA models)
- **Label Embedding Cache:** ~5-10 seconds for 50 labels
- **Per-Image Processing (after warm-up):** 
  - **GPU (MPS):** ~130ms per image (parallel execution, optimized I/O)
  - **CPU Only:** ~200-300ms per image (parallel execution)
  - **Traditional metrics only:** ~43ms per image (shared grayscale load)
- **Cache Hits:** ~10-50ms per image (cached features with version validation)
- **IQA Model Warm-up:** First inference ~2s, subsequent ~50-100ms
- **Device Selection:** Automatic MPS detection on MacBook Pro for optimal performance

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

## Recent Optimizations (v2.1)

### Performance Improvements
- ✅ **MPS Auto-Detection:** Automatically uses Metal Performance Shaders on MacBook Pro
- ✅ **Optimized I/O:** Single grayscale image load shared across traditional metrics (3x reduction in file reads)
- ✅ **Static Method Fixes:** Eliminated unnecessary instance creation in metric calculations
- ✅ **Parallel Execution:** Reduced from 5 to 3 workers for better resource utilization

### CLIPIQA Preprocessing Fixes
- ✅ **Input Format:** Fixed to use [0,1] range instead of CLIP-normalized values
- ✅ **Image Sizing:** Proper 336px bicubic resize for optimal CLIPIQA performance
- ✅ **Error Handling:** Resolved "Expected values >= 0.0" errors

### Advanced IQA Integration
- ✅ **PIQ Library:** Integrated CLIP-IQA and BRISQUE via PyTorch Image Quality library
- ✅ **Quality Metrics:** Added 2 additional no-reference quality assessment scores
- ✅ **Caching Enhancement:** Version-aware cache with automatic model/config invalidation

### Technical Specifications
- **Traditional Metrics:** Shared grayscale preview (512px max, INTER_AREA)
- **IQA Models:** Device-specific optimization (MPS/CUDA/CPU)
- **Performance:** ~130ms per image on MPS, ~200-300ms on CPU
- **Cache Format:** `{"version": {...}, "features": {...}}` with model versioning

---

**The service is now fully optimized with advanced IQA capabilities and ready for production use on MacBook Pro with MPS acceleration!**

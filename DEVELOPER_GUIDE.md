# Framed 3B - Developer's Guide

## 🎯 Quick Start

This repository implements an intelligent photo curation system that automatically selects the best photos from a collection using AI-powered quality assessment, clustering, and optimization algorithms.

**Current Status:** ✅ Built through Features Service (MVP Stage 1 Complete)

### Prerequisites

- **Python 3.13+** (managed via Pixi)
- **macOS** (optimized for Apple Silicon with MPS acceleration)
- **Pixi** package manager

### Setup (30 seconds)

```bash
# Clone and enter the repo
cd framed_3b

# Install dependencies with Pixi
pixi install


# Add some photos to process
cp /path/to/your/photos/* data/input/

# Run the pipeline
pixi run python orchestrator.py
```

## 🏗️ Architecture Overview

### Core Pipeline Flow

```
Input Photos → Ingest → Preprocess → Features → Scoring → Clustering → Ranking → Optimizer → Curated List
     ↓           ↓         ↓           ✅         📋         📋         📋         📋           ↓
   ./data/input/ → photo_index → standardized → extracted → quality → moment → pairwise → optimal → ./data/output/
```

**Legend:**
- ✅ = Fully implemented and working
- 📋 = Planned/partially implemented
- ↓ = Data flow

### Microservices Architecture

The system uses an event-driven microservices architecture where each service:
- Operates independently with clear inputs/outputs
- Publishes events to trigger downstream services
- Caches results to avoid recomputation
- Maintains comprehensive audit logs

## 📁 Repository Structure

```
framed_3b/
├── 🎬 orchestrator.py              # Main pipeline coordinator
├── 📋 pixi.toml                    # Dependencies and environment
├── 📊 overall_plan.md              # Complete technical specification
├── 
├── 🔧 services/                    # Core microservices
│   ├── ✅ ingest_service.py        # Photo registration & EXIF extraction
│   ├── ✅ preprocess_service.py    # Image standardization
│   ├── ✅ features_service.py      # AI feature extraction (OpenCLIP + IQA)
│   ├── 📋 scoring_service.py       # Quality assessment
│   ├── 📋 clustering_service.py    # Near-duplicate grouping
│   ├── 📋 ranking_service.py       # LLM-powered ranking
│   ├── 📋 optimizer_service.py     # Diversity optimization
│   └── 📋 exporter_service.py      # Final output generation
│
├── 📚 Docs/                        # Detailed service documentation
│   ├── features_service.md         # Feature extraction deep dive
│   ├── scoring_service.md          # Quality scoring algorithms
│   ├── clustering_service.md       # Moment clustering strategies
│   ├── optimizer_service.md        # Diversity optimization math
│   └── remaining_services.md       # Ranking & export services
│
├── 🗂️ data/                        # Processing directories
│   ├── input/                      # 📥 Original photos (your input)
│   ├── rankingInput/               # 🔄 Standardized photos (1024px)
│   ├── processed/                  # 🔄 Alternative processing location
│   ├── emb/                        # 🧠 CLIP embeddings (.npy files)
│   ├── sal/                        # 👁️ Saliency heatmaps (.png files)
│   ├── features/                   # 💾 Feature cache
│   ├── themes/                     # 🎨 Theme specifications (.yaml)
│   ├── cache/                      # ⚡ Service-specific caches
│   └── output/                     # 📤 Final curated lists
│
├── 🔍 intermediateJsons/           # Pipeline outputs for debugging
│   ├── ingest/                     # Photo registration results
│   ├── features/                   # Feature extraction results
│   ├── scoring/                    # Quality scores
│   └── ...                        # Other pipeline stages
│
├── 📋 schemas/                     # JSON schemas for data validation
├── 🧪 mock_data/                   # Sample data for testing
└── 🎭 mock_services.py             # Mock implementations for development
```

## 🚀 How to Run the System

### Method 1: Auto-Discovery (Recommended)

```bash
# Place photos in data/input/
cp /path/to/photos/* data/input/

# Run the orchestrator - it auto-discovers photos and generates batch IDs
pixi run python orchestrator.py
```

### Method 2: Manual Batch Creation

```bash
# Generate ingest input from directory
pixi run python services/generate_ingest_input_service.py

# Run specific batch
pixi run python orchestrator.py --batch-id batch_20241024_143000
```

### Method 3: Individual Service Testing

```bash
# Test features service directly
pixi run python -c "
from services.features_service import FeaturesService
service = FeaturesService()
result = service.process('test_batch', './data/input')
print(f'Processed {len(result[\"artifacts\"])} photos')
"
```

## 🔧 Current Implementation Status

### ✅ Completed (MVP Stage 1)

#### 1. **Ingest Service** - Photo Registration
- ✅ Multi-format support (JPEG, PNG, HEIC, RAW, TIFF, WebP, BMP, GIF)
- ✅ EXIF extraction (camera, lens, exposure, GPS, timestamps)
- ✅ Content-addressed photo IDs (SHA256 hashing)
- ✅ Format conversion and validation
- ✅ Comprehensive error handling

#### 2. **Preprocess Service** - Image Standardization  
- ✅ Quality-preserving resize (max 2048px, 95% JPEG quality)
- ✅ Aspect ratio preservation
- ✅ Color profile maintenance
- ✅ Metadata preservation
- ✅ Batch processing optimization

#### 3. **Features Service** - AI-Powered Analysis
- ✅ **OpenCLIP Integration**: ViT-L-14 model with 50 photography labels
- ✅ **Advanced IQA**: CLIP-IQA and BRISQUE quality assessment
- ✅ **Technical Metrics**: Tenengrad sharpness, percentile exposure, wavelet noise
- ✅ **Performance Optimized**: MPS auto-detection, parallel execution
- ✅ **Robust Caching**: Version-aware cache with automatic invalidation
- ✅ **Production Ready**: Comprehensive error handling and logging

**Key Features Extracted:**
- Semantic similarity vectors (768-dim CLIP embeddings)
- Technical quality metrics (sharpness, exposure, noise)
- Photography-specific labels with confidence scores
- Advanced image quality assessment scores

### 📋 Planned (MVP Stage 2)

#### 4. **Scoring Service** - Quality Assessment
- 📋 Multi-dimensional scoring (Technical, Aesthetic, Vibe, Typography, Composition)
- 📋 Hard quality gates (Q_tech > 0.3 threshold)
- 📋 Feature-based score computation
- 📋 Normalization and bounds checking

#### 5. **Clustering Service** - Moment Grouping
- 📋 Quality-based clustering (current implementation)
- 📋 Perceptual hash similarity (planned upgrade)
- 📋 DBSCAN/HDBSCAN algorithms
- 📋 Near-duplicate detection

#### 6. **Ranking Service** - LLM-Assisted Ranking
- 📋 Pairwise photo comparison using GPT models
- 📋 Theme-aware ranking criteria
- 📋 Elo rating system
- 📋 Cost tracking and optimization

#### 7. **Optimizer Service** - Diversity Selection
- 📋 Submodular optimization for diversity
- 📋 Role assignment (opener, anchor, hero, body)
- 📋 Coverage analysis across multiple dimensions
- 📋 Marginal gain calculations

#### 8. **Exporter Service** - Final Output
- 📋 Curated list generation with metadata
- 📋 Diversity tagging and role assignment
- 📋 Audit trail creation
- 📋 Export-ready artifact preparation

## 🧠 Key Concepts

### Content-Addressed Storage
Every photo gets a unique ID based on its content hash (SHA256):
```python
photo_id = "sha256:a6d54a6157e85c3fce236acca72d7f67..."
```
This ensures:
- Deduplication across batches
- Deterministic processing
- Cache efficiency
- Audit trail integrity

### Event-Driven Architecture
Services communicate via events published to a message bus:
```python
# Service A completes and publishes event
message_bus.publish("features.completed", features_output)

# Service B automatically triggered
def handle_features_completed(event):
    scoring_output = scoring_service.process(event["data"])
```

### Intelligent Caching
Expensive operations (like CLIP inference) are cached:
```python
cache_key = hashlib.md5(f"{photo_id}_features_v2.1").hexdigest()
# Only recompute if photo or algorithm changes
```

### Quality Gates
Photos must pass quality thresholds to advance:
```python
if technical_quality < 0.3:
    # Photo dropped from further processing
    dropped_photos.append(photo_id)
```

## 🔍 Understanding the Data Flow

### 1. Input → Ingest
```json
{
  "batch_id": "batch_20241024_143000",
  "photos": [{"uri": "./data/input/IMG_001.jpg"}]
}
```

### 2. Ingest → Preprocess
```json
{
  "batch_id": "batch_20241024_143000",
  "photo_index": [{
    "photo_id": "sha256:a6d54a61...",
    "uri": "./data/input/IMG_001.jpg",
    "exif": {"camera": "iPhone 15 Pro", "iso": 100}
  }]
}
```

### 3. Preprocess → Features
```json
{
  "batch_id": "batch_20241024_143000",
  "artifacts": [{
    "photo_id": "sha256:a6d54a61...",
    "std_uri": "./data/rankingInput/a6d54a61_1024.jpg",
    "processing_metadata": {"standardized_size": [1024, 768]}
  }]
}
```

### 4. Features → Scoring (Current End Point)
```json
{
  "batch_id": "batch_20241024_143000",
  "artifacts": [{
    "photo_id": "sha256:a6d54a61...",
    "features": {
      "tech": {"sharpness": 0.85, "exposure": 0.72, "noise": 0.15},
      "clip_labels": [
        {"label": "photography", "confidence": 0.89},
        {"label": "landscape", "confidence": 0.82}
      ]
    }
  }]
}
```

## 🛠️ Development Workflow

### Adding a New Service

1. **Create the service file:**
```python
# services/new_service.py
class NewService:
    def process(self, batch_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Your processing logic
        return {"batch_id": batch_id, "results": [...]}
```

2. **Add event handling to orchestrator:**
```python
# orchestrator.py
def handle_previous_completed(self, event):
    new_output = self.services.new_service(event["data"])
    self.message_bus.publish("new.completed", new_output)
```

3. **Create documentation:**
```markdown
# services/service_documentation/new_service.md
## Overview
Purpose and responsibilities...
```

4. **Add JSON schema:**
```json
// schemas/new_schemas.json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "New Service Schemas"
}
```

### Testing Your Changes

```bash
# Run with mock data
pixi run python orchestrator.py

# Check intermediate outputs
ls intermediateJsons/*/

# Analyze results
pixi run python intermediateJsons/analyze_pipeline.py

# Test individual service
pixi run python -c "
from services.your_service import YourService
service = YourService()
result = service.process('test', sample_data)
"
```

### Debugging Pipeline Issues

1. **Check intermediate outputs:**
```bash
# See what each stage produced
find intermediateJsons/ -name "*.json" | sort

# Inspect specific stage
cat intermediateJsons/features/batch_xxx_features_output.json | jq '.artifacts | length'
```

2. **Check logs:**
```bash
# Service-specific logs
tail -f intermediateJsons/features/features_service.log

# Orchestrator events
pixi run python -c "
from orchestrator import Orchestrator
orch = Orchestrator()
print(orch.message_bus.get_event_log())
"
```

3. **Validate data schemas:**
```bash
pixi run python -c "
import json, jsonschema
with open('schemas/features_schemas.json') as f:
    schema = json.load(f)
with open('intermediateJsons/features/batch_xxx_features_output.json') as f:
    data = json.load(f)
jsonschema.validate(data, schema)
print('✅ Schema validation passed')
"
```

## 📊 Performance & Optimization

### Current Performance Characteristics

- **Features Service**: ~2-5 seconds per photo (with CLIP inference)
- **Cache Hit Rate**: 90%+ for repeated processing
- **Memory Usage**: ~200MB per photo during processing
- **Batch Processing**: Linear scaling with photo count

### Optimization Tips

1. **Use MPS acceleration** (automatic on Apple Silicon)
2. **Batch photos** for efficient GPU utilization
3. **Enable caching** to avoid recomputation
4. **Monitor memory usage** for large batches

### Scaling Considerations

- **Current**: Handles 10-100 photos efficiently
- **Target**: 1000+ photos with batch optimization
- **Bottlenecks**: CLIP inference, file I/O
- **Solutions**: Parallel processing, streaming, cloud deployment

## 🎯 Next Steps for Development

### Immediate (Complete MVP Stage 2)

1. **Implement Scoring Service**
   - Use extracted features to compute quality scores
   - Apply technical quality gates
   - Generate multi-dimensional assessments

2. **Build Clustering Service**
   - Start with quality-based grouping
   - Upgrade to perceptual hash similarity
   - Implement moment detection

3. **Create Ranking Service**
   - Simple quality-based ranking first
   - Add LLM pairwise comparison later
   - Implement cost tracking

4. **Develop Optimizer Service**
   - Basic cluster winner selection
   - Add diversity optimization algorithms
   - Implement role assignment

5. **Complete Exporter Service**
   - Generate final curated lists
   - Add metadata and audit trails
   - Create export artifacts

### Medium Term (Production Ready)

1. **Add Human Review UI**
   - Visual interface for curation decisions
   - Lock/exclude photo controls
   - Real-time pipeline re-running

2. **Implement LLM Integration**
   - GPT-4V for pairwise photo ranking
   - Theme-aware comparison prompts
   - Cost optimization and caching

3. **Performance Optimization**
   - Parallel service execution
   - Streaming for large batches
   - Advanced caching strategies

### Long Term (Scale & Features)

1. **Cloud Deployment**
   - Kubernetes orchestration
   - S3-compatible storage
   - Redis caching layer

2. **Advanced AI Features**
   - Custom vision models
   - Personalized curation
   - Style transfer and enhancement

3. **Integration Ecosystem**
   - Layout service integration
   - CMS connectors
   - API for external systems

## 🤝 Contributing

### Code Style
- Follow PEP 8 for Python code
- Use type hints for all functions
- Add comprehensive docstrings
- Include error handling and logging

### Testing
- Test individual services in isolation
- Validate against JSON schemas
- Check intermediate outputs
- Verify end-to-end pipeline flow

### Documentation
- Update service documentation in `Docs/`
- Add examples and use cases
- Document configuration options
- Include performance characteristics

## 📚 Additional Resources

- **`overall_plan.md`**: Complete technical specification
- **`Docs/`**: Detailed service documentation
- **`schemas/`**: Data contracts and validation
- **`intermediateJsons/README.md`**: Pipeline analysis guide
- **`services/service_documentation/`**: Implementation guides

## 🆘 Getting Help

1. **Check the logs** in `intermediateJsons/*/`
2. **Validate your data** against schemas in `schemas/`
3. **Review the documentation** in `Docs/`
4. **Test with mock data** in `mock_data/`
5. **Run the analysis script** in `intermediateJsons/analyze_pipeline.py`

---

**Happy Curating! 📸✨**

This system represents a sophisticated approach to automated photo curation, combining computer vision, AI, and optimization algorithms to select the best photos from any collection. The modular architecture makes it easy to understand, extend, and deploy at scale.

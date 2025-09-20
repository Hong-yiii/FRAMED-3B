# Photo Curation System - Services Overview

## Architecture Overview

The Framed 3B photo curation system uses a modular microservices architecture to process, analyze, and curate photo collections. Each service handles a specific stage of the curation pipeline, allowing for independent development, testing, and scaling.

## Core Data Flow

```
Input Photos → Ingest → Preprocess → Features → Scoring → Clustering → Ranking → Optimizer → Curated List
     ↓           ↓         ↓           ↓         ↓         ↓         ↓         ↓           ↓
   ./data/input/ → photo_index → standardized → extracted → quality → moment → pairwise → optimal → ./data/output/
```

## Service Directory

### 1. **Ingest Service** (`ingest_service.py`)
**Purpose:** Register photos, verify media files, extract metadata, and prepare for processing

**Key Functions:**
- Validates photo files and URIs
- Converts various formats (RAW, HEIC, etc.) to JPEG
- Extracts comprehensive EXIF metadata (camera, lens, exposure, GPS, etc.)
- Generates content-addressed photo IDs using SHA256 hashes
- Creates standardized ranking-ready copies

**Input:** Batch configuration with photo URIs
**Output:** Photo index with metadata and processed URIs

### 2. **Preprocess Service** (`preprocess_service.py`)
**Purpose:** Create standardized versions of photos without quality loss

**Key Functions:**
- Resizes large images to reasonable dimensions (2048px max)
- Maintains aspect ratios and high JPEG quality (95%)
- Preserves color profiles and metadata
- Creates consistent format for downstream processing

**Input:** Photo index from ingest
**Output:** Standardized photo artifacts with processing metadata

### 3. **Features Service** (`features_service.py`)
**Purpose:** Extract visual and technical features from processed images

**Key Functions:**
- CLIP embeddings for semantic similarity
- Perceptual hash (pHash) for near-duplicate detection
- Technical quality metrics (sharpness, exposure, noise)
- Saliency analysis and negative space detection
- Face detection and landmark extraction
- Color palette analysis and clustering

**Input:** Standardized photo artifacts
**Output:** Rich feature vectors and analysis results

### 4. **Scoring Service** (`scoring_service.py`)
**Purpose:** Compute quality scores for photos using extracted features

**Key Functions:**
- Technical quality assessment (sharpness, exposure, noise)
- Aesthetic scoring based on composition and visual appeal
- Vibe analysis for emotional impact
- Typography assessment for text space analysis
- Composition evaluation using rule-of-thirds and balance
- Hard quality gates to filter unusable photos

**Input:** Extracted features
**Output:** Normalized quality scores (0-1 scale)

### 5. **Clustering Service** (`clustering_service.py`)
**Purpose:** Group near-duplicate photos into moment clusters

**Key Functions:**
- Uses perceptual hashing for similarity detection
- Groups photos taken at the same moment/event
- DBSCAN/HDBSCAN clustering algorithms
- Prevents duplicate selection while preserving variety

**Input:** Scored photos with quality metrics
**Output:** Moment clusters with member photo IDs

### 6. **Ranking Service** (`ranking_service.py`)
**Purpose:** Rank photos within clusters using LLM-assisted pairwise judgments

**Key Functions:**
- Pairwise comparison using GPT models
- Theme-aware ranking based on curation criteria
- Elo rating system for consistent rankings
- Rationale generation for human review
- Cost tracking and token optimization

**Input:** Photo clusters and theme specifications
**Output:** Ranked cluster winners with LLM rationales

### 7. **Optimizer Service** (`optimizer_service.py`)
**Purpose:** Select optimal photo subset with diversity optimization

**Key Functions:**
- Submodular optimization for diversity coverage
- Role assignment (opener, anchor, etc.)
- Theme constraint satisfaction
- Coverage analysis across multiple dimensions
- Marginal gain calculations for selection justification

**Input:** Ranked clusters and curation parameters
**Output:** Optimal photo selection with roles and coverage metrics

### 8. **Exporter Service** (`exporter_service.py`)
**Purpose:** Produce final curated list and export artifacts

**Key Functions:**
- Generate human-readable curated lists
- Export processed images to final locations
- Create audit trails with decision rationales
- Produce layout-ready metadata

**Input:** Optimization results
**Output:** Curated photo collection and artifacts

## Supporting Components

### Theme Manager
- Loads and validates theme specifications
- Provides rubric weights and constraints
- Manages curation criteria and preferences

### Local Cache
- File-based caching for features and embeddings
- Avoids recomputation of expensive operations
- Content-addressed storage using photo IDs

### Orchestrator
- Coordinates service execution via async events
- Manages batch state and error handling
- Provides retry logic and failure recovery

## Key Design Principles

1. **Content-Addressed:** All photos identified by SHA256 hash of processed content
2. **Idempotent:** Services can be safely re-run with same inputs
3. **Stateless:** Each service operates independently with clear inputs/outputs
4. **Observable:** Comprehensive logging and audit trails
5. **Efficient:** Caching, batching, and incremental processing
6. **Extensible:** Easy to add new features or replace components

## Data Contracts

- **Photo IDs:** SHA256 hashes for content addressing
- **Scores:** All normalized to 0-1 scale with version tracking
- **Features:** Standardized format with metadata
- **Events:** Async communication between services
- **Cache:** File-based with hash-based keys

## Quality Gates

- Technical quality threshold (Q_tech > 0.3)
- Duplicate filtering via clustering
- Diversity optimization across multiple axes
- Theme constraint validation

## Performance Characteristics

- **Local Processing:** All computation done offline
- **Batch Optimization:** Process photos in batches
- **Incremental Updates:** Cache expensive operations
- **Cost Control:** LLM usage limited to critical ranking decisions

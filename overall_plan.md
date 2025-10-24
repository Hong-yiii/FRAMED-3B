awesome — let's lock (B) Curation into a clean, composable **micro-services** architecture. This is the implementation plan for the curation algorithm detailed in `in_depth_breakdown.md`. Code comes later; for now we'll define **stages, contracts, and responsibilities** so you can wire it up with Cursor (or swap parts) without surprises.

---

# MVP Architecture (Simplified Microservices)

## Core Flow (Local MVP)

```
ingest → preprocess → features → scoring → clustering
      → ranking (LLM) → optimizer → exporter → curated_list.json
```

**MVP Simplifications:**
- ✅ **Local file storage** (no cloud dependencies except LLM)
- ✅ **In-memory message bus** (RabbitMQ optional for MVP)
- ✅ **Separate services** (preprocess and features are distinct for flexibility)
- ✅ **File-based caching** (instead of SQLite for simplicity)
- ✅ **Minimal external dependencies** (only LLM service)

Each service is a stateless Python module with async communication. Keep everything **idempotent** and **content-addressed** (photo\_id = hash).

---

# Services, inputs & outputs

## 1) Ingest Service

**Purpose:** register a batch, verify media, extract EXIF.

* **Input (REST/POST `/batches`):**

  ```json
  {
    "batch_id": "batch_2025-09-11_a",
    "photos": [{"uri":"./data/input/P001.jpg"}, {"uri":"./data/input/P002.jpg"}],
    "theme_spec_ref": "./data/themes/theme.yaml",
    "user_overrides": {"lock_in":["P017.jpg"], "exclude":["P099.jpg"]}
  }
  ```
* **Output (event `ingest.completed`):**

  ```json
  {
    "batch_id": "batch_2025-09-11_a",
    "photo_index": [
      {"photo_id":"sha256:...P001", "uri":"./data/input/P001.jpg", "exif":{...}},
      ...
    ]
  }
  ```

## 2) Preprocess Service

**Purpose:** standardize images without quality loss for consistent processing.

* **Input (event `ingest.completed`)**
* **Output (event `preprocess.completed`):**

  ```json
  {
    "batch_id":"...",
    "artifacts":[
      {
        "photo_id":"sha256:...P001",
        "original_uri":"./data/input/P001.jpg",
        "ranking_uri":"./data/rankingInput/P001.jpg",
        "std_uri":"./data/rankingInput/P001_1024.jpg",
        "processing_metadata": {
          "original_size": [3264, 2448],
          "standardized_size": [2048, 1536],
          "processing_method": "quality_preserved"
        }
      }
    ]
  }
  ```

## 3) Features Service ✅ **COMPLETED & OPTIMIZED**

**Purpose:** extract rich visual and technical features from standardized images using OpenCLIP and advanced IQA models.

* **Input (event `preprocess.completed`)**
* **Output (event `features.completed`):**

  ```json
  {
    "batch_id":"...",
    "artifacts":[
      {
        "photo_id":"sha256:...P001",
        "std_uri":"./data/rankingInput/P001_1024.jpg",
        "features": {
          "tech": {
            "sharpness":0.85,
            "exposure":0.72,
            "noise":0.15,
            "clip_iqa":0.78,
            "brisque":0.82
          },
          "clip_labels":[
            {"label":"photography","confidence":0.89,"cosine_score":0.76},
            {"label":"landscape","confidence":0.82,"cosine_score":0.71},
            {"label":"nature","confidence":0.75,"cosine_score":0.68}
          ]
        }
      }
    ]
  }
  ```

**Key Features:**
- **OpenCLIP Integration:** ViT-L-14 model with 50 photography-focused labels
- **Advanced IQA:** CLIP-IQA and BRISQUE quality assessment via PIQ library
- **Performance Optimized:** MPS auto-detection, parallel execution, shared I/O
- **Robust Caching:** Version-aware cache with automatic invalidation

## 4) Scoring Service

**Purpose:** compute designer-aligned quality scores from extracted features.

* **Input:** `features.completed`
* **Output (`score.completed`):**

  ```json
  {
    "batch_id":"...",
    "scores":[
      {
        "photo_id":"sha256:...P001",
        "Q_tech":0.82,
        "Aesthetic":0.73,
        "Vibe":0.64,
        "Typography":0.47,
        "Composition":0.75,
        "Total_prelim":0.76
      }
    ],
    "dropped_for_tech":["sha256:...P233","sha256:...P241"]
  }
  ```

## 5) Clustering Service

**Purpose:** group near-duplicates ("same moment").

* **Input:** `score.completed`
* **Output (`cluster.completed`):**

  ```json
  {
    "batch_id":"...",
    "clusters":[
      {"cluster_id":"m_0001","members":["sha256:...P001","sha256:...P004","sha256:...P005"]},
      {"cluster_id":"m_0002","members":["sha256:...P017"]}
    ]
  }
  ```

## 6) Ranking Service (LLM-assisted)

**Purpose:** rank photos within clusters using pairwise LLM judgments.

* **Input:** `cluster.completed`, `score.completed`, `theme_spec_ref`
* **Output (`cluster.rank.completed`):**

  ```json
  {
    "batch_id":"...",
    "cluster_winners":[
      {"cluster_id":"m_0001","hero":"sha256:...P004","alternates":["sha256:...P001"]},
      {"cluster_id":"m_0002","hero":"sha256:...P017","alternates":[]}
    ],
    "llm_rationales":{
      "sha256:...P004":[ "Balanced subject, dusk tones match palette, space for headline." ]
    },
    "judge_costs":{"pairs_scored":0,"tokens_est":0}
  }
  ```

## 7) Optimizer Service

**Purpose:** diversity optimization and final photo selection.

* **Input:** `cluster.rank.completed`, `score.completed`, `theme_spec_ref`
* **Output (`selection.completed`):**

  ```json
  {
    "batch_id":"...",
    "selected_ids":["sha256:...P004","sha256:...P017","..."],
    "roles":[
      {"photo_id":"sha256:...P004","role":"opener","prob":0.78},
      {"photo_id":"sha256:...P017","role":"body","prob":0.65}
    ],
    "coverage":{
      "scene_type":0.85,"palette_cluster":0.78,"time_of_day":0.82,
      "people_count":0.80,"orientation":0.95
    },
    "marginal_gains":{"sha256:...P004":0.043,"sha256:...P017":0.038}
  }
  ```

## 8) Exporter Service

**Purpose:** produce final curated list and export artifacts.

* **Input:** `selection.completed`
* **Output:** `curated_list.json` + exported images

  ```json
  {
    "batch_id":"...",
    "version":"1.0.0",
    "theme_spec_ref":"./data/themes/theme.yaml",
    "items":[
      {
        "photo_id":"sha256:...P004",
        "rank":1,
        "cluster_id":"m_0001",
        "role":"opener",
        "scores":{"Q_tech":0.82,"Aesthetic":0.73,"Vibe":0.64,"Typography":0.47,"Composition":0.75,"Total":0.76},
        "diversity_tags":["scene:street","palette:warm","time:dusk","people:2","orient:landscape"],
        "reasons":[
          "Cluster hero (highest quality in group)",
          "Excellent technical scores across dimensions"
        ],
        "artifacts":{"std":"./data/rankingInput/P004_1024.jpg"}
      }
    ],
    "audit":{"created_at":"2025-09-20T14:37:03Z","optimizer_params":{"diversity_weight":0.3,"quality_weight":0.7}}
  }
  ```

## 9) Generate Ingest Input Service (Utility)

**Purpose:** scan directories and create batch input for the curation pipeline.

* **Input:** directory path and batch parameters
* **Output:** `ingest_input.json` with photo batch configuration

---

# MVP Supporting Components

## Core Services

* **Theme Manager**
  * Loads and validates `theme.yaml` files from `./data/themes/`
  * Provides rubric weights and constraints to scoring/optimization
  * **MVP:** Simple YAML loader (no service layer needed)

* **Local Cache**
  * File-based cache for features/embeddings (avoid recompute)
  * Simple key-value store using SQLite
  * **MVP:** Local filesystem cache with hash-based keys

* **Orchestrator**
  * Coordinates service execution via async events
  * Manages batch state and error handling
  * **MVP:** In-memory coordinator with retry logic

## External Dependencies (MVP)

* **LLM Service** (External API)
  * Pairwise photo ranking and rationale generation
  * **MVP:** OpenAI API or local LLM server
  * Cost tracking and token limits

## Storage Strategy

* **Local Filesystem Structure:**
  ```
  ./data/
  ├── input/          # Original photos (read-only)
  ├── rankingInput/   # Standardized images for curation
  ├── processed/      # Alternative processed images location
  ├── emb/            # CLIP embedding NumPy arrays
  ├── sal/            # Saliency heatmap PNG files
  ├── features/       # Feature extraction cache
  ├── themes/         # Theme YAML files
  ├── cache/          # Service-specific caches
  └── output/         # Final curated sets and exports
  ```

* **intermediateJsons/**     # Service outputs and logs
  ```
  ├── ingest/         # Ingest service outputs
  ├── preprocess/     # Preprocess service outputs
  ├── features/       # Features service outputs
  ├── scoring/        # Scoring service outputs
  ├── clustering/     # Clustering service outputs
  ├── ranking/        # Ranking service outputs
  ├── optimizer/      # Optimizer service outputs
  └── exporter/       # Final curated lists
  ```

* **Database:** File-based caching (upgrade to SQLite/PostgreSQL later)
* **Message Bus:** In-memory async queues (upgrade to RabbitMQ later)

---

# Contracts that keep things swappable

* **Embeddings:** always `.npy` float32 + metadata `{model_name, dim, pooling}`.
* **Scores:** all normalized to `[0,1]` with a `scorecard_version`.
* **Coverage tags:** `diversity_tags` are `axis:value` strings + a machine map `{axis: one-hot or numeric}`.
* **Constraints:** human inputs are **declarative** (`lock_in`, `exclude`, `role_quotas`, `axis_emphasis`) so the optimizer can re-run without code changes.

---

# Cost & latency strategy (important for you)

* Do **everything** local/offline first: tech quality, aesthetics, vibe, typography, composition.
* Call LLM **only**:

  1. within **clusters** (≤4 contenders)
  2. among **opener/anchor** finalists (e.g., top 20)
* Cap pairwise comparisons (e.g., **6** per cluster via round-robin + Elo).
* **Cache** judgments; never re-ask for the same pair & theme.
* Allow **LLM-free mode** (use LAION aesthetics + heuristics) for ultra-cheap MVPs.

---

# MVP milestones (execution order)



## ✅ **M1 – Local Signals (COMPLETED)**

   * ✅ **Ingest Service:** Photo registration, format conversion, EXIF extraction
   * ✅ **Preprocess Service:** Image standardization without quality loss
   * ✅ **Features Service:** Rich feature extraction (OpenCLIP + advanced IQA metrics)
     - ✅ OpenCLIP ViT-L-14 with 50 photography labels
     - ✅ Technical metrics: Tenengrad sharpness, percentile exposure, wavelet noise
     - ✅ Advanced IQA: CLIP-IQA and BRISQUE via PIQ library
     - ✅ Performance optimizations: MPS auto-detection, parallel execution, shared I/O
     - ✅ Enhanced caching with version tracking and auto-invalidation
   * 📋 **Scoring Service:** Quality assessment with technical gate (Q_tech > 0.3)
   * ✅ **Advanced Caching:** Version-aware file-based caching with automatic invalidation

## 🚧 **M2 – Moment Clustering & Selection (IN PROGRESS)**

   * 📋 **Clustering Service:** Quality-based photo grouping
   * 📋 **Ranking Service:** Intra-cluster ranking (quality-based, LLM-ready)
   * 📋 **Optimizer Service:** Diversity optimization and selection
   * 📋 **Exporter Service:** Final curated list generation
   * 📋 **pHash Clustering:** Upgrade from quality-based to perceptual similarity
   * 📋 **Submodular Optimization:** Replace simple selection with mathematical optimization

## 📋 **M3 – Human Review UI (PLANNED)**

   * Why-cards (bars for each score, coverage heat), lock/exclude, nudges → re-run.

## 📋 **M4 – LLM Pairwise (Scoped) (PLANNED)**

   * Add pairwise judge inside clusters + for opener finalists; audit + caching.


---

# Operational notes

* **Idempotency keys:** every POST includes `{batch_id, stage, hash(inputs)}`.
* **Determinism:** fix random seeds for clustering/selection; log them.
* **Observability:** per-stage timings, LLM token spend, drop reasons, coverage deltas.
* **Fallbacks:** if coverage low on an axis, surface a **prompt** (“need more night shots?”).
* **Failure isolation:** if LLM judge fails, default to `Total_prelim` ordering.

---

# MVP Integration & Next Steps

## Local Development Setup

1. **Directory Structure:**
   ```bash
   mkdir -p data/{input,rankingInput,emb,sal,features,themes,cache,output}
   mkdir -p intermediateJsons/{ingest,preprocess,features,scoring,clustering,ranking,optimizer,exporter}
   ```

2. **Theme Files:** Place theme YAML files in `./data/themes/`

3. **Input Photos:** Place photos in `./data/input/`

4. **Run MVP Pipeline:**
   ```bash
   # Generate ingest input from photos
   python services/generate_ingest_input_service.py

   # Run full curation pipeline
   python orchestrator.py
   ```

## Integration Points

* **`curated_list.json`** → **Layout Service** (Stage C)
  * Provides roles, diversity tags, scores, and rationales
  * Enables intelligent page layout and sequencing

* **Local Cache** → **Performance Optimization**
  * Avoids recomputing features for repeated photos
  * Enables incremental batch processing

* **Audit Logs** → **Quality Tracking**
  * Track curation decisions and performance metrics
  * Enable A/B testing of different curation strategies

## Scaling Path (Post-MVP)

1. **Database:** SQLite → PostgreSQL
2. **Message Bus:** In-memory → RabbitMQ/NATS
3. **Storage:** Local → S3-compatible (MinIO)
4. **Cache:** File-based → Redis
5. **Compute:** Local → Kubernetes deployment

---

awesome — let's lock (B) Curation into a clean, composable **micro-services** architecture. This is the implementation plan for the curation algorithm detailed in `in_depth_breakdown.md`. Code comes later; for now we'll define **stages, contracts, and responsibilities** so you can wire it up with Cursor (or swap parts) without surprises.

---

# MVP Architecture (Simplified Microservices)

## Core Flow (Local MVP)

```
ingest → preprocess → features → scoring → clustering
      → ranking (LLM) → optimizer → curated_list.json
```

**MVP Simplifications:**
- ✅ **Local file storage** (no cloud dependencies except LLM)
- ✅ **In-memory message bus** (RabbitMQ optional for MVP)
- ✅ **Combined services** (preprocess + features can be one service)
- ✅ **SQLite database** (instead of Postgres)
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

## 2) Process & Feature Service (Combined for MVP)

**Purpose:** standardize images + extract features in one step (MVP optimization).

* **Input (event `ingest.completed`)**
* **Output (event `features.completed`):**

  ```json
  {
    "batch_id":"...",
    "artifacts":[
      {
        "photo_id":"sha256:...P001",
        "thumb_uri":"./data/thumbs/P001_512.jpg",
        "std_uri":"./data/processed/P001_1024.jpg",
        "features": {
          "embeddings":{"clip_L14":"./data/features/P001_clip.npy"},
          "hashes":{"phash":"22f9a3..."},
          "tech": {"sharpness":0.61,"exposure":0.74,"noise":0.18,"horizon_deg":-1.7},
          "saliency":{"heatmap_uri":"./data/features/P001_saliency.png","neg_space_ratio":0.36},
          "faces":{"count":2,"landmarks_ok":true},
          "palette":{"lab_centroids":[[62,10,-5],[48,-3,22]],"cluster_id":"pal_07"}
        }
      }
    ]
  }
  ```

## 3) Scoring Service

**Purpose:** compute designer-aligned scores without LLM (fast/cheap).

* **Input:** `features.completed`
* **Output (`score.completed`):**

  ```json
  {
    "batch_id":"...",
    "scores":[
      {
        "photo_id":"sha256:...P001",
        "Q_tech":0.72,
        "Aesthetic":0.68,
        "Vibe":0.81,
        "Typography":0.62,
        "Composition":0.77,
        "Total_prelim":0.74
      }
    ],
    "dropped_for_tech":["sha256:...P233","sha256:...P241"]
  }
  ```

## 4) Clustering Service

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

## 5) Ranking Service (LLM-assisted)

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
    "judge_costs":{"pairs_scored":26,"tokens_est":14500}
  }
  ```

## 6) Optimizer Service (Combined)

**Purpose:** role classification + diversity optimization + final selection.

* **Input:** `cluster.rank.completed`, `score.completed`, `theme_spec_ref`
* **Output (`selection.completed`):**

  ```json
  {
    "batch_id":"...",
    "selected_ids":["sha256:...P004","sha256:...P017","..."],
    "roles":[
      {"photo_id":"sha256:...P004","role":"opener","prob":0.61},
      {"photo_id":"sha256:...P017","role":"anchor","prob":0.82}
    ],
    "coverage":{
      "scene_type":0.88,"palette_cluster":0.81,"time_of_day":0.73,
      "people_count":0.77,"orientation":0.95
    },
    "marginal_gains":{"sha256:...P004":0.043,"sha256:...P017":0.039}
  }
  ```

## 7) Human Review Service (Optional MVP)

**Purpose:** present explainable cards, accept lock-in/exclude adjustments.

* **Input:** `selection.completed`
* **Output (`review.update`):**

  ```json
  {
    "batch_id":"...",
    "lock_in":["sha256:...P017"],
    "exclude":["sha256:...P044"],
    "nudges":{"emphasize_axes":["night","people>=2"]}
  }
  ```

## 8) Export Service

**Purpose:** produce final curated list and export artifacts.

* **Input:** `selection.completed` (+ optional `review.update`)
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
        "scores":{"Q_tech":0.72,"Aesthetic":0.68,"Vibe":0.81,"Typography":0.62,"Composition":0.77,"LLM":0.84,"Total":0.79},
        "diversity_tags":["scene:street","palette:teal-orange","time:dusk","people:2","orient:landscape"],
        "reasons":[
          "Cluster hero (won 5/6 pairwise).",
          "Lower-right third subject; sky space for headline.",
          "Palette matches theme accent."
        ],
        "artifacts":{"thumb":"./data/thumbs/P004_512.jpg","std":"./data/processed/P004_1024.jpg"}
      }
    ],
    "audit":{"created_at":"2025-09-11T11:05:00Z","optimizer_params":{"alpha":1.0,"beta":1.0,"gamma":1.0}}
  }
  ```

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
  ├── input/          # Original photos
  ├── processed/      # Standardized images
  ├── thumbs/         # Thumbnails (512px)
  ├── features/       # Embeddings, saliency maps
  ├── themes/         # Theme YAML files
  ├── cache/          # Computed features cache
  └── output/         # Final curated sets
  ```

* **Database:** SQLite for metadata, audit logs, and batch state
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



2. **M1 – Local Signals**

   * Implement preprocess, features, scoring (no LLM), hard tech gate.

3. **M2 – Moment Clustering & Selection**

   * pHash+embedding DBSCAN/HDBSCAN → cluster heroes using **Total\_prelim** only.
   * Submodular diversity optimizer with role quotas; produce `curated_list.json`.

4. **M3 – Human Review UI**

   * Why-cards (bars for each score, coverage heat), lock/exclude, nudges → re-run.

5. **M4 – LLM Pairwise (Scoped)**

   * Add pairwise judge inside clusters + for opener finalists; audit + caching.

6. **M5 – Personalization**

   * Collect pairwise picks → fit **Bradley–Terry** or a small linear head; add as `BT_personal`.

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
   mkdir -p data/{input,processed,thumbs,features,themes,cache,output}
   ```

2. **Theme Files:** Place theme YAML files in `./data/themes/`

3. **Input Photos:** Place photos in `./data/input/`

4. **Run MVP Pipeline:**
   ```bash
   python -m m0.test_m0  # Test the skeleton
   python orchestrator.py  # Run full pipeline
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

# Remaining Services Documentation

This document covers the remaining services in the photo curation system: Ranking Service, Exporter Service, and Generate Ingest Input Service.

---

# Ranking Service

## Overview

The Ranking Service ranks photos within clusters to identify the best representatives. It serves as the quality assessment layer that determines cluster winners and alternates, preparing the optimal candidates for final selection. Currently implemented with simple quality-based ranking that can be enhanced with LLM-powered pairwise comparisons.

## Key Responsibilities

1. **Intra-Cluster Ranking:** Order photos within each cluster by quality
2. **Winner Selection:** Identify cluster heroes and alternates
3. **Rationale Generation:** Provide human-readable explanations
4. **Cost Tracking:** Monitor LLM usage and token consumption
5. **Fallback Handling:** Graceful degradation when advanced ranking unavailable

## Processing Workflow

### 1. Input Processing
- Receives clusters from clustering service
- Validates cluster structure and member photos
- Handles both clustered and singleton photo sets

### 2. Ranking Algorithm
- Apply ranking strategy to each cluster
- Identify highest quality photos as winners
- Select alternates for diversity and backup

### 3. Rationale Generation
- Generate human-readable explanations for rankings
- Track decision-making process
- Prepare audit trail for transparency

### 4. Output Generation
- Structure cluster winners with IDs and metadata
- Include ranking rationales and cost information
- Generate summary statistics

## Input Format

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
    }
  ]
}
```

## Output Format

```json
{
  "batch_id": "batch_20250920_143703",
  "cluster_winners": [
    {
      "cluster_id": "m_0001",
      "hero": "a6d54a6157e85c3fce236acca72d7f67...",
      "alternates": ["b8c92d4e6f1a3h5j...", "c9d38e5f7g2b4i6k..."]
    }
  ],
  "llm_rationales": {
    "a6d54a6157e85c3fce236acca72d7f67...": [
      "Highest technical quality with excellent sharpness",
      "Balanced composition with good negative space"
    ]
  },
  "judge_costs": {
    "pairs_scored": 0,
    "tokens_est": 0
  }
}
```

## Ranking Algorithm Details

### Current Quality-Based Ranking
```python
def rank_cluster(members):
    if len(members) == 1:
        return members[0], []

    # Sort by quality score (placeholder)
    hero = members[0]  # Assume sorted by quality
    alternates = members[1:min(3, len(members))]
    return hero, alternates
```

### Future LLM Pairwise Ranking
**Elo Rating System:**
1. Initialize ratings for all photos
2. Compare photo pairs using LLM judge
3. Update ratings based on comparison outcomes
4. Select top-rated photos as winners

**Pairwise Comparison:**
- Prompt: "Which photo better fits the theme?"
- Criteria: Technical quality, composition, relevance
- Output: Winner selection with rationale

## Performance Characteristics

- **Processing Time:** O(n) linear in number of clusters
- **Memory Usage:** Minimal, operates on photo IDs only
- **Scalability:** Handles hundreds of clusters efficiently

---

# Exporter Service

## Overview

The Exporter Service creates the final curated list output for the photo curation system. It transforms the optimized photo selection into a comprehensive, human-readable format with rich metadata, diversity tags, and audit information. This service serves as the final output stage that prepares the curation results for layout services or human consumption.

## Key Responsibilities

1. **Curated List Generation:** Create structured photo collections
2. **Diversity Tagging:** Add categorization metadata
3. **Role Assignment:** Classify photos by layout function
4. **Audit Trail Creation:** Include decision rationale and metadata
5. **Export Preparation:** Format for downstream consumption

## Processing Workflow

### 1. Input Processing
- Receives selected photo IDs from optimizer service
- Validates selection completeness and structure
- Prepares for metadata enrichment

### 2. Metadata Enrichment
- Add diversity tags for each photo
- Assign layout roles and probabilities
- Include quality scores and rankings

### 3. Rationale Integration
- Incorporate ranking rationales from previous stages
- Add selection reasoning and marginal gains
- Create comprehensive audit trail

### 4. Output Formatting
- Structure final curated list
- Generate export-ready JSON format
- Include version and timestamp information

## Input Format

```json
{
  "batch_id": "batch_20250920_143703",
  "selected_ids": [
    "a6d54a6157e85c3fce236acca72d7f67...",
    "b8c92d4e6f1a3h5j..."
  ]
}
```

## Output Format

```json
{
  "batch_id": "batch_20250920_143703",
  "version": "1.0.0",
  "theme_spec_ref": "./data/themes/theme.yaml",
  "items": [
    {
      "photo_id": "a6d54a6157e85c3fce236acca72d7f67...",
      "rank": 1,
      "cluster_id": "m_0001",
      "role": "opener",
      "scores": {
        "Q_tech": 0.82,
        "Aesthetic": 0.73,
        "Vibe": 0.64,
        "Typography": 0.47,
        "Composition": 0.75,
        "Total": 0.76
      },
      "diversity_tags": [
        "scene:street",
        "palette:warm",
        "time:dusk",
        "people:2",
        "orient:landscape"
      ],
      "reasons": [
        "Cluster hero (highest quality in group)",
        "Excellent technical scores across dimensions",
        "Diverse representation in scene and time"
      ],
      "artifacts": {
        "std": "./data/processed/a6d54a6157e85c3fce236acca72d7f67_1024.jpg"
      }
    }
  ],
  "audit": {
    "created_at": "2025-09-20T14:37:03Z",
    "optimizer_params": {
      "alpha": 1.0,
      "beta": 1.0,
      "gamma": 1.0
    }
  }
}
```

## Diversity Tagging System

### Scene Type Tags
- **street:** Urban environments, city scenes
- **architecture:** Buildings, structures, design
- **people:** Portraits, groups, human subjects
- **interior:** Indoor spaces, rooms, environments
- **nature:** Landscapes, outdoors, natural scenes

### Palette Tags
- **warm:** Reds, oranges, yellows dominant
- **cool:** Blues, greens, purples dominant
- **monochrome:** Black, white, gray dominant
- **vibrant:** High saturation, colorful
- **muted:** Low saturation, subdued colors

### Time of Day Tags
- **dawn:** Early morning, sunrise
- **morning:** Late morning light
- **afternoon:** Daytime, direct sun
- **dusk:** Evening, sunset
- **night:** Low light, nighttime

### People Count Tags
- **0:** No people visible
- **1:** Single person
- **2:** Couple or duo
- **3+:** Groups of three or more

### Orientation Tags
- **landscape:** Wider than tall (horizontal)
- **portrait:** Taller than wide (vertical)
- **square:** Approximately equal dimensions

## Layout Role System

### Role Definitions
- **opener:** First photo, sets context and tone
- **hero:** Main focal photos, highest visual impact
- **body:** Supporting photos, maintain narrative flow
- **anchor:** Final photo, provides conclusion
- **connector:** Transitional photos between sections

### Role Assignment Logic
- Position-based: First photo = opener, last = anchor
- Quality-based: Highest scoring = hero
- Diversity-based: Spread different types across roles
- Theme-based: Match editorial requirements

## Audit Trail Features

### Creation Metadata
- **Timestamp:** When curation was completed
- **Version:** Software version used
- **Parameters:** Optimizer settings and constraints

### Decision Tracking
- **Selection Reasons:** Why each photo was chosen
- **Quality Scores:** Technical and aesthetic metrics
- **Marginal Gains:** Value added by each selection

---

# Generate Ingest Input Service

## Overview

The Generate Ingest Input Service scans photo directories and creates structured input for the photo curation pipeline. It serves as the bridge between raw photo collections and the automated curation system, handling file discovery, validation, and metadata preparation.

## Key Responsibilities

1. **Directory Scanning:** Recursively scan photo directories
2. **File Validation:** Identify and validate image files
3. **Batch Creation:** Generate structured ingest input
4. **Metadata Extraction:** Basic file information gathering
5. **Progress Tracking:** Monitor scanning and processing status

## Processing Workflow

### 1. Directory Scanning
- Recursively traverse input directories
- Identify files with supported image extensions
- Filter out non-image files and directories

### 2. File Validation
- Check file existence and accessibility
- Validate image format compatibility
- Generate file size and modification metadata

### 3. Batch Generation
- Create structured batch configuration
- Assign batch IDs with timestamps
- Include user overrides and theme specifications

### 4. Output Creation
- Generate ingest_input.json format
- Save to intermediate directory
- Provide processing statistics and summaries

## Supported Image Formats

- **JPEG:** .jpg, .jpeg
- **PNG:** .png
- **TIFF:** .tiff, .tif
- **BMP:** .bmp
- **GIF:** .gif
- **HEIC/HEIF:** .heic, .heif
- **WebP:** .webp

## Input Parameters

```python
def process(self, batch_id: str, input_dir: str = "./data/input/")
```

- **batch_id:** Unique identifier for the photo batch
- **input_dir:** Directory path to scan for photos (default: "./data/input/")

## Output Format

```json
{
  "batch_id": "batch_20250920_143703",
  "photos": [
    {
      "uri": "./data/input/photo1.jpg"
    },
    {
      "uri": "./data/input/photo2.png"
    }
  ],
  "theme_spec_ref": "./data/themes/default.yaml",
  "user_overrides": {
    "lock_in": [],
    "exclude": []
  }
}
```

## Directory Structure Expectations

```
/data/
├── input/          # Photos to be processed
│   ├── event1/
│   │   ├── photo1.jpg
│   │   ├── photo2.png
│   │   └── ...
│   └── event2/
│       └── ...
└── themes/         # Theme specification files
    └── default.yaml
```

## Processing Features

### Recursive Scanning
- Traverses subdirectories automatically
- Maintains relative path structure
- Handles symbolic links appropriately

### File Filtering
- Extension-based format detection
- Size validation (optional minimums)
- Modification date tracking

### Batch Configuration
- Automatic batch ID generation with timestamps
- Default theme specification
- Empty user overrides template

## Performance Characteristics

### Scanning Performance
- **Speed:** Thousands of files per minute
- **Memory:** Minimal memory footprint
- **CPU:** Lightweight file system operations

### Scalability
- **Large Directories:** Handles 10,000+ files efficiently
- **Deep Hierarchies:** Processes nested directory structures
- **Network Drives:** Works with remote file systems

## Error Handling

### File System Issues
- **Permission Errors:** Skip inaccessible files with warnings
- **Missing Directories:** Create input directory if needed
- **Corrupt Files:** Log and continue processing

### Configuration Issues
- **Invalid Paths:** Validate directory existence
- **Empty Directories:** Handle gracefully with appropriate messaging
- **Format Conflicts:** Skip unsupported file types

## Logging and Monitoring

### Log Files
- **Service Logs:** `intermediateJsons/ingest/generate_ingest_input_service.log`
- **Console Output:** Progress indicators and statistics

### Performance Metrics
- **Files Scanned:** Total files discovered
- **Images Found:** Valid image files identified
- **Processing Time:** Directory scanning duration
- **Error Count:** Files skipped due to errors

## Integration Points

### Upstream Integration
- **File System:** Direct directory scanning
- **External Tools:** Can be triggered by file watchers
- **Batch Systems:** Integrates with automated workflows

### Downstream Integration
- **Ingest Service:** Primary consumer of generated input
- **Orchestrator:** Can be called as first pipeline step
- **Monitoring:** Feeds metrics to observability systems

## Best Practices

### Directory Organization
1. **Consistent Structure:** Use predictable directory layouts
2. **Naming Conventions:** Descriptive folder and file names
3. **Size Management:** Keep batches to manageable sizes
4. **Backup Strategy:** Maintain original file integrity

### Batch Management
1. **Logical Grouping:** Group related photos together
2. **Size Optimization:** Balance batch size with processing time
3. **Incremental Processing:** Support resumable batch processing
4. **Version Control:** Track batch configurations over time

## Use Cases

### Event Photography
- **Input:** Multiple subfolders from different event segments
- **Processing:** Scan entire event directory structure
- **Output:** Consolidated batch for unified curation

### Portfolio Management
- **Input:** Photographer's organized project folders
- **Processing:** Respect existing organization while flattening for processing
- **Output:** Structured input maintaining project context

### Automated Workflows
- **Input:** Watch directory for new photo uploads
- **Processing:** Trigger batch generation on file changes
- **Output:** Seamless integration with CI/CD pipelines

## Future Enhancements

### Advanced Features
- **Metadata Extraction:** Include basic EXIF in batch generation
- **Quality Pre-check:** Initial quality assessment during scanning
- **Duplicate Detection:** Identify potential duplicates during scanning
- **Smart Batching:** Intelligent batch size optimization

### Integration Features
- **Webhook Support:** Notify external systems of batch creation
- **API Integration:** RESTful interface for remote triggering
- **Queue Integration:** Push batches to processing queues

### Monitoring Features
- **Real-time Progress:** WebSocket-based progress updates
- **Batch Analytics:** Statistics and insights on batch composition
- **Error Dashboards:** Visual error tracking and resolution

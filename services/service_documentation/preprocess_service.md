# Preprocess Service Documentation

## Overview

The Preprocess Service creates standardized versions of photos without quality loss. It serves as a critical bridge between the ingest service's raw file handling and the feature extraction/analysis pipeline, ensuring all photos are in a consistent, high-quality format for downstream processing.

## Key Responsibilities

1. **Image Standardization:** Convert all photos to consistent format and quality
2. **Size Optimization:** Resize large images to reasonable dimensions
3. **Quality Preservation:** Maintain high JPEG quality settings
4. **Aspect Ratio Maintenance:** Preserve original proportions
5. **Processing Metadata:** Track transformation details for audit trails

## Processing Workflow

### 1. Input Validation
- Receives photo index from ingest service
- Validates existence of ranking-ready files
- Handles both artifact and photo_index input formats

### 2. Image Analysis
- Load images using PIL for format detection
- Extract original dimensions and color profiles
- Assess need for resizing based on size thresholds

### 3. Standardization Process
- Convert color modes to RGB for consistency
- Apply high-quality resizing when needed
- Preserve aspect ratios during transformations
- Save with optimized JPEG settings

### 4. Metadata Generation
- Record original and processed dimensions
- Track processing method and parameters
- Generate processing metadata for audit trails

## Image Processing Details

### Resize Thresholds
- **Maximum Dimension:** 2048 pixels (width or height)
- **Trigger Condition:** Either dimension exceeds 2048px
- **Method:** Only resize when necessary to preserve quality

### Quality Settings
- **JPEG Quality:** 95 (high quality, minimal compression)
- **Optimization:** Enabled for file size reduction
- **Resampling:** Lanczos algorithm for high-quality downsampling

### Color Mode Handling
- **Source Modes:** RGB, RGBA, CMYK, Grayscale, etc.
- **Target Mode:** RGB for consistency
- **Profile Preservation:** Maintain color accuracy during conversion

## Input Format

The service accepts two input formats:

### From Ingest Service (Preferred)
```json
{
  "batch_id": "batch_20250920_143703",
  "artifacts": [
    {
      "photo_id": "a6d54a6157e85c3fce236acca72d7f67...",
      "original_uri": "./data/input/photo1.jpg",
      "ranking_uri": "./data/rankingInput/a6d54a6157e85c3fce236acca72d7f67.jpg",
      "exif": {...}
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
      "uri": "./data/rankingInput/photo.jpg",
      "exif": {...}
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
      "ranking_uri": "./data/rankingInput/a6d54a6157e85c3fce236acca72d7f67.jpg",
      "std_uri": "./data/rankingInput/a6d54a6157e85c3fce236acca72d7f67_1024.jpg",
      "processing_metadata": {
        "original_size": [3264, 2448],
        "standardized_size": [2048, 1536],
        "processing_method": "quality_preserved"
      }
    }
  ]
}
```

## Directory Structure

```
/data/
├── rankingInput/   # Input: From ingest service
│                  # Output: Standardized versions with _1024 suffix
├── processed/     # Alternative output location (configurable)
/intermediateJsons/preprocess/  # Service outputs and logs
```

## File Naming Convention

### Input Files
- From ingest: `{photo_id}.jpg`
- Example: `a6d54a6157e85c3fce236acca72d7f67.jpg`

### Output Files
- Standardized: `{photo_id}_1024.jpg`
- Example: `a6d54a6157e85c3fce236acca72d7f67_1024.jpg`

## Processing Logic

### Size Assessment Algorithm
```python
def needs_resize(width, height):
    return width > 2048 or height > 2048

def calculate_new_dimensions(width, height):
    if width > height:
        new_width = 2048
        new_height = int(height * (2048 / width))
    else:
        new_height = 2048
        new_width = int(width * (2048 / height))
    return new_width, new_height
```

### Image Processing Pipeline
1. **Load:** Open image with PIL
2. **Convert:** Ensure RGB color mode
3. **Assess:** Check if resizing needed
4. **Resize:** Apply Lanczos resampling if required
5. **Save:** Export with high quality settings
6. **Record:** Log processing metadata

## Error Handling

### File Processing Errors
- **Corrupt Images:** Return original URI, log error
- **Permission Issues:** Attempt alternative save locations
- **Memory Issues:** Process large images in chunks if needed

### Format Compatibility
- **Unsupported Formats:** Should be handled by ingest service
- **Color Profile Issues:** Graceful fallback to standard RGB
- **Metadata Preservation:** Maintain EXIF data when possible

## Performance Characteristics

### Processing Speed
- **Small Images (< 2MP):** Near-instant processing
- **Large Images (> 20MP):** 2-5 seconds with Lanczos resampling
- **Batch Efficiency:** Linear scaling with image count

### Memory Usage
- **Per-Image Memory:** Proportional to image size
- **Peak Usage:** Largest image in batch
- **Optimization:** Streaming processing for very large files

## Quality Preservation

### Resampling Quality
- **Algorithm:** Lanczos (best quality for downsampling)
- **Anti-aliasing:** Automatic edge smoothing
- **Artifact Prevention:** Maintains sharpness while reducing size

### Color Accuracy
- **Profile Handling:** Preserve ICC color profiles when possible
- **Gamma Correction:** Maintain proper brightness levels
- **Bit Depth:** 24-bit RGB output for consistency

## Dependencies

- **PIL (Pillow):** Core image processing and manipulation
- **NumPy:** Optional for advanced image operations
- **Image Formats:** Built-in JPEG support with optimization

## Logging and Monitoring

### Log Files
- **Service Logs:** `intermediateJsons/preprocess/preprocess_service.log`
- **Batch Outputs:** `intermediateJsons/preprocess/{batch_id}_preprocess_output.json`

### Performance Metrics
- **Processing Time:** Per-image and total batch time
- **Resize Statistics:** Count and percentage of resized images
- **Quality Metrics:** File size ratios and dimension changes
- **Error Rates:** Failed processing counts and reasons

## Configuration Options

### Size Thresholds
```python
MAX_DIMENSION = 2048  # Maximum width or height
QUALITY = 95          # JPEG quality (1-100)
OPTIMIZE = True       # JPEG optimization
RESAMPLING = Image.Resampling.LANCZOS
```

### Output Formats
- **Primary:** JPEG with quality optimization
- **Alternatives:** PNG for lossless (configurable)
- **Naming:** Configurable suffix patterns

## Integration Points

### Upstream Services
- **Ingest Service:** Primary input source with ranking-ready files
- **Direct Input:** Can process raw photo index for testing

### Downstream Services
- **Features Service:** Consumes standardized images for analysis
- **Cache System:** Uses photo IDs for processing result caching
- **Quality Control:** Provides processing metadata for audit trails

## Best Practices

### Image Quality
1. **Minimal Intervention:** Only resize when absolutely necessary
2. **Quality First:** Use highest quality settings available
3. **Aspect Preservation:** Never crop or distort aspect ratios
4. **Color Fidelity:** Maintain original color characteristics

### Performance Optimization
1. **Conditional Processing:** Skip processing for already-standard images
2. **Batch Processing:** Process multiple images efficiently
3. **Memory Management:** Handle large images appropriately
4. **Caching Strategy:** Cache processing results when possible

### Error Recovery
1. **Graceful Degradation:** Continue processing despite individual failures
2. **Fallback Options:** Alternative processing methods for edge cases
3. **Detailed Logging:** Comprehensive error reporting for debugging
4. **Recovery Mechanisms:** Ability to reprocess failed items

## Use Cases

### Standard Photography Workflow
- **Input:** Mixed format photos from various cameras
- **Processing:** Convert to consistent JPEG format
- **Output:** Standardized images ready for feature extraction

### Large Batch Processing
- **Input:** Thousands of photos from events
- **Processing:** Efficient batch resizing and standardization
- **Output:** Optimized collection for curation pipeline

### Quality Preservation
- **Input:** High-resolution professional photos
- **Processing:** Maintain quality while optimizing for analysis
- **Output:** Analysis-ready images with preserved visual quality

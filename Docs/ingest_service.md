# Ingest Service Documentation

## Overview

The Ingest Service is the entry point of the photo curation pipeline. It registers photos in a batch, verifies media files, extracts comprehensive metadata, and prepares them for downstream processing. This service handles the critical first step of converting various photo formats into a standardized, processable format.

## Key Responsibilities

1. **Photo Registration:** Validate and register photo URIs in a batch
2. **Format Conversion:** Convert RAW, HEIC, and other formats to JPEG
3. **Metadata Extraction:** Extract comprehensive EXIF and technical metadata
4. **Content Addressing:** Generate SHA256-based photo IDs
5. **File Preparation:** Create ranking-ready copies with consistent formatting

## Supported Formats

### Native PIL Formats (Direct Processing)
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)
- GIF (.gif)
- WebP (.webp)
- HEIC/HEIF (.heic, .heif) - via pillow-heif

### RAW and Specialized Formats (FFmpeg Processing)
- RAW formats (.raw, .cr2, .nef, .arw, .dng, .orf, .rw2, .raf)
- Additional formats requiring conversion

## Processing Workflow

### 1. Input Validation
- Verifies file existence for each photo URI
- Validates batch structure and required fields
- Handles missing files gracefully with error logging

### 2. Format Detection and Conversion
- Detects file format from extension and content
- Uses PIL for direct format support
- Falls back to FFmpeg for RAW and specialized formats
- Generates high-quality JPEG output (quality=95)

### 3. EXIF Metadata Extraction
- **Multi-method approach** for maximum compatibility:
  - Primary: piexif library for standard EXIF data
  - Secondary: PIL fallback for basic metadata
  - Tertiary: ExifTool for comprehensive RAW/HEIC support

### 4. Content-Based ID Generation
- Generates SHA256 hash of converted JPEG content
- Ensures content-addressed, deterministic photo IDs
- Enables caching and duplicate detection

## Metadata Extraction Details

### Camera Information
- **Make/Model:** Camera manufacturer and model
- **Lens:** Lens make and model (when available)
- **Serial Numbers:** Camera and lens serial numbers

### Technical Parameters
- **ISO:** Light sensitivity setting
- **Aperture:** F-stop value (e.g., f/2.8)
- **Shutter Speed:** Exposure time (e.g., 1/250s)
- **Focal Length:** Lens focal length (e.g., 50mm)

### GPS Data
- **Coordinates:** Latitude and longitude in decimal degrees
- **Altitude:** Elevation data when available
- **Hemisphere:** Automatic N/S/E/W conversion

### Additional Metadata
- **DateTime:** Original capture timestamp
- **Orientation:** Image rotation metadata
- **Color Space:** Color profile information

## Input Format

```json
{
  "batch_id": "batch_20250920_143703",
  "photos": [
    {
      "uri": "./data/input/photo1.jpg"
    },
    {
      "uri": "./data/input/photo2.HEIC"
    }
  ]
}
```

## Output Format

```json
{
  "batch_id": "batch_20250920_143703",
  "photo_index": [
    {
      "photo_id": "a6d54a6157e85c3fce236acca72d7f67...",
      "original_uri": "./data/input/photo1.jpg",
      "ranking_uri": "./data/rankingInput/a6d54a6157e85c3fce236acca72d7f67.jpg",
      "exif": {
        "camera": "Sony ILCE-7M4",
        "lens": "FE 50mm F1.8",
        "iso": 100,
        "aperture": "f/2.8",
        "shutter_speed": "1/250",
        "focal_length": "50mm",
        "datetime": "2025-09-20T14:37:03",
        "gps": {
          "lat": 37.7749,
          "lon": -122.4194,
          "alt": 15.2
        }
      },
      "format": ".jpg"
    }
  ]
}
```

## Directory Structure

```
/data/
├── input/          # Original photos (read-only)
├── rankingInput/   # Processed JPEG copies (created)
/cache/            # Temporary processing files
/intermediateJsons/ingest/  # Service outputs and logs
```

## Error Handling

### File Processing Errors
- **Missing Files:** Logged as warnings, processing continues
- **Corrupt Files:** Logged as errors, photo skipped
- **Permission Issues:** Logged as errors, photo skipped

### Metadata Extraction Errors
- **EXIF Corruption:** Falls back to alternative extraction methods
- **Unsupported Formats:** Uses ExifTool as universal fallback
- **GPS Issues:** Graceful degradation to coordinate-less metadata

## Performance Characteristics

- **Batch Processing:** Handles hundreds of photos efficiently
- **Progressive Conversion:** Processes files sequentially with progress indicators
- **Memory Efficient:** Streams file content without loading all into memory
- **Caching Ready:** Content-based IDs enable result caching

## Dependencies

- **PIL (Pillow):** Primary image processing and format conversion
- **piexif:** EXIF data extraction and manipulation
- **pillow-heif:** HEIC/HEIF format support
- **ExifTool:** Comprehensive metadata extraction for RAW files
- **FFmpeg:** Video format conversion and RAW processing

## Logging and Monitoring

### Log Files
- **Service Logs:** `intermediateJsons/ingest/ingest_service.log`
- **Batch Outputs:** `intermediateJsons/ingest/{batch_id}_ingest_output.json`

### Log Levels
- **INFO:** Successful processing milestones
- **WARNING:** Non-critical issues (missing files, metadata extraction failures)
- **ERROR:** Critical failures requiring attention

### Performance Metrics
- Processing time per photo
- Success/failure counts
- Format conversion statistics
- Metadata extraction coverage

## Configuration Options

### Supported Extensions
```python
self.pil_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp', '.heic', '.heif'}
self.ffmpeg_formats = {'.raw', '.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2', '.raf'}
```

### Conversion Quality
- **JPEG Quality:** 95 (high quality, minimal compression)
- **Resize Threshold:** 2048px maximum dimension (handled by preprocess service)
- **Color Mode:** RGB conversion for consistency

## Integration Points

### Upstream Services
- **Input Source:** File system or external photo management systems
- **Batch Generator:** Can consume outputs from `generate_ingest_input_service.py`

### Downstream Services
- **Preprocess Service:** Consumes `ranking_uri` for standardization
- **Cache System:** Uses `photo_id` for result caching
- **Monitoring:** Feeds metrics to observability systems

## Best Practices

1. **Input Validation:** Always validate file existence before batch processing
2. **Format Diversity:** Support wide range of formats for user flexibility
3. **Metadata Preservation:** Extract as much metadata as possible for curation decisions
4. **Content Addressing:** Use content-based IDs for reliable duplicate detection
5. **Error Resilience:** Continue processing despite individual file failures
6. **Performance Monitoring:** Track processing times and success rates
7. **Log Aggregation:** Centralize logs for debugging and optimization

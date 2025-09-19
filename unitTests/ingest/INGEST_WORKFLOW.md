# Ingest Service Workflow

The `ingest_service.py` handles the first step of the photo curation pipeline:

## Workflow

```
ingest_input.json ──► Ingest Service ──► rankingInput/ + ingest_output.json
                        │
                        ▼
                 intermediateJsons/ingest/
```

## Step-by-Step Process

1. **Input**: Place your photos in `data/input/` directory
2. **Create Input File**: Create `data/ingest_input.json` with photo references
3. **Run Ingest**: The service processes each photo
4. **Output**: Processed images saved to `data/rankingInput/` + metadata to `intermediateJsons/ingest/`

## Input File Format

```json
{
  "batch_id": "my_photo_batch",
  "photos": [
    {"uri": "./data/input/vacation1.jpg"},
    {"uri": "./data/input/vacation2.JPG"},
    {"uri": "./data/input/family_photo.png"}
  ]
}
```

## What the Ingest Service Does

- ✅ Converts all images to high-quality JPEG format
- ✅ Extracts EXIF metadata (camera, lens, GPS, etc.)
- ✅ Generates content-addressable IDs (SHA256 hashes)
- ✅ Handles various formats: JPEG, PNG, HEIC, RAW, etc.
- ✅ Preserves quality while standardizing format
- ✅ Saves processed images to `data/rankingInput/` for ranking

## Output

After processing, you'll get:

1. **Processed Images**: `data/rankingInput/{photo_id}.jpg`
2. **Metadata File**: `intermediateJsons/ingest/{batch_id}_ingest_output.json`

## Testing

Run the test script to see it in action:

```bash
python test_ingest_workflow.py
```

## Next Steps

After ingest completes, the `preprocess_service.py` can further optimize images if needed, and then the full curation pipeline (features → scoring → clustering → ranking → optimizer → exporter) can run.

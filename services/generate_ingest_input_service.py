"""
Generate Ingest Input Service

Service for scanning directories and creating ingest input JSON for the photo curation pipeline.
"""

import os
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List


class GenerateIngestInputService:
    """Service for generating ingest input from photo directories."""

    def __init__(self):
        # Supported image extensions
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif', '.heic', '.heif', '.webp'}

    def process(self, batch_id: str, input_dir: str = "./data/input/") -> Dict[str, Any]:
        """Generate ingest input from photos in a directory."""

        photos = []

        if not os.path.exists(input_dir):
            print(f"❌ Input directory {input_dir} does not exist")
            return None

        # Scan directory for image files
        for filename in os.listdir(input_dir):
            if not os.path.isfile(os.path.join(input_dir, filename)):
                continue

            # Check if it's an image file
            _, ext = os.path.splitext(filename.lower())
            if ext in self.image_extensions:
                # Create content-addressable photo_id
                file_path = os.path.join(input_dir, filename)
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                    photo_id = hashlib.sha256(file_content).hexdigest()

                photos.append({
                    "uri": f"./data/input/{filename}",
                    "photo_id": photo_id  # Include photo_id for reference
                })

        if not photos:
            print(f"⚠️  No image files found in {input_dir}")
            return None

        # Sort photos by filename for consistent ordering
        photos.sort(key=lambda x: x["uri"])

        ingest_input = {
            "batch_id": batch_id,
            "photos": [{"uri": photo["uri"]} for photo in photos],  # Remove photo_id from input
            "theme_spec_ref": f"./data/themes/{batch_id}_theme.yaml",
            "user_overrides": {
                "lock_in": [],  # Add specific photo filenames to lock in
                "exclude": []   # Add specific photo filenames to exclude
            }
        }

        print(f"✅ Generated ingest input for {len(photos)} photos:")
        for photo in photos[:5]:  # Show first 5
            print(f"  📸 {photo['uri']} -> {photo['photo_id'][:8]}...")
        if len(photos) > 5:
            print(f"  ... and {len(photos) - 5} more")

        return ingest_input

    def save_ingest_input(self, ingest_input: Dict[str, Any], output_file: str = "./data/ingest_input.json"):
        """Save the ingest input to a JSON file."""
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w') as f:
            import json
            json.dump(ingest_input, f, indent=2)

        print(f"💾 Ingest input saved to {output_file}")

    def process_and_save(self, batch_id: str, input_dir: str = "./data/input/", output_file: str = "./data/ingest_input.json"):
        """Process ingest input and save to file."""
        ingest_input = self.process(batch_id, input_dir)
        if ingest_input:
            self.save_ingest_input(ingest_input, output_file)
        return ingest_input


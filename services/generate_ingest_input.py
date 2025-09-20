"""
Generate Ingest Input for Real Photos

This script scans the data/input/ directory and creates a proper ingest input
JSON that can be used with the real services.
"""

import os
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List


def generate_real_ingest_input(batch_id: str, input_dir: str = "./data/input/") -> Dict[str, Any]:
    """Generate ingest input from actual files in the input directory."""

    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif', '.heic', '.heif', '.webp'}

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
        if ext in image_extensions:
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


def save_ingest_input(ingest_input: Dict[str, Any], output_file: str = "./data/ingest_input.json"):
    """Save the ingest input to a JSON file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(ingest_input, f, indent=2)

    print(f"💾 Ingest input saved to {output_file}")


def main():
    """Main function to generate ingest input."""
    batch_id = f"batch_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')}"

    print("🔍 Scanning data/input/ for photos...")
    ingest_input = generate_real_ingest_input(batch_id)

    if ingest_input:
        save_ingest_input(ingest_input)

        # Also save to mock_data for reference
        mock_data_path = "./mock_data/real_ingest_input.json"
        save_ingest_input(ingest_input, mock_data_path)

        print("\n🚀 Ready to run the pipeline:")
        print("python -c \"from orchestrator import Orchestrator; o = Orchestrator(); o.start_batch(ingest_input)\"")
    else:
        print("❌ No photos found. Please place photos in data/input/ and run again.")


if __name__ == "__main__":
    main()

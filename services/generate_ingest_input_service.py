"""
Generate Ingest Input Service

Service for scanning directories and creating ingest input JSON for the photo curation pipeline.
"""

import os
import json
import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional


class GenerateIngestInputService:
    """Service for generating ingest input from photo directories."""

    def __init__(self):
        # Setup logging
        self.logger = logging.getLogger('GenerateIngestInputService')
        self.logger.setLevel(logging.DEBUG)

        # Create logs directory if it doesn't exist
        log_dir = "intermediateJsons/ingest"
        os.makedirs(log_dir, exist_ok=True)

        # File handler - logs everything
        file_handler = logging.FileHandler(os.path.join(log_dir, 'generate_ingest_input_service.log'))
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # Console handler - only errors
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Supported image extensions
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif', '.heic', '.heif', '.webp'}

    def process(self, batch_id: str, input_dir: str = "./data/input/") -> Optional[Dict[str, Any]]:
        """Generate ingest input from photos in a directory."""
        start_time = time.time()
        print("🔍 Scanning directory for images...")

        photos = []

        if not os.path.exists(input_dir):
            self.logger.error(f"Input directory {input_dir} does not exist")
            return None

        # Scan directory for image files
        all_files = os.listdir(input_dir)
        total_files = len(all_files)

        print(f"📂 Scanning {total_files} files...")

        for i, filename in enumerate(all_files):
            if not os.path.isfile(os.path.join(input_dir, filename)):
                continue

            # Check if it's an image file
            _, ext = os.path.splitext(filename.lower())
            if ext in self.image_extensions:
                # Progress indicator for every 10 files or so
                if (i + 1) % max(1, total_files // 10) == 0 or i == total_files - 1:
                    print(f"\r🔄 Scanned {i+1}/{total_files} files... {filename}", end="", flush=True)

                # Create content-addressable photo_id
                file_path = os.path.join(input_dir, filename)
                try:
                    with open(file_path, 'rb') as f:
                        file_content = f.read()
                        photo_id = hashlib.sha256(file_content).hexdigest()

                    photos.append({
                        "uri": f"./data/input/{filename}",
                        "photo_id": photo_id  # Include photo_id for reference
                    })
                except Exception as e:
                    print(f"\r⚠️  Error reading {filename}: {e}")
                    self.logger.warning(f"Error reading {filename}: {e}")
                    continue

        # Clear progress line and show results
        print(f"\r✅ Found {len(photos)} image files from {total_files} total files")

        if not photos:
            self.logger.warning(f"No image files found in {input_dir}")
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

        # Calculate and display timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        timing_msg = f"Generated ingest input for {len(photos)} photos in {elapsed_time:.2f}s"
        print(f"📤 {timing_msg}")
        self.logger.info(timing_msg)
        for photo in photos[:5]:  # Log first 5
            self.logger.debug(f"  {photo['uri']} -> {photo['photo_id'][:8]}...")
        if len(photos) > 5:
            self.logger.debug(f"  ... and {len(photos) - 5} more")

        return ingest_input

    def save_ingest_input(self, ingest_input: Dict[str, Any], output_file: str = "./data/ingest_input.json"):
        """Save the ingest input to a JSON file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(ingest_input, f, indent=2)

        self.logger.info(f"Ingest input saved to {output_file}")

    def process_and_save(self, batch_id: str, input_dir: str = "./data/input/", output_file: str = "./data/ingest_input.json"):
        """Process ingest input and save to file."""
        ingest_input = self.process(batch_id, input_dir)
        if ingest_input:
            self.save_ingest_input(ingest_input, output_file)
        return ingest_input


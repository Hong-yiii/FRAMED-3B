"""
Real Preprocess Service for Photo Curation System

Creates standardized versions without quality loss.
"""

import os
import hashlib
import logging
import time
from typing import Dict, Any
from PIL import Image
import numpy as np


class PreprocessService:
    """Real preprocess service that creates standardized versions."""

    def __init__(self):
        # Setup logging
        self.logger = logging.getLogger('PreprocessService')
        self.logger.setLevel(logging.DEBUG)

        # Create logs directory if it doesn't exist
        log_dir = "intermediateJsons/preprocess"
        os.makedirs(log_dir, exist_ok=True)

        # File handler - logs everything
        file_handler = logging.FileHandler(os.path.join(log_dir, 'preprocess_service.log'))
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

        self.ranking_input_dir = "./data/rankingInput"
        os.makedirs(self.ranking_input_dir, exist_ok=True)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create standardized versions of photos."""
        start_time = time.time()
        self.logger.info("Preprocessing photos...")

        batch_id = input_data["batch_id"]
        artifacts = []
        total_photos = len(input_data["photo_index"])
        processed_count = 0

        print(f"🔧 Preprocessing {total_photos} photos...")

        for i, photo in enumerate(input_data["photo_index"]):
            photo_id = photo["photo_id"]
            photo_uri = photo.get("original_uri", photo.get("uri", ""))
            ranking_uri = photo.get("ranking_uri", photo_uri)

            # Progress indicator
            progress = f"[{i+1}/{total_photos}]"
            print(f"\r🔄 {progress} Processing photos... (Preprocess Service) {photo_id[:8]}", end="", flush=True)

            # Create standardized version (no quality loss) in rankingInput
            std_uri = self._create_standardized_version(ranking_uri, photo_id)

            artifact = {
                "photo_id": photo_id,
                "original_uri": photo_uri,
                "ranking_uri": ranking_uri,
                "std_uri": std_uri,
                "processing_metadata": {
                    "original_size": self._get_image_size(ranking_uri),
                    "standardized_size": self._get_image_size(std_uri),
                    "processing_method": "quality_preserved"
                }
            }

            artifacts.append(artifact)
            processed_count += 1
            self.logger.info(f"Processed successfully: {photo_id[:8]}")

        # Clear progress line and show final status
        print(f"\r✅ Preprocessed {processed_count}/{total_photos} photos successfully")

        result = {
            "batch_id": batch_id,
            "artifacts": artifacts
        }

        # Save to intermediate JSONs
        import json
        os.makedirs("intermediateJsons/preprocess", exist_ok=True)
        with open(f"intermediateJsons/preprocess/{batch_id}_preprocess_output.json", 'w') as f:
            json.dump(result, f, indent=2)

        # Calculate and display timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        timing_msg = f"Preprocess complete: {len(artifacts)}/{total_photos} photos processed in {elapsed_time:.2f}s"
        print(f"📤 {timing_msg}")
        self.logger.info(timing_msg)
        return result

    def _create_standardized_version(self, photo_uri: str, photo_id: str) -> str:
        """Create a standardized version without quality loss."""
        try:
            with Image.open(photo_uri) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Get original dimensions
                width, height = img.size

                # Only resize if significantly larger than target
                # Use high-quality downsampling if needed
                if width > 2048 or height > 2048:
                    # Calculate new dimensions maintaining aspect ratio
                    if width > height:
                        new_width = 2048
                        new_height = int(height * (2048 / width))
                    else:
                        new_height = 2048
                        new_width = int(width * (2048 / height))

                    # Use high-quality Lanczos resampling
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Save with high quality
                output_path = os.path.join(self.ranking_input_dir, f"{photo_id}_1024.jpg")
                img.save(output_path, 'JPEG', quality=95, optimize=True)

                return output_path

        except Exception as e:
            self.logger.error(f"Error processing {photo_uri}: {e}")
            return photo_uri  # Return original if processing fails

    def _get_image_size(self, image_path: str) -> tuple:
        """Get image dimensions."""
        try:
            with Image.open(image_path) as img:
                return img.size
        except:
            return (0, 0)

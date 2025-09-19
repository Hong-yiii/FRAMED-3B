"""
Real Ingest Service for Photo Curation System

WORKFLOW:
1. Reads ingest_input.json with photo URIs (./data/input/filename.jpg)
2. Processes each photo: converts to JPEG, extracts EXIF, generates IDs
3. Saves processed images to ./data/rankingInput/ directory
4. Outputs ingest_output.json with photo metadata to intermediateJsons/ingest/

INPUT FORMAT:
{
  "batch_id": "batch_name",
  "photos": [
    {"uri": "./data/input/photo1.jpg"},
    {"uri": "./data/input/photo2.JPG"}
  ]
}

OUTPUT FORMAT:
{
  "batch_id": "batch_name",
  "photo_index": [
    {
      "photo_id": "sha256_hash_of_processed_image",
      "original_uri": "./data/input/photo1.jpg",
      "ranking_uri": "./data/rankingInput/photo_id.jpg",
      "exif": {...},
      "format": ".jpg"
    }
  ]
}
"""

import os
import hashlib
import subprocess
from typing import Dict, Any, List
from datetime import datetime, timezone
from PIL import Image
import piexif


class IngestService:
    """Real ingest service that processes actual photo files."""

    def __init__(self):
        self.cache_dir = "./data/cache"
        self.ranking_input_dir = "./data/rankingInput"
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.ranking_input_dir, exist_ok=True)

        # Supported formats that PIL can handle directly
        self.pil_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}

        # Formats that need ffmpeg conversion
        self.ffmpeg_formats = {'.heic', '.heif', '.raw', '.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2', '.raf'}

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process actual photos from the input directory."""
        print("🔄 Real Ingest Service: Processing photos from data/input/")

        batch_id = input_data["batch_id"]
        photo_index = []

        # Process each photo URI
        for i, photo in enumerate(input_data["photos"]):
            photo_uri = photo["uri"]

            # Extract filename from URI
            if photo_uri.startswith("./data/input/"):
                filename = photo_uri.replace("./data/input/", "")
                full_path = photo_uri
            else:
                filename = os.path.basename(photo_uri)
                full_path = f"./data/input/{filename}"

            # Check if file exists
            if not os.path.exists(full_path):
                print(f"⚠️  File not found: {full_path}")
                continue

            try:
                # Generate initial photo_id from original file
                with open(full_path, 'rb') as f:
                    file_content = f.read()
                    initial_photo_id = hashlib.sha256(file_content).hexdigest()

                # Convert to JPEG if needed and copy to rankingInput
                ranking_uri = self._prepare_for_ranking(full_path, filename, initial_photo_id)

                # Generate final photo_id based on converted file content
                with open(ranking_uri, 'rb') as f:
                    converted_content = f.read()
                    photo_id = hashlib.sha256(converted_content).hexdigest()

                # Extract EXIF data from original file
                exif_data = self._extract_exif_data(full_path)

                photo_index.append({
                    "photo_id": photo_id,
                    "original_uri": photo_uri,
                    "ranking_uri": ranking_uri,
                    "exif": exif_data,
                    "format": self._get_file_format(filename)
                })

                print(f"✅ Processed: {filename} -> {photo_id[:8]}...")

            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
                continue

        result = {
            "batch_id": batch_id,
            "photo_index": photo_index
        }

        # Save to intermediate JSONs
        import json
        os.makedirs("intermediateJsons/ingest", exist_ok=True)
        with open(f"intermediateJsons/ingest/{batch_id}_ingest_output.json", 'w') as f:
            json.dump(result, f, indent=2)

        print(f"✅ Real Ingest Service: Successfully processed {len(photo_index)} photos")
        print(f"💾 Saved output to intermediateJsons/ingest/{batch_id}_ingest_output.json")
        return result

    def _extract_exif_data(self, file_path: str) -> Dict[str, Any]:
        """Extract EXIF metadata from image file."""
        try:
            # Load EXIF data
            exif_dict = piexif.load(file_path)

            # Extract basic camera info
            exif_data = {
                "camera": "Unknown",
                "lens": "Unknown",
                "iso": None,
                "aperture": "Unknown",
                "shutter_speed": "Unknown",
                "focal_length": "Unknown",
                "datetime": None,
                "gps": None
            }

            # Extract camera info
            if "0th" in exif_dict:
                zeroth_ifd = exif_dict["0th"]
                if piexif.ImageIFD.Make in zeroth_ifd and piexif.ImageIFD.Model in zeroth_ifd:
                    exif_data["camera"] = f"{zeroth_ifd[piexif.ImageIFD.Make].decode()} {zeroth_ifd[piexif.ImageIFD.Model].decode()}"

            # Extract EXIF info
            if "Exif" in exif_dict:
                exif_ifd = exif_dict["Exif"]

                # ISO
                if piexif.ExifIFD.ISOSpeedRatings in exif_ifd:
                    exif_data["iso"] = exif_ifd[piexif.ExifIFD.ISOSpeedRatings]

                # Aperture
                if piexif.ExifIFD.FNumber in exif_ifd:
                    fnumber = exif_ifd[piexif.ExifIFD.FNumber]
                    if isinstance(fnumber, tuple):
                        exif_data["aperture"] = f"f/{fnumber[0]/fnumber[1]}"
                    else:
                        exif_data["aperture"] = f"f/{fnumber}"

                # Shutter speed
                if piexif.ExifIFD.ExposureTime in exif_ifd:
                    exposure_time = exif_ifd[piexif.ExifIFD.ExposureTime]
                    if isinstance(exposure_time, tuple):
                        exif_data["shutter_speed"] = f"1/{exposure_time[1]//exposure_time[0]}"
                    else:
                        exif_data["shutter_speed"] = f"1/{int(1/exposure_time)}"

                # Focal length
                if piexif.ExifIFD.FocalLength in exif_ifd:
                    focal_length = exif_ifd[piexif.ExifIFD.FocalLength]
                    if isinstance(focal_length, tuple):
                        exif_data["focal_length"] = f"{focal_length[0]//focal_length[1]}mm"
                    else:
                        exif_data["focal_length"] = f"{focal_length}mm"

                # DateTime
                if piexif.ExifIFD.DateTimeOriginal in exif_ifd:
                    exif_data["datetime"] = exif_ifd[piexif.ExifIFD.DateTimeOriginal].decode()

            # GPS data
            if "GPS" in exif_dict:
                gps_ifd = exif_dict["GPS"]
                if piexif.GPSIFD.GPSLatitude in gps_ifd and piexif.GPSIFD.GPSLongitude in gps_ifd:
                    lat = self._convert_gps_to_decimal(gps_ifd[piexif.GPSIFD.GPSLatitude])
                    lon = self._convert_gps_to_decimal(gps_ifd[piexif.GPSIFD.GPSLongitude])

                    if piexif.GPSIFD.GPSLatitudeRef in gps_ifd and gps_ifd[piexif.GPSIFD.GPSLatitudeRef].decode() == "S":
                        lat = -lat
                    if piexif.GPSIFD.GPSLongitudeRef in gps_ifd and gps_ifd[piexif.GPSIFD.GPSLongitudeRef].decode() == "W":
                        lon = -lon

                    exif_data["gps"] = {"lat": lat, "lon": lon}

        except Exception as e:
            print(f"⚠️  Error extracting EXIF from {file_path}: {e}")

        return exif_data

    def _convert_gps_to_decimal(self, gps_tuple):
        """Convert GPS coordinate tuple to decimal degrees."""
        degrees, minutes, seconds = gps_tuple
        if isinstance(degrees, tuple):
            degrees = degrees[0] / degrees[1]
        if isinstance(minutes, tuple):
            minutes = minutes[0] / minutes[1]
        if isinstance(seconds, tuple):
            seconds = seconds[0] / seconds[1]

        return degrees + (minutes / 60.0) + (seconds / 3600.0)

    def _get_file_format(self, filename: str) -> str:
        """Get the file format from filename."""
        _, ext = os.path.splitext(filename.lower())
        return ext

    def _prepare_for_ranking(self, source_path: str, filename: str, photo_id: str) -> str:
        """Prepare file for ranking by converting to JPEG if needed."""
        _, ext = os.path.splitext(filename.lower())

        # If already JPEG, just copy to rankingInput
        if ext in ['.jpg', '.jpeg']:
            ranking_filename = f"{photo_id}.jpg"
            ranking_path = os.path.join(self.ranking_input_dir, ranking_filename)

            # Copy file
            with open(source_path, 'rb') as src, open(ranking_path, 'wb') as dst:
                dst.write(src.read())

            return ranking_path

        # If PIL can handle it directly, convert with PIL
        elif ext in self.pil_formats:
            try:
                with Image.open(source_path) as img:
                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    ranking_filename = f"{photo_id}.jpg"
                    ranking_path = os.path.join(self.ranking_input_dir, ranking_filename)

                    # Save as high-quality JPEG
                    img.save(ranking_path, 'JPEG', quality=95, optimize=True)
                    return ranking_path
            except Exception as e:
                print(f"⚠️  PIL conversion failed for {filename}: {e}")

        # Try ffmpeg for other formats
        if self._has_ffmpeg():
            try:
                ranking_filename = f"{photo_id}.jpg"
                ranking_path = os.path.join(self.ranking_input_dir, ranking_filename)

                # Use ffmpeg to convert to JPEG
                cmd = [
                    'ffmpeg', '-i', source_path,
                    '-q:v', '2',  # High quality
                    '-y',  # Overwrite output
                    ranking_path
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

                if result.returncode == 0 and os.path.exists(ranking_path):
                    print(f"📹 Converted {filename} with ffmpeg")
                    return ranking_path
                else:
                    print(f"⚠️  ffmpeg conversion failed for {filename}: {result.stderr}")

            except subprocess.TimeoutExpired:
                print(f"⚠️  ffmpeg conversion timed out for {filename}")
            except Exception as e:
                print(f"⚠️  ffmpeg conversion failed for {filename}: {e}")

        # Fallback: try to copy as-is if it's a common image format
        try:
            ranking_filename = f"{photo_id}{ext}"
            ranking_path = os.path.join(self.ranking_input_dir, ranking_filename)

            with open(source_path, 'rb') as src, open(ranking_path, 'wb') as dst:
                dst.write(src.read())

            print(f"⚠️  Copied {filename} as-is (no conversion)")
            return ranking_path

        except Exception as e:
            print(f"❌ Failed to prepare {filename} for ranking: {e}")
            raise

    def _has_ffmpeg(self) -> bool:
        """Check if ffmpeg is available."""
        try:
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False

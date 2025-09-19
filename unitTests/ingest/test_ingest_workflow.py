#!/usr/bin/env python3
"""
Test script to demonstrate the ingest service workflow:
1. Reads ingest_input.json
2. Processes photos and saves to rankingInput/
3. Outputs ingest_output.json with metadata
"""

import json
import os
from services.ingest_service import IngestService

def test_ingest_workflow():
    """Test the complete ingest workflow."""

    # Load input data
    input_file = "./data/ingest_input.json"
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return

    with open(input_file, 'r') as f:
        input_data = json.load(f)

    print("📁 Loaded ingest input:")
    print(f"   Batch ID: {input_data['batch_id']}")
    print(f"   Photos: {len(input_data['photos'])}")

    # Initialize ingest service
    ingest_service = IngestService()

    # Process the photos
    print("🔄 Processing photos...")
    result = ingest_service.process(input_data)

    print("✅ Ingest processing complete!")
    print(f"   Processed: {len(result['photo_index'])} photos")
    print(f"   Ranking images saved to: {ingest_service.ranking_input_dir}/")
    print(f"   Metadata saved to: intermediateJsons/ingest/{result['batch_id']}_ingest_output.json")

    # Show sample output
    if result['photo_index']:
        sample = result['photo_index'][0]
        print("\n📊 Sample processed photo:")
        print(f"   Photo ID: {sample['photo_id'][:16]}...")
        print(f"   Original: {sample['original_uri']}")
        print(f"   Ranking: {sample['ranking_uri']}")
        print(f"   Format: {sample['format']}")
        print(f"   Camera: {sample['exif']['camera']}")

if __name__ == "__main__":
    test_ingest_workflow()

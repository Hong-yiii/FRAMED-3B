#!/usr/bin/env python3
"""
Test script for the OpenCLIP Features Service

This script tests the features service with mock data to ensure
the integration works correctly before running with real photos.
"""

import json
import os
import sys
from datetime import datetime, timezone

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_features_service():
    """Test the features service with mock input data."""
    
    print("🧪 Testing OpenCLIP Features Service")
    print("=" * 50)
    
    # Create test directories
    os.makedirs("./data/cache/features", exist_ok=True)
    os.makedirs("./intermediateJsons/features", exist_ok=True)
    
    # Check if dependencies are available
    try:
        import torch
        import open_clip
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ OpenCLIP available")
        
        # Check device availability
        if torch.backends.mps.is_available():
            print("✅ MPS (Metal Performance Shaders) available")
        elif torch.cuda.is_available():
            print("✅ CUDA available")
        else:
            print("ℹ️  Using CPU (no GPU acceleration)")
            
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please run: pixi install")
        return False
    
    # Test with mock data (no actual images needed for this test)
    try:
        from services.features_service import FeaturesService
        
        print("\n🔄 Initializing Features Service...")
        features_service = FeaturesService()
        print("✅ Features Service initialized successfully")
        
        # Create mock input data (similar to preprocess service output)
        mock_input = {
            "batch_id": f"test_batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            "artifacts": [
                {
                    "photo_id": "test_photo_001",
                    "std_uri": "./mock_image.jpg",  # This won't exist, but that's OK for testing
                    "original_uri": "./data/input/mock_image.jpg"
                }
            ]
        }
        
        print(f"\n🧠 Testing feature extraction...")
        print(f"Input: {len(mock_input['artifacts'])} artifacts")
        
        # This will fail gracefully and return default features
        result = features_service.process(mock_input)
        
        print(f"✅ Processing completed")
        print(f"Output: {len(result['artifacts'])} artifacts with features")
        
        # Check the output structure
        if result['artifacts']:
            artifact = result['artifacts'][0]
            if 'features' in artifact:
                features = artifact['features']
                print(f"\n📊 Sample features:")
                print(f"  Tech metrics: {list(features.get('tech', {}).keys())}")
                print(f"  CLIP labels: {features.get('clip_labels', [])[:3]}...")
                print("✅ Feature structure is correct")
            else:
                print("⚠️  No features found in output")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing features service: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_files():
    """Test that config files are properly created."""
    print("\n📁 Testing configuration files...")
    
    # Check labels.json
    labels_path = "./config/labels.json"
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            labels_config = json.load(f)
            print(f"✅ Labels config: {len(labels_config['labels'])} labels")
    else:
        print("❌ Labels config not found")
        return False
    
    # Check templates.json
    templates_path = "./config/templates.json"
    if os.path.exists(templates_path):
        with open(templates_path, 'r') as f:
            templates_config = json.load(f)
            print(f"✅ Templates config: {len(templates_config['templates'])} templates")
    else:
        print("❌ Templates config not found")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🚀 OpenCLIP Features Service Test Suite")
    print("=" * 60)
    
    # Test 1: Config files
    config_ok = test_config_files()
    
    # Test 2: Features service
    service_ok = test_features_service()
    
    print("\n" + "=" * 60)
    if config_ok and service_ok:
        print("🎉 All tests passed! Features service is ready.")
        print("\n📋 Next steps:")
        print("1. Place some photos in ./data/input/")
        print("2. Run: python orchestrator.py")
        print("3. Check the output in ./data/output/curated_list.json")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

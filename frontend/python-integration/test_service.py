#!/usr/bin/env python3
"""
Test script for the nameplate detection service
"""

import base64
import json
import sys
import os
from nameplate_service import NameplateDetectionService

def test_with_image_file(image_path):
    """Test the service with an image file"""
    try:
        # Read and encode image
        with open(image_path, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Initialize service
        service = NameplateDetectionService()
        
        # Test detection
        result = service.detect_nameplate(image_data)
        
        print(f"Testing with image: {image_path}")
        print(f"Result: {json.dumps(result, indent=2)}")
        
        return result
        
    except Exception as e:
        print(f"Error testing with image {image_path}: {e}")
        return None

def find_test_images():
    """Find available test images"""
    test_patterns = [
        "../data/nameplate_classification/test/has_nameplate/*.jpg",
        "../data/samples/*.jpg",
        "../data/experimento3_detection/test/images/*.jpg",
        "../*.jpg",
        "../*.png"
    ]
    
    import glob
    for pattern in test_patterns:
        files = glob.glob(pattern)
        if files:
            return files[:3]  # Return first 3 files
    
    return []

def main():
    """Main test function"""
    print("🧪 Testing Nameplate Detection Service")
    print("=" * 50)
    
    # Check if image path is provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            test_with_image_file(image_path)
        else:
            print(f"Error: Image file not found: {image_path}")
    else:
        # Find test images
        test_images = find_test_images()
        
        if not test_images:
            print("No test images found. Please provide an image path:")
            print("python test_service.py <image_path>")
            return
        
        print(f"Found {len(test_images)} test images")
        
        for i, image_path in enumerate(test_images):
            print(f"\nTest {i+1}:")
            test_with_image_file(image_path)

if __name__ == "__main__":
    main() 
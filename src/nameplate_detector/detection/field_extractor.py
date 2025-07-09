#!/usr/bin/env python3
"""
Test script for field extraction functionality
Tests the inference endpoint integration
"""

import cv2
import requests
import json
import io
from PIL import Image
import os
import glob

def test_field_extraction(image_path):
    """
    Test field extraction with a sample image
    
    Args:
        image_path: Path to test image
    """
    print(f"🧪 Testing field extraction with: {image_path}")
    
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Could not load image: {image_path}")
        return False
    
    try:
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Save image to bytes buffer
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        
        # Prepare the request
        url = 'http://localhost:8000/inference'
        files = {
            'image': ('nameplate.jpg', img_buffer, 'image/jpeg')
        }
        data = {
            'prompt': 'Extract all key-value pairs from this nameplate image and return them in JSON format. Include information like model numbers, serial numbers, voltage, current, power, manufacturer, and any other technical specifications visible.',
            'max_new_tokens': 512,
            'temperature': 0.7
        }
        
        print("🔍 Sending image for field extraction...")
        response = requests.post(url, files=files, data=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Field extraction successful!")
            print("\n" + "="*50)
            print("📋 EXTRACTED FIELDS:")
            print("="*50)
            print(json.dumps(result, indent=2))
            print("="*50)
            return True
        else:
            print(f"❌ API request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error during field extraction: {e}")
        return False
    except Exception as e:
        print(f"❌ Error during field extraction: {e}")
        return False

def find_test_images():
    """Find available test images"""
    test_patterns = [
        "data/nameplate_classification/test/has_nameplate/*.jpg",
        "data/samples/*.jpg",
        "data/experimento3_detection/test/images/*.jpg",
        "*.jpg",
        "*.png"
    ]
    
    for pattern in test_patterns:
        files = glob.glob(pattern)
        if files:
            return files[:5]  # Return first 5 files
    
    return []

def main():
    print("🚀 Field Extraction Test")
    print("=" * 50)
    
    # Check if inference endpoint is available
    try:
        response = requests.get('http://localhost:8000/health', timeout=5)
        if response.status_code == 200:
            print("✅ Inference endpoint is available")
        else:
            print(f"⚠️ Inference endpoint returned status {response.status_code}")
    except requests.exceptions.RequestException:
        print("❌ Inference endpoint is not available at http://localhost:8000")
        print("💡 Make sure the inference server is running first")
        return
    
    # Get image path from user or find test images
    image_path = input("Enter image path (or press Enter to use test images): ").strip()
    
    if not image_path:
        test_images = find_test_images()
        if not test_images:
            print("❌ No test images found")
            return
        
        print(f"📸 Found {len(test_images)} test images")
        for i, img_path in enumerate(test_images):
            print(f"{i+1}. {img_path}")
        
        choice = input("Select image number (1-{}): ".format(len(test_images))).strip()
        try:
            image_path = test_images[int(choice) - 1]
        except (ValueError, IndexError):
            image_path = test_images[0]
            print(f"Using first image: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return
    
    # Test field extraction
    success = test_field_extraction(image_path)
    
    if success:
        print("\n✅ Field extraction test passed!")
    else:
        print("\n❌ Field extraction test failed!")

if __name__ == "__main__":
    main() 
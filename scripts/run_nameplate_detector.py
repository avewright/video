#!/usr/bin/env python3
"""
Nameplate Detector Launcher
Helps you choose the best detection method for your system.
"""

import os
import sys
import subprocess

def check_model_exists():
    """Check if the trained model exists"""
    model_path = "best_nameplate_classifier.pth"
    if not os.path.exists(model_path):
        print("❌ Model file not found: best_nameplate_classifier.pth")
        print("💡 Please train the model first:")
        print("   python train_nameplate_classifier.py")
        return False
    return True

def test_opencv_gui():
    """Test if OpenCV GUI components work"""
    try:
        import cv2
        # Try to create a simple window
        test_frame = cv2.imread("test.jpg") if os.path.exists("test.jpg") else None
        if test_frame is None:
            import numpy as np
            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        cv2.imshow('OpenCV Test', test_frame)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        return True
    except Exception as e:
        print(f"OpenCV GUI test failed: {e}")
        return False

def main():
    print("🚀 Nameplate Detector Launcher")
    print("=" * 50)
    
    # Check if model exists
    if not check_model_exists():
        return
    
    print("✅ Model found!")
    print("\nChoose your detection method:\n")
    
    print("1. 🚀 QuickStart (OpenCV) - Simple and fast")
    print("   Best for: First try, simple usage")
    print("   Features: Auto-pause, basic overlay\n")
    
    print("2. 🔧 Advanced (OpenCV) - Full features")
    print("   Best for: Advanced users, customization")
    print("   Features: Adjustable settings, detailed stats\n")
    
    print("3. 🪟 Windows Compatible (matplotlib)")
    print("   Best for: Windows GUI issues, alternative display")
    print("   Features: Matplotlib display, threaded capture\n")
    
    print("4. 🧪 Test OpenCV GUI")
    print("   Check if OpenCV display works on your system\n")
    
    print("5. 📱 Single Image Test")
    print("   Test with a single image file\n")
    
    choice = input("Enter your choice (1-5): ").strip()
    
    if choice == "1":
        print("\n🚀 Starting QuickStart detector...")
        subprocess.run([sys.executable, "quickstart_camera.py"])
        
    elif choice == "2":
        print("\n🔧 Starting Advanced detector...")
        subprocess.run([sys.executable, "realtime_nameplate_detector.py"])
        
    elif choice == "3":
        print("\n🪟 Starting Windows-compatible detector...")
        subprocess.run([sys.executable, "camera_detector_windows.py"])
        
    elif choice == "4":
        print("\n🧪 Testing OpenCV GUI...")
        if test_opencv_gui():
            print("✅ OpenCV GUI works! You can use options 1 or 2.")
        else:
            print("❌ OpenCV GUI doesn't work. Use option 3 (Windows compatible).")
            print("💡 Or try: pip install opencv-python==4.8.1.78")
        
    elif choice == "5":
        print("\n📱 Testing with single image...")
        image_path = input("Enter image path (or press Enter for default test): ").strip()
        if not image_path:
            # Find a test image
            test_paths = [
                "data/nameplate_classification/test/has_nameplate/*.jpg",
                "data/samples/*.jpg",
                "*.jpg"
            ]
            import glob
            for pattern in test_paths:
                files = glob.glob(pattern)
                if files:
                    image_path = files[0]
                    break
        
        if image_path and os.path.exists(image_path):
            subprocess.run([sys.executable, "predict_nameplate.py", "--image", image_path])
        else:
            print("❌ No test image found. Please provide a valid image path.")
    
    else:
        print("❌ Invalid choice. Please run the script again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}") 
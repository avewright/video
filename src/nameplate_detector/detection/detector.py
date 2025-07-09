#!/usr/bin/env python3
"""
Real-time Nameplate Detection from Camera Stream
Streams from camera, detects nameplates, and pauses when one is found.
"""

import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import time
import argparse
import os
import requests
import json
import threading
import io

class LightweightNameplateClassifier(nn.Module):
    """Very lightweight CNN for binary nameplate classification"""
    
    def __init__(self, num_classes=2):
        super(LightweightNameplateClassifier, self).__init__()
        
        # Use pre-trained MobileNetV2 backbone (very lightweight)
        self.backbone = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')
        
        # Replace classifier for binary classification
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class RealtimeNameplateDetector:
    def __init__(self, model_path, device='auto', confidence_threshold=0.7):
        """
        Initialize the real-time nameplate detector
        
        Args:
            model_path: Path to the trained model
            device: 'cpu', 'cuda', or 'auto'
            confidence_threshold: Minimum confidence to trigger pause
        """
        self.confidence_threshold = confidence_threshold
        self.paused = False
        self.show_overlay = True
        self.frame_count = 0
        self.detection_count = 0
        self.field_extraction_in_progress = False
        self.extracted_fields = None
        self.extraction_error = None
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🔧 Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path)
        print(f"✅ Model loaded from: {model_path}")
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"🎯 Confidence threshold: {confidence_threshold}")
        print(f"📹 Initializing camera...")
    
    def _load_model(self, model_path):
        """Load the trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = LightweightNameplateClassifier(num_classes=2)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        model.to(self.device)
        return model
    
    def predict_frame(self, frame):
        """
        Predict nameplate presence in a frame
        
        Args:
            frame: OpenCV frame (BGR format)
            
        Returns:
            tuple: (has_nameplate, confidence)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)
        
        # Apply transforms
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted = torch.max(outputs, 1)[1].item()
            confidence = probabilities[0][predicted].item()
        
        has_nameplate = predicted == 1
        return has_nameplate, confidence
    
    def extract_fields_from_image(self, frame):
        """
        Extract fields from nameplate image using inference endpoint
        
        Args:
            frame: OpenCV frame containing the nameplate
            
        Returns:
            dict: Extracted fields as JSON or None if failed
        """
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
                return result
            else:
                print(f"❌ API request failed with status {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error during field extraction: {e}")
            return None
        except Exception as e:
            print(f"❌ Error during field extraction: {e}")
            return None

    def prompt_for_field_extraction(self, frame):
        """
        Prompt user for field extraction in a separate thread
        
        Args:
            frame: The current frame to extract fields from
        """
        def extraction_thread():
            self.field_extraction_in_progress = True
            self.extracted_fields = None
            self.extraction_error = None
            
            print("\n" + "="*50)
            print("🎯 NAMEPLATE DETECTED!")
            print("="*50)
            user_input = input("Would you like to extract fields from this nameplate? (y/n): ").strip().lower()
            
            if user_input in ['y', 'yes']:
                result = self.extract_fields_from_image(frame)
                if result:
                    self.extracted_fields = result
                    print("\n" + "="*50)
                    print("📋 EXTRACTED FIELDS:")
                    print("="*50)
                    print(json.dumps(result, indent=2))
                    print("="*50)
                else:
                    self.extraction_error = "Failed to extract fields"
                    print("❌ Failed to extract fields from nameplate")
            else:
                print("⏩ Skipping field extraction")
            
            self.field_extraction_in_progress = False
            
        thread = threading.Thread(target=extraction_thread)
        thread.daemon = True
        thread.start()

    def draw_overlay(self, frame, has_nameplate, confidence):
        """Draw prediction overlay on frame"""
        height, width = frame.shape[:2]
        
        # Prediction text
        if has_nameplate:
            label = "NAMEPLATE DETECTED"
            color = (0, 255, 0)  # Green
            if confidence >= self.confidence_threshold:
                status = "🔴 PAUSED"
                status_color = (0, 0, 255)  # Red
            else:
                status = "⚠️ LOW CONFIDENCE"
                status_color = (0, 165, 255)  # Orange
        else:
            label = "NO NAMEPLATE"
            color = (0, 0, 255)  # Red
            status = "🟢 SCANNING"
            status_color = (0, 255, 0)  # Green
        
        confidence_text = f"Confidence: {confidence*100:.1f}%"
        
        # Adjust overlay size based on extraction status
        overlay_height = 120
        if self.field_extraction_in_progress:
            overlay_height = 150
        elif self.extracted_fields:
            overlay_height = 180
        elif self.extraction_error:
            overlay_height = 150
        
        # Background rectangles for text
        cv2.rectangle(frame, (10, 10), (400, overlay_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, overlay_height), (255, 255, 255), 2)
        
        # Main prediction
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, confidence_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, status, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Field extraction status
        if self.field_extraction_in_progress:
            cv2.putText(frame, "🔍 Extracting fields...", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        elif self.extracted_fields:
            cv2.putText(frame, "✅ Fields extracted!", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, "Check console for details", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        elif self.extraction_error:
            cv2.putText(frame, "❌ Field extraction failed", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Frame counter and stats
        stats_text = f"Frame: {self.frame_count} | Detections: {self.detection_count}"
        cv2.putText(frame, stats_text, (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Controls
        controls = "Controls: SPACE=Pause/Resume | Q=Quit | O=Toggle Overlay | E=Extract Fields"
        cv2.putText(frame, controls, (20, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Threshold indicator
        threshold_text = f"Threshold: {self.confidence_threshold*100:.1f}%"
        cv2.putText(frame, threshold_text, (width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self, camera_id=0):
        """
        Start the real-time detection
        
        Args:
            camera_id: Camera device ID (usually 0 for default camera)
        """
        # Initialize camera
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera {camera_id}")
            print("💡 Try different camera IDs: 0, 1, 2...")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("🚀 Starting real-time nameplate detection...")
        print("📋 Controls:")
        print("   SPACE: Pause/Resume")
        print("   Q: Quit")
        print("   O: Toggle overlay")
        print("   E: Extract fields (when paused)")
        print("   +/-: Adjust confidence threshold")
        print("   R: Reset detection counter")
        
        try:
            frozen_frame = None
            prompted_for_extraction = False
            
            while True:
                if not self.paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("❌ Error reading from camera")
                        break
                    
                    self.frame_count += 1
                    
                    # Make prediction
                    has_nameplate, confidence = self.predict_frame(frame)
                    
                    # Check if we should pause
                    if has_nameplate and confidence >= self.confidence_threshold:
                        if not self.paused:
                            print(f"🎯 NAMEPLATE DETECTED! Confidence: {confidence*100:.1f}% - PAUSING")
                            self.detection_count += 1
                            self.paused = True
                            frozen_frame = frame.copy()
                            prompted_for_extraction = False
                    
                    # Draw overlay if enabled
                    if self.show_overlay:
                        frame = self.draw_overlay(frame, has_nameplate, confidence)
                    
                    current_frame = frame
                else:
                    # Show frozen frame when paused
                    current_frame = frozen_frame.copy() if frozen_frame is not None else frame
                    
                    # Prompt for field extraction once when paused
                    if not prompted_for_extraction and not self.field_extraction_in_progress:
                        self.prompt_for_field_extraction(frozen_frame)
                        prompted_for_extraction = True
                    
                    if self.show_overlay:
                        # Add pause indicator
                        cv2.rectangle(current_frame, (0, 0), (current_frame.shape[1], 50), (0, 0, 255), -1)
                        cv2.putText(current_frame, "⏸️ PAUSED - Press SPACE to resume or E to extract fields", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Update overlay with extraction status
                        current_frame = self.draw_overlay(current_frame, True, 1.0)
                
                # Display frame
                cv2.imshow('Real-time Nameplate Detector', current_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # Q or ESC to quit
                    break
                elif key == ord(' '):  # SPACE to pause/resume
                    self.paused = not self.paused
                    if not self.paused:
                        print("▶️ Resuming detection...")
                        # Reset extraction state
                        self.extracted_fields = None
                        self.extraction_error = None
                        prompted_for_extraction = False
                    else:
                        print("⏸️ Manually paused")
                elif key == ord('e') and self.paused:  # E to extract fields when paused
                    if frozen_frame is not None and not self.field_extraction_in_progress:
                        self.prompt_for_field_extraction(frozen_frame)
                elif key == ord('o'):  # O to toggle overlay
                    self.show_overlay = not self.show_overlay
                    print(f"📊 Overlay: {'ON' if self.show_overlay else 'OFF'}")
                elif key == ord('+') or key == ord('='):  # Increase threshold
                    self.confidence_threshold = min(1.0, self.confidence_threshold + 0.05)
                    print(f"🎯 Confidence threshold: {self.confidence_threshold*100:.1f}%")
                elif key == ord('-'):  # Decrease threshold
                    self.confidence_threshold = max(0.1, self.confidence_threshold - 0.05)
                    print(f"🎯 Confidence threshold: {self.confidence_threshold*100:.1f}%")
                elif key == ord('r'):  # Reset counter
                    self.detection_count = 0
                    self.frame_count = 0
                    print("🔄 Counters reset")
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            print("👋 Camera released and windows closed")
            print(f"📊 Final stats: {self.frame_count} frames processed, {self.detection_count} nameplates detected")

def main():
    parser = argparse.ArgumentParser(description='Real-time Nameplate Detection')
    parser.add_argument('--model', type=str, default='best_nameplate_classifier.pth',
                       help='Path to trained model')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Confidence threshold for detection (0.1-1.0)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use: cpu, cuda, or auto')
    
    args = parser.parse_args()
    
    # Validate threshold
    if not 0.1 <= args.threshold <= 1.0:
        print("❌ Threshold must be between 0.1 and 1.0")
        return
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        print("💡 Make sure you've trained the model first with train_nameplate_classifier.py")
        return
    
    # Create detector and run
    detector = RealtimeNameplateDetector(
        model_path=args.model,
        device=args.device,
        confidence_threshold=args.threshold
    )
    
    detector.run(camera_id=args.camera)

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Windows-Compatible Real-time Nameplate Detection
Uses matplotlib for display to avoid OpenCV GUI issues on Windows.
"""

import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import time
import os
import threading
import queue
import requests
import json
import io

class LightweightNameplateClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(LightweightNameplateClassifier, self).__init__()
        self.backbone = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class WindowsNameplateDetector:
    def __init__(self, model_path, confidence_threshold=0.8):
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.paused = False
        self.detection_count = 0
        self.frame_count = 0
        self.running = True
        self.field_extraction_in_progress = False
        self.extracted_fields = None
        self.extraction_error = None
        
        # Load model
        print(f"🧠 Loading model on {self.device}...")
        self.model = self._load_model(model_path)
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
        
        # Setup matplotlib
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_title('Real-time Nameplate Detector (Windows)', fontsize=16)
        self.ax.axis('off')
        
        # Create frame queue for thread communication
        self.frame_queue = queue.Queue(maxsize=2)
        
        print("✅ Detector initialized successfully!")
    
    def _load_model(self, model_path):
        model = LightweightNameplateClassifier(num_classes=2)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        model.to(self.device)
        return model
    
    def predict_frame(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
    
    def capture_frames(self):
        """Thread function to capture frames from camera"""
        while self.running:
            if not self.paused:
                ret, frame = self.cap.read()
                if ret:
                    # Put frame in queue (non-blocking)
                    try:
                        self.frame_queue.put(frame, block=False)
                    except queue.Full:
                        # Remove old frame if queue is full
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put(frame, block=False)
                        except queue.Empty:
                            pass
            time.sleep(0.03)  # ~30 FPS
    
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
        """Draw prediction overlay on frame using matplotlib"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Clear previous plot
        self.ax.clear()
        self.ax.imshow(rgb_frame)
        self.ax.axis('off')
        
        # Determine colors and text
        if has_nameplate:
            color = 'green'
            status = f"NAMEPLATE DETECTED! ({confidence*100:.1f}%)"
            if confidence >= self.confidence_threshold:
                border_color = 'red'
                status += " - PAUSED"
            else:
                border_color = 'orange'
                status += " - LOW CONFIDENCE"
        else:
            color = 'red'
            border_color = 'green'
            status = f"No nameplate detected ({confidence*100:.1f}%)"
        
        # Add title with detection status
        title = f"Frame: {self.frame_count} | Detections: {self.detection_count}"
        if self.paused:
            title += " | PAUSED"
        
        self.ax.set_title(title, fontsize=14, color=border_color)
        
        # Add text overlay
        text_box = dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7)
        self.ax.text(10, 30, status, fontsize=12, color=color, 
                    bbox=text_box, transform=self.ax.transData)
        
        # Add field extraction status
        if self.field_extraction_in_progress:
            self.ax.text(10, 60, "🔍 Extracting fields...", fontsize=10, color='yellow',
                        bbox=text_box, transform=self.ax.transData)
        elif self.extracted_fields:
            self.ax.text(10, 60, "✅ Fields extracted! Check console", fontsize=10, color='green',
                        bbox=text_box, transform=self.ax.transData)
        elif self.extraction_error:
            self.ax.text(10, 60, "❌ Field extraction failed", fontsize=10, color='red',
                        bbox=text_box, transform=self.ax.transData)
        
        # Add threshold info
        threshold_text = f"Threshold: {self.confidence_threshold*100:.1f}%"
        self.ax.text(10, rgb_frame.shape[0] - 20, threshold_text, 
                    fontsize=10, color='white', bbox=text_box, transform=self.ax.transData)
        
        # Add controls info
        controls = "Controls: Close window=Quit | Space=Pause | E=Extract Fields | +/-=Adjust threshold"
        self.ax.text(10, rgb_frame.shape[0] - 50, controls, 
                    fontsize=8, color='yellow', bbox=text_box, transform=self.ax.transData)
        
        # Add border for detection state
        if has_nameplate and confidence >= self.confidence_threshold:
            rect = patches.Rectangle((0, 0), rgb_frame.shape[1], rgb_frame.shape[0], 
                                   linewidth=5, edgecolor='red', facecolor='none')
            self.ax.add_patch(rect)
        
        plt.draw()
        plt.pause(0.001)
    
    def on_key_press(self, event):
        """Handle keyboard events"""
        if event.key == ' ':  # Space to pause/resume
            self.paused = not self.paused
            if not self.paused:
                # Reset extraction state
                self.extracted_fields = None
                self.extraction_error = None
            print(f"{'⏸️ Paused' if self.paused else '▶️ Resumed'}")
        elif event.key == 'e' and self.paused:  # E to extract fields when paused
            if hasattr(self, 'last_detection_frame') and not self.field_extraction_in_progress:
                self.prompt_for_field_extraction(self.last_detection_frame)
        elif event.key == '+' or event.key == '=':
            self.confidence_threshold = min(1.0, self.confidence_threshold + 0.05)
            print(f"🎯 Threshold: {self.confidence_threshold*100:.1f}%")
        elif event.key == '-':
            self.confidence_threshold = max(0.1, self.confidence_threshold - 0.05)
            print(f"🎯 Threshold: {self.confidence_threshold*100:.1f}%")
        elif event.key == 'r':
            self.detection_count = 0
            self.frame_count = 0
            print("🔄 Counters reset")
    
    def run(self):
        """Start the detection loop"""
        print("🚀 Starting Windows-compatible nameplate detection...")
        print("📋 Controls:")
        print("   - Close window: Quit")
        print("   - SPACE: Pause/Resume")
        print("   - E: Extract fields (when paused)")
        print("   - +/-: Adjust confidence threshold")
        print("   - R: Reset counters")
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Start camera capture thread
        capture_thread = threading.Thread(target=self.capture_frames)
        capture_thread.daemon = True
        capture_thread.start()
        
        last_frame = None
        prompted_for_extraction = False
        
        try:
            while self.running:
                try:
                    # Get latest frame from queue
                    frame = self.frame_queue.get_nowait()
                    last_frame = frame
                    
                    if not self.paused:
                        self.frame_count += 1
                        
                        # Make prediction
                        has_nameplate, confidence = self.predict_frame(frame)
                        
                        # Check if we should pause
                        if has_nameplate and confidence >= self.confidence_threshold:
                            if not self.paused:
                                print(f"🎯 NAMEPLATE DETECTED! #{self.detection_count + 1} - Confidence: {confidence*100:.1f}% - PAUSING")
                                self.detection_count += 1
                                self.paused = True
                                self.last_detection_frame = frame.copy()
                                prompted_for_extraction = False
                        
                        # Update display
                        self.draw_overlay(frame, has_nameplate, confidence)
                    else:
                        # Show paused state
                        if last_frame is not None:
                            # Prompt for field extraction once when paused
                            if not prompted_for_extraction and not self.field_extraction_in_progress:
                                self.prompt_for_field_extraction(self.last_detection_frame)
                                prompted_for_extraction = True
                            
                            self.draw_overlay(last_frame, True, 1.0)  # Show as detected when paused
                
                except queue.Empty:
                    # No new frame available, continue
                    time.sleep(0.01)
                    continue
                
                # Check if window is still open
                if not plt.get_fignums():
                    break
                
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        self.cap.release()
        plt.close('all')
        print("👋 Camera released and windows closed")
        print(f"📊 Final stats: {self.frame_count} frames processed, {self.detection_count} nameplates detected")

def main():
    model_path = "best_nameplate_classifier.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("💡 Please make sure you have trained the model first!")
        return
    
    try:
        detector = WindowsNameplateDetector(model_path, confidence_threshold=0.8)
        detector.run()
    except Exception as e:
        print(f"❌ Error initializing detector: {e}")
        print("💡 Make sure your camera is connected and not being used by another application")

if __name__ == "__main__":
    main() 
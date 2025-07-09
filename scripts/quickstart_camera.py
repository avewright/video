#!/usr/bin/env python3
"""
Quickstart Real-time Nameplate Detection
Simple script to run camera-based nameplate detection with minimal setup.
"""

import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
import os

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

def main():
    print("🚀 Starting Quickstart Nameplate Detector")
    print("="*50)
    
    # Check for model file
    model_path = "best_nameplate_classifier.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("💡 Please make sure you have trained the model first!")
        return
    
    # Load model
    print("🧠 Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LightweightNameplateClassifier(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    print(f"✅ Model loaded on {device}")
    
    # Setup image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize camera
    print("📹 Opening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open camera. Please check camera connection.")
        return
    
    print("🎯 Real-time detection started!")
    print("📋 Controls:")
    print("   - Press 'q' to quit")
    print("   - Press 'SPACE' when nameplate detected to pause")
    print("   - Detection will auto-pause when nameplate found")
    
    paused = False
    frozen_frame = None
    detection_count = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read from camera")
                break
            
            # Convert frame to RGB and predict
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            input_tensor = transform(pil_image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted = torch.max(outputs, 1)[1].item()
                confidence = probabilities[0][predicted].item()
            
            has_nameplate = predicted == 1
            
            # Auto-pause if nameplate detected with high confidence
            if has_nameplate and confidence > 0.8:
                paused = True
                frozen_frame = frame.copy()
                detection_count += 1
                print(f"🎯 NAMEPLATE DETECTED! #{detection_count} - Confidence: {confidence*100:.1f}% - PAUSED")
            
            # Draw simple overlay
            if has_nameplate:
                color = (0, 255, 0)  # Green
                text = f"NAMEPLATE DETECTED! ({confidence*100:.1f}%)"
            else:
                color = (0, 0, 255)  # Red  
                text = f"No nameplate ({confidence*100:.1f}%)"
            
            cv2.rectangle(frame, (10, 10), (600, 60), (0, 0, 0), -1)
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            current_frame = frame
        else:
            # Show frozen frame when paused
            current_frame = frozen_frame.copy()
            cv2.rectangle(current_frame, (0, 0), (current_frame.shape[1], 50), (0, 0, 255), -1)
            cv2.putText(current_frame, "PAUSED - Press SPACE to continue", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow('Nameplate Detector', current_frame)
        
        # Handle keypresses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            if not paused:
                print("▶️ Resuming...")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"📊 Total detections: {detection_count}")
    print("👋 Goodbye!")

if __name__ == "__main__":
    main() 
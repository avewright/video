#!/usr/bin/env python3
"""
Python service for nameplate detection integration with Node.js backend
"""

import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import base64
import json
import io
import sys
import os

# Add the parent directory to the Python path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

class NameplateDetectionService:
    def __init__(self, model_path=r"C:\Users\AWright\OneDrive - Kahua, Inc\Projects\video\models\best_nameplate_classifier.pth"):
        """Initialize the nameplate detection service"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use the absolute path provided
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        self.model = self._load_model(model_path)
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, model_path):
        """Load the trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = LightweightNameplateClassifier(num_classes=2)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        model.to(self.device)
        return model
    
    def detect_nameplate(self, image_data):
        """
        Detect nameplate in image data
        
        Args:
            image_data: Base64 encoded image data
            
        Returns:
            dict: Detection result with confidence
        """
        try:
            # Decode base64 image
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transforms
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted = torch.max(outputs, 1)[1].item()
                confidence = probabilities[0][predicted].item()
            
            has_nameplate = predicted == 1
            
            return {
                'detected': has_nameplate,
                'confidence': confidence,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'detected': False,
                'confidence': 0.0,
                'status': 'error',
                'error': str(e)
            }

def main():
    """Main function to handle stdin usage"""
    # Initialize service
    try:
        service = NameplateDetectionService()
        
        # Read image data from stdin
        image_data = sys.stdin.read().strip()
        
        if not image_data:
            raise ValueError("No image data provided via stdin")
        
        # Detect nameplate
        result = service.detect_nameplate(image_data)
        
        # Output result as JSON
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            'detected': False,
            'confidence': 0.0,
            'status': 'error',
            'error': str(e)
        }
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == "__main__":
    main() 
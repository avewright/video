#!/usr/bin/env python3
"""
Simple inference script for nameplate classification
Load the trained model and predict on new images
"""

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
import argparse
import os

class LightweightNameplateClassifier(nn.Module):
    """Very lightweight CNN for binary nameplate classification"""
    
    def __init__(self, num_classes=2):
        super(LightweightNameplateClassifier, self).__init__()
        
        # Use pre-trained MobileNetV2 backbone (very lightweight)
        self.backbone = torchvision.models.mobilenet_v2(pretrained=True)
        
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

def load_model(model_path='best_nameplate_classifier.pth'):
    """Load the trained model"""
    model = LightweightNameplateClassifier(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def predict_image(model, image_path, device='cpu'):
    """Predict if an image contains a nameplate"""
    
    # Load and preprocess image
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Transform (same as validation transform)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted = torch.max(outputs, 1)[1].item()
        confidence = probabilities[0][predicted].item()
    
    return predicted, confidence

def main():
    parser = argparse.ArgumentParser(description='Nameplate Classification Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='best_nameplate_classifier.pth', 
                       help='Path to trained model')
    parser.add_argument('--device', type=str, default='auto', 
                       help='Device to use: cpu, cuda, or auto')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"❌ Error: Image file '{args.image}' not found!")
        return
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file '{args.model}' not found!")
        print("   Run 'python train_nameplate_classifier.py' first to train the model.")
        return
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🔧 Using device: {device}")
    print(f"📸 Processing image: {args.image}")
    print(f"🧠 Using model: {args.model}")
    print("-" * 50)
    
    # Load model
    try:
        model = load_model(args.model)
        model.to(device)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Make prediction
    try:
        predicted, confidence = predict_image(model, args.image, device)
        
        result = "HAS NAMEPLATE" if predicted == 1 else "NO NAMEPLATE"
        confidence_percent = confidence * 100
        
        print(f"\n🎯 PREDICTION RESULT:")
        print(f"   Classification: {result}")
        print(f"   Confidence: {confidence_percent:.1f}%")
        
        # Add confidence interpretation
        if confidence_percent > 90:
            print(f"   Assessment: Very confident ✅")
        elif confidence_percent > 75:
            print(f"   Assessment: Confident 👍")
        elif confidence_percent > 60:
            print(f"   Assessment: Moderately confident 🤔")
        else:
            print(f"   Assessment: Low confidence ⚠️")
            
    except Exception as e:
        print(f"❌ Error during prediction: {e}")

if __name__ == "__main__":
    main() 
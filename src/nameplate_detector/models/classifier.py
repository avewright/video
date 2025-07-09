"""
Nameplate Classification Model

Contains the lightweight MobileNetV2-based classifier for detecting nameplates.
"""

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, Union
import os

class LightweightNameplateClassifier(nn.Module):
    """
    Very lightweight CNN for binary nameplate classification.
    
    Based on MobileNetV2 backbone for efficient inference.
    """
    
    def __init__(self, num_classes: int = 2):
        """
        Initialize the classifier.
        
        Args:
            num_classes: Number of classes (default: 2 for binary classification)
        """
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with class predictions
        """
        return self.backbone(x)

class NameplateClassifierLoader:
    """
    Utility class for loading and managing the nameplate classifier.
    """
    
    def __init__(self, model_path: str = None, device: str = 'auto'):
        """
        Initialize the classifier loader.
        
        Args:
            model_path: Path to the model file
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        self.model_path = model_path or self._find_model_path()
        self.device = self._get_device(device)
        self.model = None
        self.transform = self._get_transform()
    
    def _find_model_path(self) -> str:
        """Find the model file in common locations."""
        possible_paths = [
            "models/best_nameplate_classifier.pth",
            "best_nameplate_classifier.pth",
            "../models/best_nameplate_classifier.pth",
            "../../models/best_nameplate_classifier.pth"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError("Could not find model file. Please specify model_path.")
    
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device."""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _get_transform(self) -> transforms.Compose:
        """Get the image preprocessing transform."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self) -> LightweightNameplateClassifier:
        """
        Load the model from file.
        
        Returns:
            Loaded model instance
        """
        if self.model is None:
            self.model = LightweightNameplateClassifier(num_classes=2)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            self.model.to(self.device)
        
        return self.model
    
    def predict(self, image: Union[str, Image.Image, np.ndarray]) -> Tuple[bool, float]:
        """
        Make a prediction on an image.
        
        Args:
            image: Image to classify (path, PIL Image, or numpy array)
            
        Returns:
            Tuple of (has_nameplate, confidence)
        """
        if self.model is None:
            self.load_model()
        
        # Convert input to PIL Image
        if isinstance(image, str):
            pil_image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image).convert('RGB')
        elif isinstance(image, Image.Image):
            pil_image = image.convert('RGB')
        else:
            raise ValueError("Unsupported image type")
        
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
    
    def predict_batch(self, images: list) -> list:
        """
        Make predictions on a batch of images.
        
        Args:
            images: List of images to classify
            
        Returns:
            List of (has_nameplate, confidence) tuples
        """
        return [self.predict(image) for image in images] 
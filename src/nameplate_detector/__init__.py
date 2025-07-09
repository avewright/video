"""
Nameplate Detector Package

A real-time video streaming application that uses computer vision to detect 
and analyze industrial nameplates from live camera feeds.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main classes and functions for easy access
from .detection.detector import NameplateDetector
from .api.server import create_app
from .models.classifier import LightweightNameplateClassifier

__all__ = [
    "NameplateDetector",
    "create_app", 
    "LightweightNameplateClassifier"
] 
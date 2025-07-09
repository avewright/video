"""
Detection Package

Contains the core detection logic and camera handling components.
"""

from .detector import NameplateDetector
from .camera import CameraDetector
from .field_extractor import FieldExtractor

__all__ = ["NameplateDetector", "CameraDetector", "FieldExtractor"] 
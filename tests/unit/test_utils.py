"""
Unit tests for the utils module.
"""

import pytest
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock

from nameplate_detector.utils.helpers import (
    validate_image,
    format_confidence,
    resize_image,
    ensure_rgb,
    calculate_iou,
    safe_divide
)


class TestValidateImage:
    """Test the validate_image function."""
    
    def test_validate_valid_pil_image(self):
        """Test validation with a valid PIL image."""
        image = Image.new('RGB', (100, 100), color='red')
        assert validate_image(image) == True
    
    def test_validate_valid_numpy_array(self):
        """Test validation with a valid numpy array."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        assert validate_image(image) == True
    
    def test_validate_too_large_image(self):
        """Test validation with an image that's too large."""
        image = Image.new('RGB', (5000, 5000), color='red')
        assert validate_image(image, max_size=(1000, 1000)) == False
    
    def test_validate_too_small_image(self):
        """Test validation with an image that's too small."""
        image = Image.new('RGB', (10, 10), color='red')
        assert validate_image(image, min_size=(50, 50)) == False
    
    def test_validate_invalid_input(self):
        """Test validation with invalid input."""
        assert validate_image("nonexistent_file.jpg") == False
        assert validate_image(123) == False
        assert validate_image(None) == False


class TestFormatConfidence:
    """Test the format_confidence function."""
    
    def test_format_confidence_default_precision(self):
        """Test confidence formatting with default precision."""
        assert format_confidence(0.856) == "85.6%"
        assert format_confidence(0.1) == "10.0%"
        assert format_confidence(1.0) == "100.0%"
    
    def test_format_confidence_custom_precision(self):
        """Test confidence formatting with custom precision."""
        assert format_confidence(0.856789, precision=3) == "85.679%"
        assert format_confidence(0.1, precision=0) == "10%"


class TestResizeImage:
    """Test the resize_image function."""
    
    def test_resize_pil_image(self):
        """Test resizing a PIL image."""
        image = Image.new('RGB', (100, 100), color='red')
        resized = resize_image(image, (50, 50), maintain_aspect_ratio=False)
        assert resized.size == (50, 50)
    
    def test_resize_numpy_array(self):
        """Test resizing a numpy array."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        resized = resize_image(image, (50, 50), maintain_aspect_ratio=False)
        assert resized.shape[:2] == (50, 50)
    
    def test_resize_with_aspect_ratio(self):
        """Test resizing with aspect ratio maintenance."""
        image = Image.new('RGB', (100, 50), color='red')
        resized = resize_image(image, (60, 60), maintain_aspect_ratio=True)
        assert resized.size == (60, 60)


class TestEnsureRGB:
    """Test the ensure_rgb function."""
    
    def test_ensure_rgb_pil_image(self):
        """Test RGB conversion with PIL image."""
        image = Image.new('RGBA', (100, 100), color='red')
        rgb_image = ensure_rgb(image)
        assert rgb_image.mode == 'RGB'
    
    def test_ensure_rgb_numpy_array(self):
        """Test RGB conversion with numpy array."""
        # Create a BGR image (simulating OpenCV format)
        bgr_image = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr_image[:, :, 0] = 255  # Blue channel
        
        rgb_image = ensure_rgb(bgr_image)
        assert rgb_image.shape[2] == 3
        assert rgb_image[:, :, 2].max() > 0  # Red channel should have values


class TestCalculateIOU:
    """Test the calculate_iou function."""
    
    def test_calculate_iou_no_overlap(self):
        """Test IoU calculation with no overlap."""
        box1 = (0, 0, 10, 10)
        box2 = (20, 20, 30, 30)
        assert calculate_iou(box1, box2) == 0.0
    
    def test_calculate_iou_perfect_overlap(self):
        """Test IoU calculation with perfect overlap."""
        box1 = (0, 0, 10, 10)
        box2 = (0, 0, 10, 10)
        assert calculate_iou(box1, box2) == 1.0
    
    def test_calculate_iou_partial_overlap(self):
        """Test IoU calculation with partial overlap."""
        box1 = (0, 0, 10, 10)
        box2 = (5, 5, 15, 15)
        # Intersection: 5x5 = 25
        # Union: 100 + 100 - 25 = 175
        expected_iou = 25 / 175
        assert abs(calculate_iou(box1, box2) - expected_iou) < 1e-6


class TestSafeDivide:
    """Test the safe_divide function."""
    
    def test_safe_divide_normal(self):
        """Test safe division with normal values."""
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(7, 3) == 7/3
    
    def test_safe_divide_by_zero(self):
        """Test safe division by zero."""
        assert safe_divide(10, 0) == 0.0
        assert safe_divide(10, 0, default=999) == 999
    
    def test_safe_divide_negative(self):
        """Test safe division with negative values."""
        assert safe_divide(-10, 2) == -5.0
        assert safe_divide(10, -2) == -5.0 
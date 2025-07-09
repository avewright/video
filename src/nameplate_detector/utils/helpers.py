"""
Helper Utilities for Nameplate Detector

Contains utility functions for logging, image validation, and common operations.
"""

import logging
import os
import sys
from typing import Union, Optional, Tuple, Any
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file to write logs to
        format_string: Custom format string for log messages
        
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger

def validate_image(
    image: Union[str, Path, np.ndarray, Image.Image],
    max_size: Tuple[int, int] = (4096, 4096),
    min_size: Tuple[int, int] = (32, 32)
) -> bool:
    """
    Validate an image for processing.
    
    Args:
        image: Image to validate (path, numpy array, or PIL Image)
        max_size: Maximum allowed (width, height)
        min_size: Minimum allowed (width, height)
        
    Returns:
        True if image is valid, False otherwise
    """
    try:
        # Convert to PIL Image for validation
        if isinstance(image, (str, Path)):
            if not Path(image).exists():
                return False
            pil_image = Image.open(image)
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            return False
        
        # Check dimensions
        width, height = pil_image.size
        
        if width > max_size[0] or height > max_size[1]:
            return False
        
        if width < min_size[0] or height < min_size[1]:
            return False
        
        # Check if image can be converted to RGB
        pil_image.convert('RGB')
        
        return True
        
    except Exception:
        return False

def format_confidence(confidence: float, precision: int = 1) -> str:
    """
    Format confidence score as a percentage string.
    
    Args:
        confidence: Confidence score (0.0 to 1.0)
        precision: Number of decimal places
        
    Returns:
        Formatted confidence string (e.g., "85.3%")
    """
    return f"{confidence * 100:.{precision}f}%"

def resize_image(
    image: Union[np.ndarray, Image.Image],
    target_size: Tuple[int, int],
    maintain_aspect_ratio: bool = True
) -> Union[np.ndarray, Image.Image]:
    """
    Resize an image to target size.
    
    Args:
        image: Input image (numpy array or PIL Image)
        target_size: Target (width, height)
        maintain_aspect_ratio: Whether to maintain aspect ratio
        
    Returns:
        Resized image (same type as input)
    """
    if isinstance(image, np.ndarray):
        if maintain_aspect_ratio:
            h, w = image.shape[:2]
            aspect_ratio = w / h
            target_w, target_h = target_size
            
            if aspect_ratio > target_w / target_h:
                new_w = target_w
                new_h = int(target_w / aspect_ratio)
            else:
                new_h = target_h
                new_w = int(target_h * aspect_ratio)
            
            resized = cv2.resize(image, (new_w, new_h))
            
            # Add padding if needed
            if new_w != target_w or new_h != target_h:
                padded = np.zeros((target_h, target_w, 3), dtype=image.dtype)
                y_offset = (target_h - new_h) // 2
                x_offset = (target_w - new_w) // 2
                padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                return padded
            
            return resized
        else:
            return cv2.resize(image, target_size)
    
    elif isinstance(image, Image.Image):
        if maintain_aspect_ratio:
            image.thumbnail(target_size, Image.Resampling.LANCZOS)
            # Create new image with target size and paste the resized image
            new_image = Image.new('RGB', target_size, (0, 0, 0))
            x_offset = (target_size[0] - image.width) // 2
            y_offset = (target_size[1] - image.height) // 2
            new_image.paste(image, (x_offset, y_offset))
            return new_image
        else:
            return image.resize(target_size, Image.Resampling.LANCZOS)
    
    else:
        raise ValueError("Unsupported image type")

def ensure_rgb(image: Union[np.ndarray, Image.Image]) -> Union[np.ndarray, Image.Image]:
    """
    Ensure image is in RGB format.
    
    Args:
        image: Input image
        
    Returns:
        RGB image (same type as input)
    """
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3:
            if image.shape[2] == 4:  # RGBA
                return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 3:  # Might be BGR
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif len(image.shape) == 2:  # Grayscale
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image
    
    elif isinstance(image, Image.Image):
        return image.convert('RGB')
    
    else:
        raise ValueError("Unsupported image type")

def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1: First bounding box (x1, y1, x2, y2)
        box2: Second bounding box (x1, y1, x2, y2)
        
    Returns:
        IoU value (0.0 to 1.0)
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area

def get_device_info() -> dict:
    """
    Get information about available computing devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        "cpu_available": True,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_devices": []
    }
    
    if info["cuda_available"]:
        for i in range(info["cuda_device_count"]):
            device_info = {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total": torch.cuda.get_device_properties(i).total_memory,
                "memory_free": torch.cuda.memory_reserved(i),
                "compute_capability": torch.cuda.get_device_properties(i).major
            }
            info["cuda_devices"].append(device_info)
    
    return info

def create_directories(directories: list) -> None:
    """
    Create directories if they don't exist.
    
    Args:
        directories: List of directory paths to create
    """
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning a default value if division by zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value to return if denominator is zero
        
    Returns:
        Division result or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator

def measure_execution_time(func):
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to measure
        
    Returns:
        Decorated function
    """
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger = logging.getLogger(__name__)
        logger.debug(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        
        return result
    
    return wrapper 
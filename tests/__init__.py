"""
Tests Package for Nameplate Detector

Contains unit tests, integration tests, and test fixtures.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path for testing
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Test configuration
TEST_DATA_DIR = project_root / "tests" / "fixtures"
TEST_IMAGES_DIR = TEST_DATA_DIR / "images"
TEST_MODELS_DIR = TEST_DATA_DIR / "models"

# Create test directories if they don't exist
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_IMAGES_DIR.mkdir(exist_ok=True)
TEST_MODELS_DIR.mkdir(exist_ok=True) 
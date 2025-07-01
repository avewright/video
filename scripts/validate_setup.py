#!/usr/bin/env python3
"""
Setup validation script for Qwen2.5-VL Invoice OCR fine-tuning environment.
Checks dependencies, GPU availability, and model accessibility.
"""

import sys
import os
import torch
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible."""
    logger.info("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error(f"Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    
    logger.info(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_pytorch():
    """Check PyTorch installation and CUDA availability."""
    logger.info("Checking PyTorch installation...")
    
    try:
        import torch
        logger.info(f"✓ PyTorch {torch.__version__}")
        
        # Check CUDA
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1e9
            
            logger.info(f"✓ CUDA {cuda_version} available")
            logger.info(f"✓ {gpu_count} GPU(s) detected")
            logger.info(f"✓ Current GPU: {gpu_name}")
            logger.info(f"✓ GPU Memory: {gpu_memory:.1f} GB")
            
            if gpu_memory < 16:
                logger.warning("⚠️  GPU memory < 16GB. Consider using QLoRA or reducing batch size.")
            
            return True
        else:
            logger.warning("⚠️  CUDA not available. Training will be very slow on CPU.")
            return False
            
    except ImportError:
        logger.error("✗ PyTorch not installed")
        return False

def check_transformers():
    """Check transformers library."""
    logger.info("Checking transformers library...")
    
    try:
        import transformers
        logger.info(f"✓ Transformers {transformers.__version__}")
        
        # Check if version is recent enough
        version_parts = transformers.__version__.split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        
        if major < 4 or (major == 4 and minor < 37):
            logger.warning("⚠️  Transformers version may be too old. Recommend 4.37.0+")
        
        return True
        
    except ImportError:
        logger.error("✗ Transformers not installed")
        return False

def check_model_access():
    """Check if Qwen2.5-VL model can be accessed."""
    logger.info("Checking model access...")
    
    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        
        model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        logger.info(f"Testing access to {model_name}...")
        
        # Just try to load the config to test access
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        logger.info("✓ Model access confirmed")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Model access failed: {e}")
        logger.error("Make sure you have internet connection and HuggingFace access")
        return False

def check_additional_dependencies():
    """Check additional required dependencies."""
    logger.info("Checking additional dependencies...")
    
    required_packages = [
        'accelerate', 'peft', 'bitsandbytes', 'datasets', 
        'pillow', 'opencv-python', 'wandb', 'deepspeed'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
                logger.info(f"✓ OpenCV {cv2.__version__}")
            elif package == 'pillow':
                import PIL
                logger.info(f"✓ Pillow {PIL.__version__}")
            else:
                module = __import__(package)
                if hasattr(module, '__version__'):
                    logger.info(f"✓ {package} {module.__version__}")
                else:
                    logger.info(f"✓ {package}")
                    
        except ImportError:
            missing_packages.append(package)
            logger.error(f"✗ {package} not installed")
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.error("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_directory_structure():
    """Check if required directories exist."""
    logger.info("Checking directory structure...")
    
    required_dirs = [
        "configs", "data", "scripts", "src", 
        "data/raw", "data/processed", "data/samples"
    ]
    
    missing_dirs = []
    
    for directory in required_dirs:
        if not Path(directory).exists():
            missing_dirs.append(directory)
            logger.error(f"✗ Directory missing: {directory}")
        else:
            logger.info(f"✓ {directory}")
    
    if missing_dirs:
        logger.error("Run: python scripts/prepare_dataset.py to create missing directories")
        return False
    
    return True

def check_config_files():
    """Check if configuration files exist."""
    logger.info("Checking configuration files...")
    
    config_files = [
        "configs/qwen25_3b_qlora.yaml",
        "configs/deepspeed_zero2.json"
    ]
    
    missing_configs = []
    
    for config_file in config_files:
        if not Path(config_file).exists():
            missing_configs.append(config_file)
            logger.error(f"✗ Config missing: {config_file}")
        else:
            logger.info(f"✓ {config_file}")
    
    return len(missing_configs) == 0

def check_memory_requirements():
    """Check system memory."""
    logger.info("Checking system memory...")
    
    try:
        import psutil
        total_ram = psutil.virtual_memory().total / (1024**3)  # GB
        available_ram = psutil.virtual_memory().available / (1024**3)  # GB
        
        logger.info(f"✓ Total RAM: {total_ram:.1f} GB")
        logger.info(f"✓ Available RAM: {available_ram:.1f} GB")
        
        if total_ram < 16:
            logger.warning("⚠️  Total RAM < 16GB. May experience memory issues during training.")
        
        return True
        
    except ImportError:
        logger.warning("⚠️  psutil not installed. Cannot check memory.")
        return True

def main():
    parser = argparse.ArgumentParser(description="Validate Qwen2.5-VL training setup")
    parser.add_argument("--check_model", action="store_true",
                       help="Check model accessibility (requires internet)")
    
    args = parser.parse_args()
    
    logger.info("=== Qwen2.5-VL Training Environment Validation ===")
    
    checks = [
        ("Python Version", check_python_version),
        ("PyTorch & CUDA", check_pytorch),
        ("Transformers", check_transformers),
        ("Additional Dependencies", check_additional_dependencies),
        ("Directory Structure", check_directory_structure),
        ("Configuration Files", check_config_files),
        ("System Memory", check_memory_requirements),
    ]
    
    if args.check_model:
        checks.append(("Model Access", check_model_access))
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        logger.info(f"\n--- {check_name} ---")
        if check_func():
            passed += 1
        else:
            logger.error(f"Failed: {check_name}")
    
    logger.info(f"\n=== Validation Summary ===")
    logger.info(f"Passed: {passed}/{total} checks")
    
    if passed == total:
        logger.info("🎉 All checks passed! Environment is ready for training.")
        logger.info("\nNext steps:")
        logger.info("1. python scripts/prepare_dataset.py")
        logger.info("2. python train.py --config configs/qwen25_3b_qlora.yaml")
    else:
        logger.error("❌ Some checks failed. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
#!/usr/bin/env python3
"""
Dataset preparation script for Qwen2.5-VL Invoice OCR fine-tuning.
Downloads and processes the mychen76/invoices-and-receipts_ocr_v1 dataset.
"""

import os
import json
import argparse
from pathlib import Path
from datasets import load_dataset
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary data directories."""
    directories = ["data/raw", "data/processed", "data/samples"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def download_dataset(force_download=False):
    """Download the invoice OCR dataset."""
    logger.info("Loading dataset from HuggingFace...")
    
    try:
        dataset = load_dataset("mychen76/invoices-and-receipts_ocr_v1", 
                              cache_dir="data/raw")
        
        logger.info(f"Dataset loaded successfully!")
        logger.info(f"Train split: {len(dataset['train'])} examples")
        if 'validation' in dataset:
            logger.info(f"Validation split: {len(dataset['validation'])} examples")
        if 'test' in dataset:
            logger.info(f"Test split: {len(dataset['test'])} examples")
            
        return dataset
    
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return None

def save_sample_data(dataset, num_samples=5):
    """Save sample data for inspection."""
    logger.info(f"Saving {num_samples} sample examples...")
    
    samples_dir = Path("data/samples")
    samples_dir.mkdir(exist_ok=True)
    
    train_data = dataset['train']
    
    for i in range(min(num_samples, len(train_data))):
        example = train_data[i]
        
        # Save example metadata
        sample_info = {
            "index": i,
            "keys": list(example.keys()),
            "example_structure": {k: type(v).__name__ for k, v in example.items()}
        }
        
        with open(samples_dir / f"sample_{i}_info.json", "w") as f:
            json.dump(sample_info, f, indent=2)
        
        # Save first few examples as JSON for inspection
        if i < 3:
            # Convert any non-serializable objects to strings
            serializable_example = {}
            for k, v in example.items():
                if k == 'image' and hasattr(v, 'save'):
                    # Save image separately
                    image_path = samples_dir / f"sample_{i}_image.jpg"
                    v.save(image_path)
                    serializable_example[k] = str(image_path)
                else:
                    try:
                        json.dumps(v)  # Test if serializable
                        serializable_example[k] = v
                    except:
                        serializable_example[k] = str(v)
            
            with open(samples_dir / f"sample_{i}_data.json", "w") as f:
                json.dump(serializable_example, f, indent=2)
    
    logger.info(f"Sample data saved to {samples_dir}")

def validate_dataset(dataset):
    """Validate dataset structure and content."""
    logger.info("Validating dataset structure...")
    
    required_keys = ['image', 'text']  # Adjust based on actual dataset structure
    train_data = dataset['train']
    
    if len(train_data) == 0:
        logger.error("Dataset is empty!")
        return False
    
    # Check first example
    first_example = train_data[0]
    logger.info(f"First example keys: {list(first_example.keys())}")
    
    # Basic validation
    total_examples = len(train_data)
    valid_examples = 0
    
    for i, example in enumerate(train_data):
        if i >= 100:  # Only check first 100 for efficiency
            break
            
        try:
            # Check if image exists and is valid
            if 'image' in example and example['image'] is not None:
                if hasattr(example['image'], 'size'):  # PIL Image
                    valid_examples += 1
                elif isinstance(example['image'], str) and os.path.exists(example['image']):
                    valid_examples += 1
                    
        except Exception as e:
            logger.warning(f"Invalid example at index {i}: {e}")
    
    logger.info(f"Validated {min(100, total_examples)} examples")
    logger.info(f"Valid examples: {valid_examples}")
    
    return valid_examples > 0

def create_data_info():
    """Create dataset information file."""
    info = {
        "dataset_name": "mychen76/invoices-and-receipts_ocr_v1",
        "task": "OCR text to structured JSON conversion",
        "description": "Invoice and receipt processing for structured data extraction",
        "model_target": "Qwen2.5-VL-3B-Instruct",
        "data_format": {
            "input": "Image + OCR text with bounding boxes",
            "output": "Structured JSON with extracted information"
        },
        "directories": {
            "raw": "data/raw - Original dataset cache",
            "processed": "data/processed - Preprocessed data",
            "samples": "data/samples - Sample examples for inspection"
        }
    }
    
    with open("data/dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    logger.info("Dataset info saved to data/dataset_info.json")

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for Qwen2.5-VL training")
    parser.add_argument("--force_download", action="store_true", 
                       help="Force re-download of dataset")
    parser.add_argument("--samples", type=int, default=5,
                       help="Number of sample examples to save")
    
    args = parser.parse_args()
    
    logger.info("Starting dataset preparation...")
    
    # Setup directories
    setup_directories()
    
    # Download dataset
    dataset = download_dataset(args.force_download)
    if dataset is None:
        logger.error("Failed to download dataset. Exiting.")
        return
    
    # Validate dataset
    if not validate_dataset(dataset):
        logger.error("Dataset validation failed. Exiting.")
        return
    
    # Save samples
    save_sample_data(dataset, args.samples)
    
    # Create info file
    create_data_info()
    
    logger.info("Dataset preparation completed successfully!")
    logger.info("Next steps:")
    logger.info("1. Inspect samples in data/samples/")
    logger.info("2. Run: python scripts/validate_setup.py")
    logger.info("3. Start training: python train.py --config configs/qwen25_3b_qlora.yaml")

if __name__ == "__main__":
    main() 
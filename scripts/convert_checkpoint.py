#!/usr/bin/env python3
"""
Convert training checkpoints to deployment-ready format.
Useful for converting LoRA adapters to full models.
"""

import os
import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_lora_to_full_model(
    base_model_path: str,
    lora_adapter_path: str,
    output_path: str,
    push_to_hub: bool = False,
    hub_model_id: str = None
):
    """
    Convert LoRA adapter + base model to a full merged model.
    
    Args:
        base_model_path: Path to base model (e.g., "Qwen/Qwen2.5-VL-3B-Instruct")
        lora_adapter_path: Path to LoRA adapter weights
        output_path: Path to save the merged model
        push_to_hub: Whether to push to HuggingFace Hub
        hub_model_id: HuggingFace Hub model ID for pushing
    """
    logger.info(f"Loading base model from {base_model_path}")
    
    # Load base model
    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    logger.info(f"Loading LoRA adapter from {lora_adapter_path}")
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    
    logger.info("Merging LoRA adapter with base model...")
    
    # Merge and unload adapter
    merged_model = model.merge_and_unload()
    
    logger.info(f"Saving merged model to {output_path}")
    
    # Save merged model
    merged_model.save_pretrained(output_path)
    
    # Also save the processor
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    processor.save_pretrained(output_path)
    
    logger.info("Model conversion completed successfully!")
    
    if push_to_hub and hub_model_id:
        logger.info(f"Pushing to HuggingFace Hub: {hub_model_id}")
        merged_model.push_to_hub(hub_model_id)
        processor.push_to_hub(hub_model_id)
        logger.info("Model pushed to Hub successfully!")

def main():
    parser = argparse.ArgumentParser(description="Convert model checkpoints")
    parser.add_argument("--base_model", type=str, required=True,
                       help="Base model path or HuggingFace model ID")
    parser.add_argument("--adapter_path", type=str, required=True,
                       help="Path to LoRA adapter")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Output path for merged model")
    parser.add_argument("--push_to_hub", action="store_true",
                       help="Push to HuggingFace Hub")
    parser.add_argument("--hub_model_id", type=str,
                       help="HuggingFace Hub model ID")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.adapter_path).exists():
        logger.error(f"Adapter path does not exist: {args.adapter_path}")
        return
    
    if args.push_to_hub and not args.hub_model_id:
        logger.error("--hub_model_id is required when --push_to_hub is specified")
        return
    
    # Create output directory
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    
    # Convert model
    convert_lora_to_full_model(
        base_model_path=args.base_model,
        lora_adapter_path=args.adapter_path,
        output_path=args.output_path,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id
    )

if __name__ == "__main__":
    main() 
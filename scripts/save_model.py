#!/usr/bin/env python3
"""
Script to save/convert the trained QLoRA model to a full model.
This merges the LoRA adapters with the base model for easier deployment.
"""

import argparse
import os
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
import json


def save_model(
    base_model_name: str,
    adapter_path: str,
    output_path: str,
    push_to_hub: bool = False,
    hf_repo_name: str = None,
    hf_token: str = None
):
    """
    Save the trained QLoRA model by merging adapters with base model.
    
    Args:
        base_model_name: Original base model name (e.g., "Qwen/Qwen2.5-VL-3B-Instruct")
        adapter_path: Path to the LoRA adapter checkpoint
        output_path: Local path to save the merged model
        push_to_hub: Whether to upload to HuggingFace Hub
        hf_repo_name: HuggingFace repository name (e.g., "username/model-name")
        hf_token: HuggingFace token for authentication
    """
    
    print(f"🚀 Loading base model: {base_model_name}")
    
    # Load base model
    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    
    print(f"📦 Loading LoRA adapter from: {adapter_path}")
    
    # Load and merge LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = model.merge_and_unload()
    
    # Load processor
    processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
    
    print(f"💾 Saving merged model to: {output_path}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save merged model and processor
    merged_model.save_pretrained(output_path, safe_serialization=True)
    processor.save_pretrained(output_path)
    
    # Create model card
    model_card = f"""---
license: apache-2.0
base_model: {base_model_name}
tags:
- vision
- multimodal
- invoice
- ocr
- receipt
- qwen2.5-vl
- fine-tuned
language:
- en
pipeline_tag: image-text-to-text
---

# Qwen2.5-VL Invoice OCR Fine-tuned Model

This model is a fine-tuned version of [{base_model_name}]({base_model_name}) for invoice and receipt data extraction.

## Model Details

- **Base Model**: {base_model_name}
- **Fine-tuning Method**: QLoRA (4-bit quantization + LoRA)
- **Dataset**: Invoice and receipt images
- **Task**: Vision-to-JSON structured data extraction

## Usage

```python
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch

# Load model and processor
model = AutoModelForVision2Seq.from_pretrained(
    "{hf_repo_name or 'your-username/model-name'}",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained("{hf_repo_name or 'your-username/model-name'}")

# Load image
image = Image.open("invoice.jpg")

# Prepare prompt
prompt = "Analyze the image and return in JSON format all metadata seen including company details, items, prices, totals, and dates."

# Process inputs
inputs = processor(text=prompt, images=image, return_tensors="pt")

# Generate
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    
# Decode response
response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print(response)
```

## Training Details

- **Training Data**: Invoice and receipt images with structured JSON outputs
- **Training Method**: QLoRA fine-tuning
- **Hardware**: NVIDIA A100 80GB
- **Framework**: Transformers + PEFT

## Intended Use

This model is designed to extract structured information from invoices and receipts, including:
- Company details
- Item descriptions and prices
- Totals and tax amounts
- Dates and addresses

## Limitations

- Optimized for English text
- Performance may vary on handwritten or low-quality images
- Designed specifically for invoice/receipt formats
"""

    with open(os.path.join(output_path, "README.md"), "w") as f:
        f.write(model_card)
    
    # Save training config info
    config_info = {
        "base_model": base_model_name,
        "adapter_path": adapter_path,
        "training_method": "QLoRA",
        "task": "invoice-ocr",
        "merged_at": str(torch.utils.data.get_worker_info())
    }
    
    with open(os.path.join(output_path, "training_info.json"), "w") as f:
        json.dump(config_info, f, indent=2)
    
    print(f"✅ Model saved successfully to: {output_path}")
    
    if push_to_hub and hf_repo_name:
        print(f"🚀 Uploading to HuggingFace Hub: {hf_repo_name}")
        
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)
        
        merged_model.push_to_hub(
            hf_repo_name,
            use_temp_dir=False,
            commit_message="Upload fine-tuned Qwen2.5-VL invoice OCR model"
        )
        processor.push_to_hub(
            hf_repo_name,
            use_temp_dir=False,
            commit_message="Upload processor for Qwen2.5-VL invoice OCR model"
        )
        
        print(f"✅ Model uploaded successfully to: https://huggingface.co/{hf_repo_name}")
    
    return output_path


def find_best_checkpoint(output_dir: str) -> str:
    """Find the best checkpoint from training output directory."""
    checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    
    if not checkpoint_dirs:
        raise ValueError(f"No checkpoints found in {output_dir}")
    
    # Sort by checkpoint number
    checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
    
    # Return the latest checkpoint (or you could add logic to find best based on eval loss)
    best_checkpoint = os.path.join(output_dir, checkpoint_dirs[-1])
    print(f"📍 Using checkpoint: {best_checkpoint}")
    return best_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Save trained QLoRA model")
    parser.add_argument(
        "--base_model", 
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Base model name"
    )
    parser.add_argument(
        "--adapter_path",
        help="Path to LoRA adapter checkpoint (if not provided, will find latest)"
    )
    parser.add_argument(
        "--training_output_dir",
        default="./outputs/qwen25-3b-qlora-invoice",
        help="Training output directory to find checkpoints"
    )
    parser.add_argument(
        "--output_path",
        default="./saved_models/qwen25-vl-invoice-ocr",
        help="Path to save the merged model"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Upload to HuggingFace Hub"
    )
    parser.add_argument(
        "--hf_repo_name",
        help="HuggingFace repository name (e.g., username/model-name)"
    )
    parser.add_argument(
        "--hf_token",
        help="HuggingFace token (or set HF_TOKEN env var)"
    )
    
    args = parser.parse_args()
    
    # Find adapter path if not provided
    if not args.adapter_path:
        args.adapter_path = find_best_checkpoint(args.training_output_dir)
    
    # Get HF token from env if not provided
    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    
    # Save model
    save_model(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        output_path=args.output_path,
        push_to_hub=args.push_to_hub,
        hf_repo_name=args.hf_repo_name,
        hf_token=hf_token
    )


if __name__ == "__main__":
    main() 
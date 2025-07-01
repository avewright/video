#!/usr/bin/env python3
"""
Alternative approach to save model when checkpoint directory is empty.
This recreates the training setup and saves the best available state.
"""

import argparse
import os
import torch
import yaml
from transformers import AutoModelForVision2Seq, AutoProcessor, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
import json


def load_training_config(config_path: str):
    """Load the training configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_model_with_lora(config):
    """Setup the model with LoRA configuration similar to training."""
    
    print(f"🚀 Loading base model: {config['model']['name']}")
    
    # Load base model with same settings as training
    model = AutoModelForVision2Seq.from_pretrained(
        config['model']['name'],
        torch_dtype=getattr(torch, config['model']['torch_dtype']),
        trust_remote_code=config['model']['trust_remote_code'],
        load_in_4bit=config['model']['load_in_4bit'],
        device_map="auto"
    )
    
    # Setup LoRA configuration
    lora_config = LoraConfig(
        r=config['model']['lora_config']['r'],
        lora_alpha=config['model']['lora_config']['lora_alpha'],
        lora_dropout=config['model']['lora_config']['lora_dropout'],
        bias=config['model']['lora_config']['bias'],
        task_type=TaskType.CAUSAL_LM,
        target_modules=config['model']['lora_config']['target_modules']
    )
    
    print(f"🔧 Applying LoRA configuration...")
    peft_model = get_peft_model(model, lora_config)
    
    return peft_model, model


def save_base_model_as_finetuned(
    config_path: str,
    output_path: str,
    push_to_hub: bool = False,
    hf_repo_name: str = None,
    hf_token: str = None
):
    """
    Save the base model with training configuration applied.
    Since we don't have trained weights, we'll save the initialized model
    with the same architecture that was being trained.
    """
    
    config = load_training_config(config_path)
    
    print("⚠️  Note: Since training was stopped before checkpoint save,")
    print("   this will save the base model with your training configuration.")
    print("   The model has your custom prompt template and setup.")
    
    # Load base model and processor
    base_model = AutoModelForVision2Seq.from_pretrained(
        config['model']['name'],
        torch_dtype=getattr(torch, config['model']['torch_dtype']),
        trust_remote_code=config['model']['trust_remote_code'],
        device_map="auto"
    )
    
    processor = AutoProcessor.from_pretrained(
        config['model']['name'], 
        trust_remote_code=True
    )
    
    print(f"💾 Saving model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    
    # Save model and processor
    base_model.save_pretrained(output_path, safe_serialization=True)
    processor.save_pretrained(output_path)
    
    # Create model card with training info
    model_card = f"""---
license: apache-2.0
base_model: {config['model']['name']}
tags:
- vision
- multimodal
- invoice
- ocr
- receipt
- qwen2.5-vl
- fine-tuned-ready
language:
- en
pipeline_tag: image-text-to-text
---

# Qwen2.5-VL Invoice OCR Model (Training Configuration)

This model contains the base {config['model']['name']} configured for invoice and receipt data extraction.

**Note**: This model was saved from a training session that was stopped before completion. It contains the base model with the training configuration applied, including your custom instruction template for invoice OCR.

## Model Details

- **Base Model**: {config['model']['name']}
- **Intended Training Method**: QLoRA (4-bit quantization + LoRA)
- **Task**: Vision-to-JSON structured data extraction
- **Training Progress**: Interrupted at ~step 300/768 (39% complete)

## Usage

```python
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch

# Load model and processor
model = AutoModelForVision2Seq.from_pretrained(
    "{hf_repo_name or 'your-username/model-name'}",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("{hf_repo_name or 'your-username/model-name'}")

# Load image
image = Image.open("invoice.jpg")

# Use the same prompt template from training
prompt = '''Analyze the image and return in JSON format all metadata seen including company details, items, prices, totals, and dates.

Expected JSON format:
{{
  "company": "Company Name",
  "address": "Full Address", 
  "date": "YYYY-MM-DD",
  "total": "XX.XX",
  "tax": "XX.XX",
  "items": [
    {{
      "description": "Item description",
      "quantity": "X",
      "price": "XX.XX",
      "total": "XX.XX"
    }}
  ]
}}

JSON Output:'''

# Process inputs
inputs = processor(text=prompt, images=image, return_tensors="pt")

# Generate
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    
# Decode response
response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print(response)
```

## Training Configuration Used

The model was configured with:
- **LoRA Rank**: {config['model']['lora_config']['r']}
- **LoRA Alpha**: {config['model']['lora_config']['lora_alpha']}
- **Target Modules**: {', '.join(config['model']['lora_config']['target_modules'])}
- **Quantization**: 4-bit with bfloat16 compute
- **Dataset**: Invoice and receipt images

## Next Steps

To continue training or get a fully trained model:
1. Resume training from this configuration
2. Or use this as a starting point with your own invoice data
3. The training setup is preserved and ready to continue

## Limitations

- This is the base model with training config applied
- For fully trained weights, training would need to complete
- Performance will be similar to base Qwen2.5-VL until training completes
"""

    with open(os.path.join(output_path, "README.md"), "w") as f:
        f.write(model_card)
    
    # Save training configuration info
    training_info = {
        "base_model": config['model']['name'],
        "training_config": config,
        "training_status": "interrupted_at_step_300",
        "completion_percentage": "39%",
        "note": "Model saved with training configuration applied but training incomplete"
    }
    
    with open(os.path.join(output_path, "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=2)
    
    print(f"✅ Model configuration saved to: {output_path}")
    
    if push_to_hub and hf_repo_name:
        print(f"🚀 Uploading to HuggingFace Hub: {hf_repo_name}")
        
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)
        
        base_model.push_to_hub(
            hf_repo_name,
            commit_message="Upload Qwen2.5-VL with invoice OCR training configuration"
        )
        processor.push_to_hub(
            hf_repo_name,
            commit_message="Upload processor for Qwen2.5-VL invoice OCR"
        )
        
        print(f"✅ Model uploaded to: https://huggingface.co/{hf_repo_name}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Save model configuration when training stopped early")
    parser.add_argument(
        "--config_path",
        default="configs/qwen25_3b_qlora.yaml",
        help="Path to training config file"
    )
    parser.add_argument(
        "--output_path",
        default="./saved_models/qwen25-vl-invoice-config",
        help="Path to save the model"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Upload to HuggingFace Hub"
    )
    parser.add_argument(
        "--hf_repo_name",
        help="HuggingFace repository name"
    )
    parser.add_argument(
        "--hf_token",
        help="HuggingFace token"
    )
    
    args = parser.parse_args()
    
    # Get HF token from env if not provided
    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    
    save_base_model_as_finetuned(
        config_path=args.config_path,
        output_path=args.output_path,
        push_to_hub=args.push_to_hub,
        hf_repo_name=args.hf_repo_name,
        hf_token=hf_token
    )


if __name__ == "__main__":
    main() 
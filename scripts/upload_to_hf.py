#!/usr/bin/env python3
"""
Script to upload a saved model to HuggingFace Hub.
"""

import argparse
import os
import json
from pathlib import Path
from huggingface_hub import HfApi, login, create_repo
from transformers import AutoModelForVision2Seq, AutoProcessor


def upload_to_hf(
    model_path: str,
    repo_name: str,
    hf_token: str = None,
    private: bool = False,
    commit_message: str = None
):
    """
    Upload a model to HuggingFace Hub.
    
    Args:
        model_path: Local path to the saved model
        repo_name: HuggingFace repository name (e.g., "username/model-name")
        hf_token: HuggingFace token for authentication
        private: Whether to create a private repository
        commit_message: Custom commit message
    """
    
    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")
    
    # Get token
    token = hf_token or os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HuggingFace token required. Set HF_TOKEN env var or pass --hf_token")
    
    print(f"🔐 Logging into HuggingFace Hub...")
    login(token=token)
    
    # Initialize API
    api = HfApi()
    
    print(f"📦 Creating repository: {repo_name}")
    try:
        create_repo(
            repo_id=repo_name,
            token=token,
            private=private,
            exist_ok=True
        )
        print(f"✅ Repository created/verified: https://huggingface.co/{repo_name}")
    except Exception as e:
        print(f"⚠️  Repository creation warning: {e}")
    
    # Load and upload model
    print(f"🚀 Loading model from: {model_path}")
    
    try:
        # Load model
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Load processor
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Upload model
        print(f"⬆️  Uploading model to {repo_name}...")
        model.push_to_hub(
            repo_name,
            token=token,
            commit_message=commit_message or "Upload fine-tuned Qwen2.5-VL invoice OCR model",
            private=private
        )
        
        # Upload processor
        print(f"⬆️  Uploading processor to {repo_name}...")
        processor.push_to_hub(
            repo_name,
            token=token,
            commit_message=commit_message or "Upload processor for Qwen2.5-VL invoice OCR model",
            private=private
        )
        
        # Upload additional files (README, training_info, etc.)
        additional_files = [
            "README.md",
            "training_info.json",
            "config.json"
        ]
        
        for file_name in additional_files:
            file_path = os.path.join(model_path, file_name)
            if os.path.exists(file_path):
                print(f"⬆️  Uploading {file_name}...")
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=file_name,
                    repo_id=repo_name,
                    token=token,
                    commit_message=f"Upload {file_name}"
                )
        
        print(f"✅ Model uploaded successfully!")
        print(f"🌐 Model URL: https://huggingface.co/{repo_name}")
        print(f"📚 Usage documentation: https://huggingface.co/{repo_name}#usage")
        
        return f"https://huggingface.co/{repo_name}"
        
    except Exception as e:
        print(f"❌ Error uploading model: {e}")
        raise


def create_model_card(model_path: str, repo_name: str, base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
    """Create or update model card with detailed information."""
    
    model_card_content = f"""---
license: apache-2.0
base_model: {base_model}
tags:
- vision
- multimodal
- invoice
- ocr
- receipt
- qwen2.5-vl
- fine-tuned
- peft
- qlora
language:
- en
pipeline_tag: image-text-to-text
library_name: transformers
---

# 🧾 Qwen2.5-VL Invoice OCR Model

This model is a fine-tuned version of [{base_model}](https://huggingface.co/{base_model}) specifically optimized for **invoice and receipt data extraction**.

## 🎯 Model Details

- **Base Model**: {base_model}
- **Fine-tuning Method**: QLoRA (4-bit quantization + LoRA)
- **Training Data**: Invoice and receipt images with structured JSON outputs
- **Task**: Vision-to-JSON structured data extraction
- **Language**: English
- **License**: Apache 2.0

## 🚀 Quick Start

```python
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch

# Load model and processor
model = AutoModelForVision2Seq.from_pretrained(
    "{repo_name}",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("{repo_name}", trust_remote_code=True)

# Load your invoice/receipt image
image = Image.open("path/to/your/invoice.jpg")

# Prepare the prompt
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

# Generate response
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        temperature=0.1
    )

# Decode the response
response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print(response)
```

## 📊 Training Details

- **Training Method**: QLoRA (Quantized Low-Rank Adaptation)
- **Base Model**: Qwen2.5-VL-3B-Instruct
- **Dataset**: Invoice and receipt images with structured outputs
- **Hardware**: NVIDIA A100 80GB
- **Framework**: 🤗 Transformers + PEFT
- **Quantization**: 4-bit with bfloat16 compute
- **LoRA Rank**: 8
- **Training Epochs**: 3
- **Batch Size**: 8 (with gradient accumulation)

## 🎯 Capabilities

This model can extract structured information from invoices and receipts including:

- ✅ **Company Information**: Name, address, contact details
- ✅ **Transaction Details**: Date, invoice number, payment terms
- ✅ **Line Items**: Product descriptions, quantities, unit prices
- ✅ **Financial Data**: Subtotals, tax amounts, total amounts
- ✅ **Additional Metadata**: Currency, tax rates, discounts

## 📈 Performance

The model has been optimized for:
- **Accuracy**: High precision in data extraction
- **Speed**: Efficient inference on consumer GPUs
- **Robustness**: Handles various invoice formats and layouts
- **JSON Output**: Structured, parseable responses

## ⚠️ Limitations

- Optimized primarily for **English text**
- Best performance on **digital/printed invoices** (handwritten may vary)
- Designed specifically for **invoice/receipt formats**
- May require fine-tuning for highly specialized document types

## 🛠️ Technical Specifications

- **Model Size**: ~3B parameters
- **Input Resolution**: Up to 1280px
- **Max Sequence Length**: 2048 tokens
- **Precision**: bfloat16
- **Memory Requirements**: ~6GB VRAM for inference

## 📝 Citation

If you use this model in your research or applications, please cite:

```bibtex
@misc{{qwen25-vl-invoice-ocr,
  title={{Qwen2.5-VL Invoice OCR Fine-tuned Model}},
  author={{Your Name}},
  year={{2024}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/{repo_name}}}
}}
```

## 🤝 Contributing

Feel free to report issues, suggest improvements, or contribute to the model's development!

## 📄 License

This model is released under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
"""

    readme_path = os.path.join(model_path, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(model_card_content)
    
    print(f"📝 Updated model card: {readme_path}")


def main():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the saved model directory"
    )
    parser.add_argument(
        "--repo_name",
        required=True,
        help="HuggingFace repository name (e.g., 'username/model-name')"
    )
    parser.add_argument(
        "--hf_token",
        help="HuggingFace token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repository"
    )
    parser.add_argument(
        "--commit_message",
        help="Custom commit message"
    )
    parser.add_argument(
        "--update_model_card",
        action="store_true",
        help="Update model card before uploading"
    )
    parser.add_argument(
        "--base_model",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Base model name for model card"
    )
    
    args = parser.parse_args()
    
    # Update model card if requested
    if args.update_model_card:
        print("📝 Updating model card...")
        create_model_card(args.model_path, args.repo_name, args.base_model)
    
    # Upload to HF Hub
    upload_to_hf(
        model_path=args.model_path,
        repo_name=args.repo_name,
        hf_token=args.hf_token,
        private=args.private,
        commit_message=args.commit_message
    )


if __name__ == "__main__":
    main() 
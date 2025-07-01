---
license: apache-2.0
base_model: Qwen/Qwen2.5-VL-3B-Instruct
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

This model is a fine-tuned version of [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) specifically optimized for **invoice and receipt data extraction**.

## 🎯 Model Details

- **Base Model**: Qwen/Qwen2.5-VL-3B-Instruct
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
    "kahua-ml/invoice1",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("kahua-ml/invoice1", trust_remote_code=True)

# Load your invoice/receipt image
image = Image.open("path/to/your/invoice.jpg")

# Prepare the prompt
prompt = '''Analyze the image and return in JSON format all metadata seen including company details, items, prices, totals, and dates.

Expected JSON format:
{
  "company": "Company Name",
  "address": "Full Address", 
  "date": "YYYY-MM-DD",
  "total": "XX.XX",
  "tax": "XX.XX",
  "items": [
    {
      "description": "Item description",
      "quantity": "X",
      "price": "XX.XX",
      "total": "XX.XX"
    }
  ]
}

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
@misc{qwen25-vl-invoice-ocr,
  title={Qwen2.5-VL Invoice OCR Fine-tuned Model},
  author={Your Name},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/kahua-ml/invoice1}
}
```

## 🤝 Contributing

Feel free to report issues, suggest improvements, or contribute to the model's development!

## 📄 License

This model is released under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

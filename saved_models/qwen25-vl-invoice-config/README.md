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
- fine-tuned-ready
language:
- en
pipeline_tag: image-text-to-text
---

# Qwen2.5-VL Invoice OCR Model (Training Configuration)

This model contains the base Qwen/Qwen2.5-VL-3B-Instruct configured for invoice and receipt data extraction.

**Note**: This model was saved from a training session that was stopped before completion. It contains the base model with the training configuration applied, including your custom instruction template for invoice OCR.

## Model Details

- **Base Model**: Qwen/Qwen2.5-VL-3B-Instruct
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
    "your-username/qwen25-vl-invoice-ocr-config",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("your-username/qwen25-vl-invoice-ocr-config")

# Load image
image = Image.open("invoice.jpg")

# Use the same prompt template from training
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

# Generate
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    
# Decode response
response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print(response)
```

## Training Configuration Used

The model was configured with:
- **LoRA Rank**: 8
- **LoRA Alpha**: 16
- **Target Modules**: q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj
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

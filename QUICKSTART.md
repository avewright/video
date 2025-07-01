# 🚀 Quick Start Guide for Qwen2.5-VL Invoice OCR

Get your Qwen2.5-VL model fine-tuned for invoice OCR in minutes!

## ⚡ Instant Setup

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd qwen25-vl-invoice-ocr

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Environment
```bash
python scripts/validate_setup.py
```

### 3. Prepare Dataset
```bash
python scripts/prepare_dataset.py
```

### 4. Start Training (QLoRA - Memory Efficient)
```bash
python train.py --config configs/qwen25_3b_qlora.yaml
```

### 5. Test Inference
```bash
# CLI inference
python inference.py --model_path ./outputs/checkpoint-best --image data/samples/sample_0_image.jpg

# Interactive demo
python inference.py --model_path ./outputs/checkpoint-best --demo
```

## 🎯 What You Get

✅ **Production-ready repository** with complete training pipeline  
✅ **Memory-efficient QLoRA training** (18GB+ VRAM)  
✅ **Interactive Gradio interface** for testing  
✅ **Comprehensive validation scripts** to check your setup  
✅ **Dataset exploration notebooks** to understand your data  
✅ **Professional logging** with Weights & Biases integration  
✅ **Easy deployment** with checkpoint conversion utilities  

## 📊 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | 18GB VRAM | 24GB+ VRAM |
| **RAM** | 16GB | 32GB+ |
| **CUDA** | 11.8+ | 12.0+ |
| **Python** | 3.8+ | 3.10+ |

## 🛠️ Key Commands

```bash
# Validate everything is working
python scripts/validate_setup.py --check_model

# Explore the dataset
jupyter notebook notebooks/dataset_exploration.ipynb

# Train with full monitoring
python train.py --config configs/qwen25_3b_qlora.yaml --wandb_project my-invoice-ocr

# Convert LoRA to full model
python scripts/convert_checkpoint.py --base_model Qwen/Qwen2.5-VL-3B-Instruct --adapter_path ./outputs/checkpoint-best --output_path ./final_model

# Launch demo interface
python inference.py --model_path ./final_model --demo --share
```

## 🔍 Common Issues & Solutions

### ❓ CUDA Out of Memory
```bash
# Reduce batch size
python train.py --config configs/qwen25_3b_qlora.yaml --per_device_train_batch_size 1
```

### ❓ Dataset Loading Issues
```bash
# Force re-download dataset
python scripts/prepare_dataset.py --force_download
```

### ❓ Model Loading Errors
```bash
# Check model accessibility
python scripts/validate_setup.py --check_model
```

## 📈 Training Progress

Monitor your training with:
- **Weights & Biases**: Real-time metrics and visualizations
- **TensorBoard**: Local training logs
- **Console Output**: Live training progress

## 🎉 Expected Results

After training, you'll have a model that can:
- 📄 Process invoice/receipt images
- 🔍 Extract structured information
- 📊 Output clean JSON format
- 🚀 Handle batch processing
- 💻 Run via web interface

## 📞 Need Help?

1. **Check the logs**: Look in `outputs/logs/` for detailed error messages
2. **Run validation**: `python scripts/validate_setup.py` catches most issues
3. **Check dataset**: Use the exploration notebook to understand your data
4. **Review configs**: All settings are in `configs/qwen25_3b_qlora.yaml`

## 🔗 What's Next?

- **Deploy**: Use `inference.py` to create a production API
- **Scale**: Batch process multiple invoices efficiently  
- **Customize**: Modify training configs for your specific use case
- **Share**: Convert and upload your model to HuggingFace Hub

---

**Happy Training! 🎯** Get your invoice OCR model ready in under an hour! 
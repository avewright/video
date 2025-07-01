# 🚀 How to Save and Upload Your Trained Model

## 📋 Prerequisites

First, install the required packages:
```bash
pip install huggingface_hub
```

Set your HuggingFace token (get one from https://huggingface.co/settings/tokens):
```bash
export HF_TOKEN="your_hf_token_here"
```

## 🔄 Option 1: Save Model Locally Only

Wait for training to complete or reach a checkpoint (step 500+), then:

```bash
python scripts/save_model.py \
    --output_path ./saved_models/qwen25-vl-invoice-ocr
```

This will:
- ✅ Find the latest checkpoint automatically
- ✅ Merge LoRA adapters with base model
- ✅ Save the full model locally
- ✅ Create a model card (README.md)

## 🌐 Option 2: Save and Upload to HuggingFace Hub

```bash
python scripts/save_model.py \
    --output_path ./saved_models/qwen25-vl-invoice-ocr \
    --push_to_hub \
    --hf_repo_name "your-username/qwen25-vl-invoice-ocr"
```

## 📤 Option 3: Upload Existing Saved Model

If you already have a saved model locally:

```bash
python scripts/upload_to_hf.py \
    --model_path ./saved_models/qwen25-vl-invoice-ocr \
    --repo_name "your-username/qwen25-vl-invoice-ocr" \
    --update_model_card
```

## 🎯 Advanced Usage

### Specify a Specific Checkpoint
```bash
python scripts/save_model.py \
    --adapter_path ./outputs/qwen25-3b-qlora-invoice/checkpoint-500 \
    --output_path ./saved_models/checkpoint-500-model
```

### Upload as Private Repository
```bash
python scripts/upload_to_hf.py \
    --model_path ./saved_models/qwen25-vl-invoice-ocr \
    --repo_name "your-username/qwen25-vl-invoice-ocr" \
    --private
```

### Custom Commit Message
```bash
python scripts/upload_to_hf.py \
    --model_path ./saved_models/qwen25-vl-invoice-ocr \
    --repo_name "your-username/qwen25-vl-invoice-ocr" \
    --commit_message "Upload v1.0 - fine-tuned on 2K invoice samples"
```

## 📊 Check Training Progress

Before saving, check if training has created any checkpoints:
```bash
# Check if checkpoints exist
ls -la outputs/qwen25-3b-qlora-invoice/

# Check training status
ps aux | grep train.py

# View TensorBoard (if running)
# http://localhost:6006
```

## 🔧 Troubleshooting

### If No Checkpoints Found
- Training might not have reached step 500 yet
- Check training logs for errors
- Wait for training to complete

### If Upload Fails
- Verify your HF_TOKEN is set correctly
- Check internet connection
- Ensure repository name is unique

### If Memory Issues During Save
- Close other processes
- Use smaller batch size during conversion
- Consider saving after training completes

## 📝 Example Complete Workflow

```bash
# 1. Wait for training to complete or reach checkpoint
# 2. Check if checkpoints exist
ls outputs/qwen25-3b-qlora-invoice/checkpoint-*

# 3. Set HuggingFace token
export HF_TOKEN="your_token_here"

# 4. Save and upload in one go
python scripts/save_model.py \
    --output_path ./saved_models/qwen25-vl-invoice-ocr \
    --push_to_hub \
    --hf_repo_name "your-username/qwen25-vl-invoice-ocr"

# 5. Your model is now available at:
# https://huggingface.co/your-username/qwen25-vl-invoice-ocr
```

## 🎉 What You Get

After running these scripts, you'll have:
- ✅ **Local saved model** ready for inference
- ✅ **HuggingFace Hub repository** with your model
- ✅ **Detailed model card** with usage instructions
- ✅ **Training metadata** preserved in JSON
- ✅ **Ready-to-use code examples** in the README

Your model will be ready for production use! 🚀 
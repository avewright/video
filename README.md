# Qwen2.5-VL 3B Fine-tuning for Invoice/Receipt OCR-to-JSON

This repository provides a complete setup for fine-tuning Qwen2.5-VL 3B model on the `mychen76/invoices-and-receipts_ocr_v1` dataset to convert OCR text from invoices and receipts into structured JSON format.

## 🚀 Quick Start

```bash
# Clone the repository
git clone <your-repo-url>
cd qwen25-vl-invoice-ocr

# Setup environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download and prepare the dataset
python scripts/prepare_dataset.py

# Start training
python train.py --config configs/qwen25_3b_config.yaml
```

## 📋 Requirements

- **GPU**: Minimum 18GB VRAM (RTX 4090, A100, etc.)
- **Python**: 3.8+
- **CUDA**: 11.8 or higher
- **Memory**: 32GB+ RAM recommended

## 📊 Dataset

This project uses the `mychen76/invoices-and-receipts_ocr_v1` dataset, which contains:
- Invoice and receipt images
- OCR text with bounding boxes
- Structured JSON outputs
- Training for converting OCR results to structured data

### Dataset Format
The dataset follows this structure:
```json
{
  "image": "path/to/image.jpg",
  "ocr_text": "[[[[184.0, 42.0], [278.0, 45.0]], ('COMPANY NAME', 0.95)], ...]",
  "structured_output": {
    "company": "COMPANY NAME",
    "total": "15.99",
    "date": "2024-01-15",
    "items": [...]
  }
}
```

## 🏗️ Repository Structure

```
qwen25-vl-invoice-ocr/
├── configs/                 # Training configurations
│   ├── qwen25_3b_config.yaml
│   └── qwen25_3b_qlora.yaml
├── data/                   # Dataset storage
│   ├── raw/
│   ├── processed/
│   └── samples/
├── scripts/                # Utility scripts
│   ├── prepare_dataset.py
│   ├── validate_setup.py
│   └── convert_checkpoint.py
├── src/                    # Source code
│   ├── data/
│   ├── models/
│   ├── training/
│   └── utils/
├── notebooks/              # Jupyter notebooks
│   ├── dataset_exploration.ipynb
│   └── inference_demo.ipynb
├── tests/                  # Test files
├── train.py               # Main training script
├── inference.py           # Inference script
├── requirements.txt       # Dependencies
└── README.md
```

## 🛠️ Installation

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 2. Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python scripts/validate_setup.py
```

## 🎯 Training

### Basic Training

```bash
# Full fine-tuning (requires high VRAM)
python train.py --config configs/qwen25_3b_config.yaml

# QLoRA training (memory efficient)
python train.py --config configs/qwen25_3b_qlora.yaml
```

### Custom Training

```python
from src.training.trainer import Qwen25VLTrainer

trainer = Qwen25VLTrainer(
    model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    dataset_name="mychen76/invoices-and-receipts_ocr_v1",
    output_dir="./outputs",
    batch_size=1,
    learning_rate=2e-4,
    num_epochs=3,
    use_qlora=True
)

trainer.train()
```

## 🔍 Inference

### Using the Fine-tuned Model

```python
from src.models.qwen_vl_model import QwenVLInvoiceModel

model = QwenVLInvoiceModel.from_pretrained("./outputs/checkpoint-best")

# Process an invoice image
result = model.process_invoice(
    image_path="data/samples/invoice.jpg",
    ocr_text="extracted_ocr_text_here"
)

print(result)  # Structured JSON output
```

### Gradio Interface

```bash
# Launch interactive demo
python inference.py --demo
```

## ⚙️ Configuration

### Training Configuration (`configs/qwen25_3b_qlora.yaml`)

```yaml
model:
  name: "Qwen/Qwen2.5-VL-3B-Instruct"
  use_qlora: true
  lora_rank: 8
  lora_alpha: 16
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

training:
  batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 2e-4
  num_epochs: 3
  warmup_steps: 100
  max_seq_length: 2048

dataset:
  name: "mychen76/invoices-and-receipts_ocr_v1"
  train_split: "train"
  val_split: "validation"
  preprocessing:
    max_image_size: 1280
    instruction_template: "Convert the following OCR text to structured JSON format:"
```

## 📈 Monitoring

### Weights & Biases Integration

```bash
# Login to W&B
wandb login

# Train with logging
python train.py --config configs/qwen25_3b_qlora.yaml --wandb_project qwen25-invoice-ocr
```

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir outputs/logs

# View at http://localhost:6006
```

## 🧪 Evaluation

```bash
# Evaluate on test set
python scripts/evaluate.py --checkpoint outputs/checkpoint-best --test_data data/test.json

# Generate sample outputs
python scripts/generate_samples.py --checkpoint outputs/checkpoint-best
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size or use gradient checkpointing
   python train.py --config configs/qwen25_3b_qlora.yaml --batch_size 1 --gradient_checkpointing
   ```

2. **Dataset Loading Issues**
   ```bash
   # Re-download dataset
   python scripts/prepare_dataset.py --force_download
   ```

3. **Model Loading Errors**
   ```bash
   # Check model compatibility
   python scripts/validate_setup.py --check_model
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the Apache 2.0 License. See `LICENSE` for details.

## 🙏 Acknowledgments

- [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) by Alibaba Cloud
- [mychen76/invoices-and-receipts_ocr_v1](https://huggingface.co/datasets/mychen76/invoices-and-receipts_ocr_v1) dataset
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft) for LoRA implementation

## 📞 Support

- Create an [Issue](https://github.com/your-username/qwen25-vl-invoice-ocr/issues) for bug reports
- Check [Discussions](https://github.com/your-username/qwen25-vl-invoice-ocr/discussions) for questions
- Review [Wiki](https://github.com/your-username/qwen25-vl-invoice-ocr/wiki) for detailed guides 
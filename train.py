#!/usr/bin/env python3
"""
Main training script for Qwen2.5-VL fine-tuning on invoice OCR-to-JSON dataset.
Supports both full fine-tuning and QLoRA training modes.

Usage:
    python train.py --config configs/qwen25_3b_qlora.yaml
    python train.py --config configs/qwen25_3b_config.yaml --wandb_project my-project
"""

import argparse
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
import yaml
import wandb
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
from qwen_vl_utils import process_vision_info

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune."""
    
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading the model"}
    )
    use_qlora: bool = field(
        default=False,
        metadata={"help": "Whether to use QLoRA for parameter-efficient fine-tuning"}
    )


@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""
    
    dataset_name: str = field(
        default="mychen76/invoices-and-receipts_ocr_v1",
        metadata={"help": "Name of the dataset to use"}
    )
    train_split: str = field(
        default="train",
        metadata={"help": "Train split of the dataset"}
    )
    validation_split: str = field(
        default="validation", 
        metadata={"help": "Validation split of the dataset"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length for input"}
    )
    max_image_size: int = field(
        default=1280,
        metadata={"help": "Maximum image size for preprocessing"}
    )


class InvoiceOCRDataset:
    """Custom dataset class for handling invoice OCR data."""
    
    def __init__(
        self,
        dataset_name: str,
        split: str,
        processor,
        instruction_template: str,
        max_seq_length: int = 2048,
        max_image_size: int = 1280,
    ):
        self.processor = processor
        self.instruction_template = instruction_template
        self.max_seq_length = max_seq_length
        self.max_image_size = max_image_size
        
        # Load dataset
        logger.info(f"Loading dataset {dataset_name}, split: {split}")
        self.dataset = load_dataset(dataset_name, split=split)
        logger.info(f"Loaded {len(self.dataset)} examples")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        return self.process_example(example)
    
    def process_example(self, example):
        """Process a single example into the format expected by Qwen2.5-VL."""
        
        # Extract components
        image = example.get("image")
        ocr_text = example.get("ocr_text", "")
        target_json = example.get("structured_output", {})
        
        # Format instruction
        instruction = self.instruction_template.format(ocr_text=ocr_text)
        
        # Create conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction}
                ]
            },
            {
                "role": "assistant", 
                "content": [
                    {"type": "text", "text": json.dumps(target_json, indent=2)}
                ]
            }
        ]
        
        return {
            "conversation": conversation,
            "image": image,
            "target": json.dumps(target_json, indent=2)
        }


def collate_fn(batch, processor, max_seq_length=2048):
    """Custom collate function for batching examples."""
    
    conversations = [item["conversation"] for item in batch]
    
    # Apply chat template
    texts = []
    image_inputs = []
    
    for conversation in conversations:
        # Apply chat template
        text = processor.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=False
        )
        texts.append(text)
        
        # Process vision info
        images, _ = process_vision_info(conversation)
        image_inputs.append(images)
    
    # Process inputs
    inputs = processor(
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        padding=True,
        max_length=max_seq_length,
        truncation=True
    )
    
    # Create labels for training
    labels = inputs["input_ids"].clone()
    
    # Mask padding tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    # Mask image tokens (specific to Qwen2.5-VL)
    image_token_ids = [151652, 151653, 151655]  # Qwen2.5-VL specific tokens
    for token_id in image_token_ids:
        labels[labels == token_id] = -100
    
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "pixel_values": inputs["pixel_values"],
        "image_grid_thw": inputs["image_grid_thw"],
        "labels": labels
    }


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_model_and_processor(config: Dict[str, Any]):
    """Setup model and processor based on configuration."""
    
    model_config = config["model"]
    model_name = model_config["name"]
    
    # Setup quantization config for QLoRA
    quantization_config = None
    if model_config.get("use_qlora", False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=model_config.get("load_in_4bit", True),
            bnb_4bit_compute_dtype=getattr(torch, model_config.get("bnb_4bit_compute_dtype", "bfloat16")),
            bnb_4bit_use_double_quant=model_config.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_quant_type=model_config.get("bnb_4bit_quant_type", "nf4"),
        )
    
    # Load model
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=model_config.get("trust_remote_code", True),
        torch_dtype=getattr(torch, model_config.get("torch_dtype", "bfloat16")),
    )
    
    # Load processor
    logger.info(f"Loading processor: {model_name}")
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=model_config.get("trust_remote_code", True),
    )
    
    # Setup LoRA if needed
    if model_config.get("use_qlora", False) or "lora_config" in model_config:
        logger.info("Setting up LoRA configuration")
        
        if model_config.get("use_qlora", False):
            model = prepare_model_for_kbit_training(model)
        
        lora_config_dict = model_config.get("lora_config", {})
        lora_config = LoraConfig(
            r=lora_config_dict.get("r", 8),
            lora_alpha=lora_config_dict.get("lora_alpha", 16),
            lora_dropout=lora_config_dict.get("lora_dropout", 0.05),
            bias=lora_config_dict.get("bias", "none"),
            task_type=lora_config_dict.get("task_type", "CAUSAL_LM"),
            target_modules=lora_config_dict.get("target_modules", ["q_proj", "v_proj"]),
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, processor


def setup_datasets(config: Dict[str, Any], processor):
    """Setup training and validation datasets."""
    
    dataset_config = config["dataset"]
    instruction_template = dataset_config.get("instruction_template", "Convert the OCR text to JSON: {ocr_text}")
    
    # Create datasets
    train_dataset = InvoiceOCRDataset(
        dataset_name=dataset_config["name"],
        split=dataset_config.get("train_split", "train"),
        processor=processor,
        instruction_template=instruction_template,
        max_seq_length=config.get("training", {}).get("model_max_length", 2048),
        max_image_size=dataset_config.get("preprocessing", {}).get("max_image_size", 1280),
    )
    
    eval_dataset = None
    if dataset_config.get("validation_split"):
        eval_dataset = InvoiceOCRDataset(
            dataset_name=dataset_config["name"],
            split=dataset_config["validation_split"],
            processor=processor,
            instruction_template=instruction_template,
            max_seq_length=config.get("training", {}).get("model_max_length", 2048),
            max_image_size=dataset_config.get("preprocessing", {}).get("max_image_size", 1280),
        )
    
    return train_dataset, eval_dataset


def setup_training_args(config: Dict[str, Any]) -> TrainingArguments:
    """Setup training arguments from configuration."""
    
    training_config = config.get("training", {})
    
    return TrainingArguments(
        output_dir=training_config.get("output_dir", "./outputs"),
        overwrite_output_dir=training_config.get("overwrite_output_dir", True),
        
        # Batch settings
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 8),
        dataloader_num_workers=training_config.get("dataloader_num_workers", 4),
        
        # Learning rate
        learning_rate=training_config.get("learning_rate", 2e-4),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=training_config.get("warmup_ratio", 0.03),
        warmup_steps=training_config.get("warmup_steps", 100),
        
        # Training duration
        num_train_epochs=training_config.get("num_train_epochs", 3),
        max_steps=training_config.get("max_steps", -1),
        
        # Evaluation and saving
        evaluation_strategy=training_config.get("evaluation_strategy", "steps"),
        eval_steps=training_config.get("eval_steps", 500),
        save_strategy=training_config.get("save_strategy", "steps"),
        save_steps=training_config.get("save_steps", 500),
        save_total_limit=training_config.get("save_total_limit", 3),
        load_best_model_at_end=training_config.get("load_best_model_at_end", True),
        metric_for_best_model=training_config.get("metric_for_best_model", "eval_loss"),
        greater_is_better=training_config.get("greater_is_better", False),
        
        # Optimization
        optim=training_config.get("optim", "adamw_torch"),
        weight_decay=training_config.get("weight_decay", 0.01),
        adam_beta1=training_config.get("adam_beta1", 0.9),
        adam_beta2=training_config.get("adam_beta2", 0.999),
        adam_epsilon=training_config.get("adam_epsilon", 1e-8),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        
        # Memory optimization
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        bf16=training_config.get("bf16", True),
        fp16=training_config.get("fp16", False),
        remove_unused_columns=training_config.get("remove_unused_columns", False),
        
        # Logging
        logging_strategy=training_config.get("logging_strategy", "steps"),
        logging_steps=training_config.get("logging_steps", 10),
        report_to=training_config.get("report_to", ["tensorboard"]),
        
        # DeepSpeed
        deepspeed=training_config.get("deepspeed"),
        
        # Reproducibility
        seed=config.get("seed", 42),
        data_seed=config.get("data_seed", 42),
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL on invoice OCR dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--wandb_project", type=str, help="Weights & Biases project name")
    parser.add_argument("--wandb_name", type=str, help="Weights & Biases run name")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set seed for reproducibility
    set_seed(config.get("seed", 42))
    
    # Setup Weights & Biases if configured
    wandb_config = config.get("wandb", {})
    if wandb_config or args.wandb_project:
        wandb.init(
            project=args.wandb_project or wandb_config.get("project", "qwen25-vl-invoice"),
            name=args.wandb_name or wandb_config.get("name"),
            tags=wandb_config.get("tags", []),
            notes=wandb_config.get("notes", ""),
            config=config,
        )
    
    # Setup model and processor
    model, processor = setup_model_and_processor(config)
    
    # Setup datasets
    train_dataset, eval_dataset = setup_datasets(config, processor)
    
    # Setup training arguments
    training_args = setup_training_args(config)
    
    # Create custom collate function
    def data_collator(batch):
        return collate_fn(batch, processor, training_args.model_max_length)
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    # Start training
    logger.info("Starting training...")
    if list(training_args.output_dir.glob("checkpoint-*")):
        logger.info("Resuming from checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model()
    processor.save_pretrained(training_args.output_dir)
    
    # Save training metrics
    metrics = trainer.state.log_history
    with open(os.path.join(training_args.output_dir, "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main() 
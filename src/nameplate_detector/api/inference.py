#!/usr/bin/env python3
"""
Inference script for Qwen2.5-VL Invoice OCR fine-tuned model.
Supports both CLI inference and Gradio demo interface.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

import torch
from PIL import Image
import gradio as gr
from transformers import AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QwenVLInvoiceProcessor:
    """Qwen2.5-VL processor for invoice OCR to JSON conversion."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize the processor.
        
        Args:
            model_path: Path to the fine-tuned model or HuggingFace model name
            device: Device to run inference on ("cuda", "cpu", or "auto")
        """
        self.model_path = model_path
        self.device = self._setup_device(device)
        
        logger.info(f"Loading model from {model_path}")
        logger.info(f"Using device: {self.device}")
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True
        )
        
        logger.info("Model loaded successfully!")
    
    def _setup_device(self, device: str) -> str:
        """Setup the appropriate device for inference."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def process_invoice(
        self, 
        image_path: Union[str, Image.Image], 
        ocr_text: Optional[str] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.8
    ) -> Dict[str, Any]:
        """
        Process an invoice image and convert OCR text to structured JSON.
        
        Args:
            image_path: Path to image file or PIL Image object
            ocr_text: Pre-extracted OCR text (optional)
            max_new_tokens: Maximum tokens to generate
            temperature: Generation temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Dictionary containing the structured JSON output
        """
        try:
            # Load image
            if isinstance(image_path, str):
                image = Image.open(image_path).convert('RGB')
            else:
                image = image_path.convert('RGB')
            
            # Create instruction prompt
            if ocr_text:
                instruction = f"""Convert the following OCR text from an invoice/receipt into structured JSON format:

OCR Text: {ocr_text}

Please extract and structure the information into JSON with fields like:
- company/vendor name
- date
- total amount
- items (if applicable)
- other relevant invoice details

Output only valid JSON:"""
            else:
                instruction = """Analyze this invoice/receipt image and extract the information into structured JSON format.

Please extract and structure the information into JSON with fields like:
- company/vendor name  
- date
- total amount
- items (if applicable)
- other relevant invoice details

Output only valid JSON:"""
            
            # Prepare messages for the model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": instruction}
                    ]
                }
            ]
            
            # Process with the model
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0
                )
            
            # Decode the response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in 
                zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            # Try to parse as JSON
            try:
                parsed_json = json.loads(output_text.strip())
                return {
                    "success": True,
                    "structured_data": parsed_json,
                    "raw_output": output_text,
                    "error": None
                }
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON: {e}")
                return {
                    "success": False,
                    "structured_data": None,
                    "raw_output": output_text,
                    "error": f"JSON parsing error: {str(e)}"
                }
                
        except Exception as e:
            logger.error(f"Error processing invoice: {e}")
            return {
                "success": False,
                "structured_data": None,
                "raw_output": None,
                "error": str(e)
            }

def create_gradio_interface(processor: QwenVLInvoiceProcessor):
    """Create a Gradio interface for the invoice processor."""
    
    def process_image_gradio(image, ocr_text, max_tokens, temperature, top_p):
        """Gradio-compatible processing function."""
        if image is None:
            return "Please upload an image.", "{}", "No image provided"
        
        result = processor.process_invoice(
            image_path=image,
            ocr_text=ocr_text if ocr_text.strip() else None,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p)
        )
        
        if result["success"]:
            structured_json = json.dumps(result["structured_data"], indent=2)
            status = "✅ Successfully extracted structured data"
        else:
            structured_json = result["raw_output"] or "{}"
            status = f"❌ Error: {result['error']}"
        
        return status, structured_json, result["raw_output"] or ""
    
    # Create the interface
    with gr.Blocks(title="Qwen2.5-VL Invoice OCR Processor") as demo:
        gr.Markdown("# 🧾 Qwen2.5-VL Invoice OCR to JSON Converter")
        gr.Markdown("Upload an invoice/receipt image to extract structured information.")
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil", 
                    label="Upload Invoice/Receipt Image"
                )
                ocr_text_input = gr.Textbox(
                    label="Pre-extracted OCR Text (Optional)",
                    placeholder="Paste OCR text here if available...",
                    lines=5
                )
                
                with gr.Accordion("Advanced Settings", open=False):
                    max_tokens_slider = gr.Slider(
                        minimum=128, maximum=2048, value=1024,
                        label="Max New Tokens"
                    )
                    temperature_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.1, step=0.1,
                        label="Temperature"
                    )
                    top_p_slider = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.8, step=0.1,
                        label="Top-p"
                    )
                
                process_btn = gr.Button("🚀 Process Invoice", variant="primary")
            
            with gr.Column(scale=1):
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False
                )
                json_output = gr.Code(
                    label="Structured JSON Output",
                    language="json",
                    interactive=False
                )
                raw_output = gr.Textbox(
                    label="Raw Model Output",
                    lines=10,
                    interactive=False
                )
        
        # Example images section
        gr.Markdown("## 📸 Example Images")
        gr.Markdown("Click on an example to process it:")
        
        example_images = []
        samples_dir = Path("data/samples")
        if samples_dir.exists():
            for img_file in samples_dir.glob("*.jpg"):
                example_images.append([str(img_file)])
            for img_file in samples_dir.glob("*.png"):
                example_images.append([str(img_file)])
        
        if example_images:
            gr.Examples(
                examples=example_images,
                inputs=[image_input],
                outputs=[status_output, json_output, raw_output],
                fn=lambda img: process_image_gradio(img, "", 1024, 0.1, 0.8),
                cache_examples=False
            )
        
        # Connect the button
        process_btn.click(
            fn=process_image_gradio,
            inputs=[
                image_input, ocr_text_input, max_tokens_slider,
                temperature_slider, top_p_slider
            ],
            outputs=[status_output, json_output, raw_output]
        )
    
    return demo

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL Invoice OCR Inference")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to fine-tuned model or HuggingFace model name")
    parser.add_argument("--image", type=str,
                       help="Path to invoice image for processing")
    parser.add_argument("--ocr_text", type=str,
                       help="Pre-extracted OCR text")
    parser.add_argument("--output", type=str,
                       help="Output file path for results")
    parser.add_argument("--demo", action="store_true",
                       help="Launch Gradio demo interface")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device for inference")
    parser.add_argument("--max_tokens", type=int, default=1024,
                       help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.8,
                       help="Top-p sampling parameter")
    parser.add_argument("--share", action="store_true",
                       help="Create shareable Gradio link")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = QwenVLInvoiceProcessor(
        model_path=args.model_path,
        device=args.device
    )
    
    if args.demo:
        # Launch Gradio demo
        logger.info("Launching Gradio demo...")
        demo = create_gradio_interface(processor)
        demo.launch(share=args.share, server_name="0.0.0.0", server_port=7860)
    
    elif args.image:
        # Process single image
        logger.info(f"Processing image: {args.image}")
        
        result = processor.process_invoice(
            image_path=args.image,
            ocr_text=args.ocr_text,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        # Output results
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {args.output}")
        else:
            print("\n" + "="*60)
            print("INVOICE PROCESSING RESULTS")
            print("="*60)
            print(f"Status: {'✅ Success' if result['success'] else '❌ Failed'}")
            
            if result['success']:
                print("\nStructured JSON Output:")
                print(json.dumps(result['structured_data'], indent=2, ensure_ascii=False))
            else:
                print(f"\nError: {result['error']}")
                if result['raw_output']:
                    print(f"\nRaw Output:\n{result['raw_output']}")
    
    else:
        logger.error("Please specify either --image for single inference or --demo for interactive interface")
        parser.print_help()

if __name__ == "__main__":
    main() 
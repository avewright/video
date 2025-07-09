#!/usr/bin/env python3
"""
FastAPI server for Qwen2.5-VL Invoice OCR fine-tuned model.
Provides REST API endpoint that matches the expected format.
"""

import os
import json
import logging
import io
from typing import Dict, Any, Optional, Union
from pathlib import Path

import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Import the processor class from inference.py
from inference import QwenVLInvoiceProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Qwen2.5-VL Invoice OCR API", description="REST API for invoice field extraction")

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global processor instance
processor = None

def initialize_processor():
    """Initialize the model processor."""
    global processor
    if processor is None:
        # Try to find the model path
        model_paths = [
            "saved_models/kahua-invoice-ocr",
            "saved_models/qwen25-vl-invoice-config",
            "Qwen/Qwen2.5-VL-3B-Instruct"  # Fallback to base model
        ]
        
        model_path = None
        for path in model_paths:
            if Path(path).exists():
                model_path = path
                break
        
        if not model_path:
            model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
            logger.warning(f"No local model found, using base model: {model_path}")
        
        logger.info(f"Initializing processor with model: {model_path}")
        processor = QwenVLInvoiceProcessor(model_path=model_path)
        logger.info("Processor initialized successfully!")

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    initialize_processor()

@app.post("/inference")
async def inference(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    max_new_tokens: int = Form(512),
    temperature: float = Form(0.7)
):
    """
    Process an image with the specified parameters.
    
    Args:
        prompt: Text prompt for the model
        image: Image file to process
        max_new_tokens: Maximum tokens to generate
        temperature: Generation temperature
    
    Returns:
        JSON response with extracted fields
    """
    try:
        if processor is None:
            raise HTTPException(status_code=503, detail="Model not initialized")
        
        # Read and process the uploaded image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Process the image using the processor
        result = processor.process_invoice(
            image_path=pil_image,
            ocr_text=None,  # We'll use the image directly
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.8
        )
        
        # Return the result in the expected format
        if result["success"]:
            return JSONResponse(
                content={
                    "success": True,
                    "extracted_fields": result["structured_data"],
                    "raw_output": result["raw_output"]
                }
            )
        else:
            return JSONResponse(
                content={
                    "success": False,
                    "error": result["error"],
                    "raw_output": result["raw_output"]
                },
                status_code=500
            )
            
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": processor is not None}

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Qwen2.5-VL Invoice OCR API",
        "endpoints": {
            "inference": "/inference (POST)",
            "health": "/health (GET)"
        }
    }

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    ) 
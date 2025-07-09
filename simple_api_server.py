#!/usr/bin/env python3
"""
Simple API server for nameplate detection and field extraction.
No complex setup required - just handles HTTP requests directly.
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import io
import base64
import json
import logging
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Simple Nameplate Detector API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple nameplate classifier model
class LightweightNameplateClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(LightweightNameplateClassifier, self).__init__()
        self.backbone = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# Global model instance
model = None
device = None
transform = None

def load_model():
    """Load the nameplate detection model."""
    global model, device, transform
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Try to load the model
    model_paths = [
        "models/best_nameplate_classifier.pth",
        "best_nameplate_classifier.pth",
        "../models/best_nameplate_classifier.pth"
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path:
        try:
            model = LightweightNameplateClassifier(num_classes=2)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            model.to(device)
            logger.info(f"✅ Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            model = None
    else:
        logger.warning("⚠️ No model file found - using simulation mode")
        model = None

def detect_nameplate(image: Image.Image) -> dict:
    """Detect if image contains a nameplate."""
    if model is None:
        # Simulation mode - randomly detect nameplates
        import random
        confidence = random.uniform(0.3, 0.9)
        detected = confidence > 0.7
        return {
            "detected": detected,
            "confidence": confidence,
            "mode": "simulation"
        }
    
    try:
        # Preprocess image
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted = torch.max(outputs, 1)[1].item()
            confidence = probabilities[0][predicted].item()
        
        has_nameplate = predicted == 1
        
        return {
            "detected": has_nameplate,
            "confidence": confidence,
            "mode": "model"
        }
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return {
            "detected": False,
            "confidence": 0.0,
            "mode": "error",
            "error": str(e)
        }

def extract_fields_simulation(image: Image.Image) -> dict:
    """Simulate field extraction for demo purposes."""
    import random
    
    # Sample nameplate data
    manufacturers = ["Siemens", "ABB", "Schneider Electric", "General Electric", "Eaton"]
    models = ["1LA7-090", "M3BP-132", "TeSys-D", "5K-180", "C25-DND"]
    
    fields = {
        "manufacturer": random.choice(manufacturers),
        "model": random.choice(models),
        "serial_number": f"SN{random.randint(100000, 999999)}",
        "voltage": f"{random.choice([220, 380, 400, 440, 480])}V",
        "power": f"{random.uniform(0.5, 50):.1f}kW",
        "frequency": f"{random.choice([50, 60])}Hz",
        "current": f"{random.uniform(1, 100):.1f}A",
        "year": str(random.randint(2015, 2024)),
        "type": random.choice(["Motor", "Transformer", "Controller"]),
        "part_number": f"PN{random.randint(1000, 9999)}",
        "rating": f"IP{random.choice([54, 55, 65])}",
        "phase": str(random.choice([1, 3]))
    }
    
    return {
        "success": True,
        "extracted_fields": fields,
        "mode": "simulation"
    }

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    load_model()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown"
    }

@app.post("/inference/upload")
async def inference_upload(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    max_new_tokens: int = Form(512)
):
    """Handle inference upload requests with the new API format."""
    try:
        # Read and process image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Field extraction - return JSON response in the expected format
        fields_result = extract_fields_simulation(pil_image)
        
        # Format response as JSON string with markdown
        json_response = json.dumps(fields_result["extracted_fields"], indent=2)
        
        return JSONResponse(content={
            "response": f"```json\n{json_response}\n```",
            "success": True,
            "error": None
        })
            
    except Exception as e:
        logger.error(f"Upload API error: {e}")
        return JSONResponse(
            content={
                "response": f"```json\n{{}}\n```",
                "success": False,
                "error": str(e)
            },
            status_code=500
        )

@app.post("/inference")
async def inference(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    max_new_tokens: int = Form(512),
    temperature: float = Form(0.7)
):
    """Handle inference requests for nameplate detection and field extraction."""
    try:
        # Read and process image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Determine if this is detection or field extraction based on prompt
        if "detect nameplate" in prompt.lower():
            # Nameplate detection
            result = detect_nameplate(pil_image)
            
            if result["detected"]:
                return JSONResponse(content={
                    "success": True,
                    "raw_output": f"Nameplate detected with {result['confidence']:.2f} confidence",
                    "confidence": result["confidence"],
                    "mode": result["mode"]
                })
            else:
                return JSONResponse(content={
                    "success": False,
                    "raw_output": "No nameplate detected",
                    "confidence": result["confidence"],
                    "mode": result["mode"]
                })
        
        elif "extract" in prompt.lower() or "json" in prompt.lower():
            # Field extraction
            fields_result = extract_fields_simulation(pil_image)
            
            return JSONResponse(content={
                "success": True,
                "extracted_fields": fields_result["extracted_fields"],
                "raw_output": json.dumps(fields_result["extracted_fields"], indent=2),
                "mode": fields_result["mode"]
            })
        
        else:
            # Generic response
            return JSONResponse(content={
                "success": True,
                "raw_output": "Image processed successfully",
                "prompt": prompt
            })
            
    except Exception as e:
        logger.error(f"API error: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e),
                "raw_output": f"Error: {e}"
            },
            status_code=500
        )

if __name__ == "__main__":
    print("🚀 Starting Simple Nameplate Detector API...")
    print("📍 API will be available at: http://localhost:8000")
    print("📋 API docs at: http://localhost:8000/docs")
    print("💡 This is a simplified version - no complex server setup required!")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    ) 
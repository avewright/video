# 📹 Real-time Nameplate Detection System

A comprehensive system for detecting nameplates in real-time using your camera, powered by a lightweight MobileNetV2-based classifier.

## 🚀 Quick Start

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# Make sure you have a trained model
# (Should exist as 'best_nameplate_classifier.pth')
```

### Simple Usage
```bash
# Run the simple version
python quickstart_camera.py
```

### Advanced Usage
```bash
# Full-featured version with options
python realtime_nameplate_detector.py
```

## 📋 Available Scripts

### 1. `quickstart_camera.py` - Simple Detection
**Best for**: First-time users, quick testing

**Features**:
- ✅ Auto-pause when nameplate detected (80%+ confidence)
- ✅ Simple overlay with confidence scores
- ✅ Basic controls (Q to quit, SPACE to pause/resume)
- ✅ Detection counter

**Usage**:
```bash
python quickstart_camera.py
```

### 2. `realtime_nameplate_detector.py` - Advanced Detection
**Best for**: Advanced users, production use, fine-tuning

**Features**:
- ✅ Configurable confidence threshold
- ✅ Adjustable threshold during runtime (+/- keys)
- ✅ Detailed overlay with statistics
- ✅ Multiple camera support
- ✅ Device selection (CPU/GPU)
- ✅ Toggle overlay on/off
- ✅ Frame and detection counters

**Usage**:
```bash
# Basic usage
python realtime_nameplate_detector.py

# With custom settings
python realtime_nameplate_detector.py --threshold 0.8 --camera 1 --device cuda

# See all options
python realtime_nameplate_detector.py --help
```

## ⚙️ Configuration Options

### Command Line Arguments (Advanced Script)

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `best_nameplate_classifier.pth` | Path to trained model |
| `--camera` | `0` | Camera device ID (0, 1, 2...) |
| `--threshold` | `0.7` | Confidence threshold (0.1-1.0) |
| `--device` | `auto` | Device: `cpu`, `cuda`, or `auto` |

### Examples
```bash
# Use external camera (camera ID 1)
python realtime_nameplate_detector.py --camera 1

# Lower threshold for more sensitive detection
python realtime_nameplate_detector.py --threshold 0.5

# Force CPU usage
python realtime_nameplate_detector.py --device cpu

# Use different model file
python realtime_nameplate_detector.py --model my_custom_model.pth
```

## 🎮 Controls

### During Runtime

| Key | Action |
|-----|--------|
| **Q** or **ESC** | Quit application |
| **SPACE** | Pause/Resume detection |
| **O** | Toggle overlay on/off |
| **+** or **=** | Increase confidence threshold |
| **-** | Decrease confidence threshold |
| **R** | Reset detection counters |

## 📊 Understanding the Interface

### Overlay Information
- **Status**: Current detection state (SCANNING/PAUSED)
- **Prediction**: NAMEPLATE DETECTED or NO NAMEPLATE
- **Confidence**: Model confidence percentage
- **Frame Count**: Total frames processed
- **Detection Count**: Number of nameplates detected
- **Threshold**: Current confidence threshold

### Color Coding
- 🟢 **Green**: Nameplate detected
- 🔴 **Red**: No nameplate or paused state
- 🟠 **Orange**: Nameplate detected but below threshold

## 🛠️ Troubleshooting

### Camera Issues

**Problem**: "Could not open camera"
```bash
# Try different camera IDs
python realtime_nameplate_detector.py --camera 1
python realtime_nameplate_detector.py --camera 2
```

**Problem**: Poor camera quality
- Ensure good lighting
- Clean camera lens
- Check camera resolution in Device Manager (Windows)

### Performance Issues

**Problem**: Slow detection/low FPS
```bash
# Use CPU instead of GPU (sometimes faster for small models)
python realtime_nameplate_detector.py --device cpu

# Lower camera resolution (modify script if needed)
```

**Problem**: High GPU memory usage
```bash
# Force CPU usage
python realtime_nameplate_detector.py --device cpu
```

### Model Issues

**Problem**: "Model file not found"
```bash
# Make sure you have trained the model first
python train_nameplate_classifier.py

# Or specify custom model path
python realtime_nameplate_detector.py --model path/to/your/model.pth
```

**Problem**: Poor detection accuracy
- Adjust confidence threshold: `--threshold 0.5` (more sensitive) or `--threshold 0.9` (less sensitive)
- Ensure good lighting conditions
- Hold nameplates steady and clearly visible
- Consider retraining model with more data

### Common Error Messages

| Error | Solution |
|-------|----------|
| `Model file not found` | Train model first or check file path |
| `Could not open camera` | Check camera connection, try different camera ID |
| `CUDA out of memory` | Use `--device cpu` |
| `ImportError: cv2` | Install OpenCV: `pip install opencv-python` |

## 🎯 Optimization Tips

### For Best Performance
1. **Good Lighting**: Ensure nameplates are well-lit
2. **Stable Camera**: Mount camera or hold steady
3. **Clear View**: Avoid obstructions, reflections
4. **Optimal Distance**: 1-3 feet from nameplate
5. **Contrast**: Ensure nameplate text contrasts with background

### For Production Use
1. Use the advanced script with custom thresholds
2. Test with your specific camera setup
3. Adjust confidence threshold based on your accuracy needs
4. Consider using a dedicated camera for better quality

## 📈 Performance Metrics

### Expected Performance
- **Model Size**: ~9.4MB
- **Inference Speed**: 20-60 FPS (depending on hardware)
- **Accuracy**: 85-95% on industrial nameplates
- **Memory Usage**: <1GB RAM
- **GPU Memory**: <500MB (if using CUDA)

### Hardware Requirements
- **Minimum**: Any computer with camera, 4GB RAM
- **Recommended**: 8GB RAM, dedicated GPU (optional)
- **Camera**: Any USB camera or built-in webcam

## 🔧 Advanced Usage

### Custom Preprocessing
Modify the `transform` in the script to customize image preprocessing:

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])
```

### Integration with Other Systems
The detection functions can be easily integrated into larger systems:

```python
from realtime_nameplate_detector import RealtimeNameplateDetector

# Create detector
detector = RealtimeNameplateDetector(
    model_path="best_nameplate_classifier.pth",
    confidence_threshold=0.8
)

# Use in your own loop
while True:
    frame = get_frame_from_somewhere()
    has_nameplate, confidence = detector.predict_frame(frame)
    
    if has_nameplate and confidence > 0.8:
        # Do something with detected nameplate
        handle_nameplate_detection(frame, confidence)
```

## 📝 Next Steps

1. **Test Detection**: Run `quickstart_camera.py` to test basic functionality
2. **Fine-tune**: Use advanced script to adjust settings
3. **Deploy**: Integrate into your workflow
4. **Expand**: Add more features or integrate with other systems

## 🆘 Support

For issues or questions:
1. Check this troubleshooting guide
2. Review the model training documentation
3. Test with sample images first using `predict_nameplate.py`
4. Ensure camera and model are working independently

## 📄 File Structure
```
project/
├── realtime_nameplate_detector.py    # Advanced detection script
├── quickstart_camera.py              # Simple detection script
├── best_nameplate_classifier.pth     # Trained model (must exist)
├── predict_nameplate.py              # Single image testing
├── train_nameplate_classifier.py     # Model training
└── requirements.txt                  # Dependencies
``` 
# 🎥 Video Streaming Site with Nameplate Detection

A real-time video streaming application that uses computer vision to detect and analyze industrial nameplates from live camera feeds.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 14+
- Camera (webcam or external)
- Required Python packages (see requirements.txt)

### Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup frontend:**
   ```bash
   cd frontend
   npm install
   cd server
   npm install
   ```

3. **Start the application:**
   ```bash
   # Terminal 1: Start the model API
   python api_server.py
   
   # Terminal 2: Start the backend server
   cd frontend/server
   npm start
   
   # Terminal 3: Start the frontend
   cd frontend
   npm start
   ```

4. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:3001
   - Model API: http://localhost:8000

## 🎯 Features

### 📹 Real-time Video Streaming
- Live camera feed processing
- WebSocket-based real-time communication
- Adjustable detection sensitivity
- Auto-pause on detection

### 🔍 Nameplate Detection
- Industrial nameplate identification
- Confidence scoring
- Visual detection overlay
- Detection history tracking

### 📋 Field Extraction
- Extract technical specifications from nameplates
- Support for 12 standard fields:
  - manufacturer, model, serial_number
  - voltage, power, frequency, current
  - year, type, part_number, rating, phase
- JSON-formatted output
- Error handling and validation

### 🖥️ Web Interface
- Modern React-based UI
- Real-time detection visualization
- Camera controls and settings
- Download detected images
- Detection statistics

## 📁 Project Structure

```
video/
├── 📂 frontend/                     # React web application
│   ├── src/App.js                   # Main React component
│   ├── server/server.js             # Node.js backend with Socket.io
│   └── python-integration/          # Python service integration
├── 📂 docs/                         # Documentation
│   ├── CAMERA_DETECTION_README.md   # Camera setup guide
│   ├── FIELD_EXTRACTION_README.md   # Field extraction guide
│   └── FRONTEND_QUICKSTART.md       # Frontend setup guide
├── 📂 data/samples/                 # Sample images and data
├── 📂 notebooks/                    # Jupyter notebooks for demos
├── 📂 src/                          # Core Python modules
├── 🎯 Core Detection Scripts
│   ├── api_server.py                # Main API server
│   ├── inference.py                 # Model inference server
│   ├── realtime_nameplate_detector.py  # Real-time detection
│   ├── camera_detector_windows.py   # Windows-compatible detector
│   ├── predict_nameplate.py         # Batch prediction
│   ├── quickstart_camera.py         # Simple camera test
│   └── test_field_extraction.py     # Field extraction testing
├── 📄 requirements.txt              # Python dependencies
├── 📄 README.md                     # This file
└── 📄 LICENSE                       # MIT License
```

## 🛠️ Core Components

### Detection System
- **Real-time Detection**: Live camera feed processing with MobileNetV2
- **Cross-platform**: Windows, macOS, Linux support
- **Multiple Interfaces**: Web UI, desktop apps, command-line tools

### Web Application
- **React Frontend**: Modern, responsive interface
- **Node.js Backend**: Express server with Socket.io
- **Python Integration**: Seamless Python-Node.js communication
- **WebSocket Communication**: Real-time updates

### Model API
- **Inference Server**: Fast model serving on port 8000
- **Field Extraction**: AI-powered text extraction
- **Batch Processing**: Support for multiple images
- **JSON API**: RESTful endpoints

## 🎮 Usage Examples

### Web Interface
```bash
# Start all services
npm run dev    # From frontend directory
```

### Command Line Detection
```bash
# Simple camera detection
python quickstart_camera.py

# Advanced detection with options
python realtime_nameplate_detector.py --threshold 0.8

# Windows-compatible version
python camera_detector_windows.py
```

### Field Extraction
```bash
# Extract fields from an image
python test_field_extraction.py --image path/to/nameplate.jpg
```

### API Usage
```bash
# Health check
curl http://localhost:8000/health

# Extract fields
curl -X POST http://localhost:8000/inference \
  -F "image=@nameplate.jpg" \
  -F "prompt=Extract nameplate fields" \
  -F "max_new_tokens=200"
```

## 🔧 Configuration

### Detection Settings
- **Confidence Threshold**: Adjust detection sensitivity (0.1-1.0)
- **Camera Selection**: Choose camera device (0, 1, 2...)
- **Processing Device**: CPU or GPU acceleration

### Web Interface Settings
- **Auto-pause**: Automatically pause on detection
- **Overlay Display**: Show/hide detection overlay
- **Download Quality**: Image save quality settings

## 📚 Documentation

Detailed guides available in the `docs/` directory:

- **[Camera Detection](docs/CAMERA_DETECTION_README.md)**: Setup and usage
- **[Field Extraction](docs/FIELD_EXTRACTION_README.md)**: Text extraction from nameplates
- **[Frontend Guide](docs/FRONTEND_QUICKSTART.md)**: Web interface setup

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For issues and questions:
1. Check the documentation in `docs/`
2. Try the quickstart scripts
3. Create an issue on GitHub

---

**Built with ❤️ for industrial applications** 
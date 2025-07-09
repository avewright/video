# 📚 Nameplate Detector Documentation

Welcome to the comprehensive documentation for the Nameplate Detector project - a real-time video streaming application with industrial nameplate detection capabilities.

## 📖 Documentation Structure

### 🚀 [User Guide](user-guide/)
- [Frontend Quickstart](user-guide/frontend-quickstart.md) - Getting started with the web interface
- [Installation Guide](user-guide/installation.md) - Step-by-step installation instructions
- [Configuration Guide](user-guide/configuration.md) - System configuration and settings

### 🔧 [API Documentation](api/)
- [Field Extraction](api/field-extraction.md) - AI-powered field extraction from nameplates
- [REST API Reference](api/rest-api.md) - Complete API endpoints documentation
- [WebSocket API](api/websocket.md) - Real-time communication interface

### 👨‍💻 [Developer Guide](developer-guide/)
- [Camera Detection](developer-guide/camera-detection.md) - Real-time camera detection implementation
- [Model Training](developer-guide/model-training.md) - Training custom nameplate models
- [Architecture Overview](developer-guide/architecture.md) - System architecture and design
- [Contributing Guidelines](developer-guide/contributing.md) - How to contribute to the project

### 🚀 [Deployment Guide](deployment/)
- [Docker Deployment](deployment/docker.md) - Containerized deployment with Docker
- [Production Setup](deployment/production.md) - Production environment configuration
- [Monitoring & Logging](deployment/monitoring.md) - System monitoring and troubleshooting

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Nameplate Detector System                   │
├─────────────────────────────────────────────────────────────────┤
│  Frontend (React)           Backend (Node.js)    API (FastAPI)  │
│  ┌─────────────────┐       ┌─────────────────┐  ┌─────────────┐ │
│  │  Web Interface  │◄──────│  WebSocket      │  │  ML Models  │ │
│  │  - Camera Feed  │       │  - Real-time    │  │  - Detector │ │
│  │  - Controls     │       │  - Proxy        │  │  - Extractor│ │
│  │  - Results      │       │  - CORS         │  │  - Inference│ │
│  └─────────────────┘       └─────────────────┘  └─────────────┘ │
│           │                          │                    │     │
│           │                          │                    │     │
│  ┌─────────────────┐       ┌─────────────────┐  ┌─────────────┐ │
│  │  Camera Input   │       │  Communication  │  │  Detection  │ │
│  │  - WebCam       │       │  - Socket.io    │  │  - PyTorch  │ │
│  │  - Video Stream │       │  - HTTP Proxy   │  │  - OpenCV   │ │
│  │  - Real-time    │       │  - Error Handle │  │  - Transform│ │
│  └─────────────────┘       └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 🏃‍♂️ Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/nameplate-detector.git
cd nameplate-detector

# Install Python dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### 2. Setup Frontend
```bash
cd frontend
npm install
cd server
npm install
```

### 3. Run the Application
```bash
# Terminal 1: Start the API server
nameplate-api

# Terminal 2: Start the backend server
cd frontend/server
npm start

# Terminal 3: Start the frontend
cd frontend
npm start
```

### 4. Access the Application
- Frontend: http://localhost:3000
- Backend: http://localhost:3001
- API: http://localhost:8000

## 🐳 Docker Quick Start

```bash
# Development
docker-compose --profile dev up

# Production
docker-compose up

# API only
docker-compose up nameplate-api
```

## 🔧 CLI Usage

The nameplate detector comes with a comprehensive CLI:

```bash
# Start API server
nameplate-detector api

# Predict nameplate in image
nameplate-detector predict image.jpg

# Show configuration
nameplate-detector config

# Run tests
nameplate-detector test --coverage
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=nameplate_detector --cov-report=html

# Run specific test categories
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only
```

## 📊 Key Features

### 🎥 Real-time Video Processing
- Live camera feed processing
- Real-time nameplate detection
- Adjustable confidence thresholds
- Frame rate optimization

### 🤖 AI-Powered Field Extraction
- 12 standard nameplate fields
- Transformer-based text extraction
- JSON structured output
- Confidence scoring

### 🌐 Web Interface
- Modern React frontend
- Real-time WebSocket communication
- Responsive design
- Professional UI/UX

### 📱 API Integration
- RESTful API endpoints
- WebSocket real-time updates
- CORS support
- OpenAPI documentation

## 🔧 Configuration

The system uses environment variables for configuration:

```bash
# Model Configuration
MODEL_PATH=models/best_nameplate_classifier.pth
CONFIDENCE_THRESHOLD=0.7
DEVICE=auto

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Frontend Configuration
FRONTEND_PORT=3000
BACKEND_PORT=3001
```

## 🚀 Deployment Options

### Local Development
- Direct Python execution
- npm development servers
- Hot reloading enabled

### Docker Containers
- Multi-stage builds
- Production optimized
- Health checks included

### Production Deployment
- Nginx reverse proxy
- Redis for caching
- SSL/TLS support
- Load balancing ready

## 📝 Documentation Standards

All documentation follows these standards:
- Clear, concise writing
- Code examples included
- Screenshots where appropriate
- API documentation with examples
- Architecture diagrams
- Troubleshooting guides

## 🤝 Contributing

See [Contributing Guidelines](developer-guide/contributing.md) for detailed information on:
- Code style and standards
- Testing requirements
- Pull request process
- Development workflow

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 🆘 Support

- 📧 Email: support@nameplate-detector.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/nameplate-detector/issues)
- 📖 Documentation: [Full Documentation](https://nameplate-detector.readthedocs.io/)
- 💬 Community: [Discord](https://discord.gg/nameplate-detector)

---

*Last updated: January 2025* 
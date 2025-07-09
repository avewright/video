# 🏭 Nameplate Detector Frontend

A React-based web application for real-time industrial nameplate detection and field extraction.

## 🚀 Quick Start

The frontend has been restored after the refactoring. Here's how to get it running:

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Install Server Dependencies

```bash
cd server
npm install
```

### 3. Start the Application

**Option A: Manual Start (Recommended)**
```bash
# Terminal 1: Start the backend server
cd frontend/server
npm start

# Terminal 2: Start the frontend
cd frontend
npm start
```

**Option B: Automatic Start**
```bash
cd frontend
node start.js
```

### 4. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:3001
- **WebSocket**: ws://localhost:3001

## 🔧 Prerequisites

1. **Model API**: Make sure your inference model is running on `localhost:8000/inference`
2. **Node.js**: Version 14 or higher
3. **npm**: Latest version
4. **Camera**: Connected camera for real-time detection

## 🎯 Features

- **Real-time Detection**: Live camera feed with nameplate detection
- **Field Extraction**: Extract technical specifications from detected nameplates
- **Confidence Thresholds**: Adjustable detection sensitivity
- **Download Images**: Save detected nameplate images
- **Socket.io Integration**: Real-time updates between frontend and backend

## 📁 Project Structure

```
frontend/
├── src/
│   ├── App.js          # Main React component
│   ├── App.css         # Styling
│   └── index.js        # React entry point
├── public/
│   └── index.html      # HTML template
├── server/
│   ├── server.js       # Express server with Socket.io
│   ├── python-bridge.js # Python integration
│   └── package.json    # Server dependencies
├── package.json        # Frontend dependencies
└── start.js           # Startup script
```

## 🔄 What Was Fixed

After the refactoring, the following files were missing and have been restored:

1. **package.json** - React dependencies and scripts
2. **public/index.html** - HTML template
3. **src/index.js** - React entry point
4. **server/package.json** - Backend dependencies

## 🛠️ Configuration

The frontend includes:
- Proxy configuration for API calls
- CORS setup for localhost development
- Socket.io client configuration
- Webcam integration with react-webcam

## 🚦 API Endpoints

- **POST /api/inference** - Field extraction from images
- **WebSocket Events**:
  - `startDetection` - Begin nameplate detection
  - `stopDetection` - Stop detection
  - `cameraFrame` - Send camera frame data
  - `adjustThreshold` - Modify confidence threshold

## 🎨 UI Components

- **Camera Feed**: Live webcam display
- **Detection Controls**: Start/stop detection buttons
- **Settings Panel**: Adjustable confidence threshold
- **Detection Stats**: Real-time statistics
- **Field Display**: Extracted nameplate information

## 📋 Usage

1. **Start Detection**: Click "Start Detection" to begin camera monitoring
2. **Point Camera**: Aim camera at industrial nameplates
3. **Automatic Detection**: System will detect nameplates automatically
4. **Extract Fields**: Click "Extract Fields" on detected nameplates
5. **View Results**: See extracted technical specifications

## 🔍 Troubleshooting

- **Port Conflicts**: Ensure ports 3000, 3001, and 8000 are available
- **Camera Access**: Allow camera permissions in browser
- **CORS Issues**: The backend proxy handles CORS for the inference API
- **Model API**: Verify your model is running on localhost:8000

## 🧪 Testing

The application connects to your existing model API for field extraction. Make sure your model is running and accessible at `localhost:8000/inference`.

## 📝 Notes

- The frontend uses a proxy configuration to avoid CORS issues
- Field extraction requires the model API to be running
- Detection uses the existing Python nameplate classifier
- Real-time updates are handled via Socket.io 
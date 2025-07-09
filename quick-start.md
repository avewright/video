# Quick Start Guide - Nameplate Detector

## Overview

This guide helps you quickly start the nameplate detection system with debugging capabilities.

## System Architecture

The system consists of 3 services:

1. **Python API Server** (Port 8000): Handles ML inference
2. **Backend Server** (Port 3001): WebSocket server for real-time communication
3. **Frontend** (Port 3000): React web interface

## Prerequisites

- Python 3.8+
- Node.js 16+
- npm

## 🔧 Quick Start with Debugging

### Step 1: Run the Debug Script

```bash
# Run the comprehensive debugging script
debug_all_services.bat
```

This will show you exactly what's working and what isn't.

### Step 2: Start Services Individually

Based on the debug output, start services one by one:

#### Start Python API Server (Required First)
```bash
python simple_api_server.py
```

#### Start Backend Server (Port 3001)
```bash
# Test the backend server
test_server.bat

# Or manually:
cd frontend/server
npm install  # if needed
node server.js
```

#### Start Frontend (Port 3000)
```bash
# Test the frontend
test_frontend.bat

# Or manually:
cd frontend
npm install  # if needed
npm start
```

## 📋 Service Status Checks

### Check if Python API is running:
```bash
curl http://localhost:8000/health
```

### Check if Backend is running:
```bash
curl http://localhost:3001/health
```

### Check if Frontend is running:
```bash
# Open in browser
http://localhost:3000
```

## 🔍 Debugging

### View Backend Server Logs
```bash
# Check the log file
type logs\server.log

# Or watch in real-time
tail -f logs\server.log
```

### Port Status
```bash
# Check which ports are in use
netstat -an | findstr ":8000\|:3001\|:3000"
```

### Common Issues

1. **"write EOF" error**: Python API server (port 8000) is not running
2. **Backend fails to start**: Check `logs\server.log` for detailed error messages
3. **Frontend crashes**: Ensure both API and backend are running first

## 🚀 Automated Startup

### Option 1: Simple Startup (Recommended)
```bash
start_simple.bat
```

### Option 2: Full Development Mode
```bash
# In 3 separate terminals:
# Terminal 1:
python simple_api_server.py

# Terminal 2:
cd frontend/server && node server.js

# Terminal 3:
cd frontend && npm start
```

## 📱 Using the Application

1. **Access the web interface**: http://localhost:3000
2. **Allow camera access** when prompted
3. **Click "Start Detection"** to begin nameplate detection
4. **Point camera at nameplate** - detection will highlight nameplates
5. **Click "Extract Fields"** to get structured data from detected nameplates

## 🔧 Development Mode

### With hot reload for frontend:
```bash
cd frontend
npm run dev
```

### With backend server:
```bash
cd frontend
npm run dev:full
```

## 📊 Health Checks

- Python API: http://localhost:8000/health
- Backend API: http://localhost:3001/health
- Frontend: http://localhost:3000

## 🆘 Troubleshooting

### Step 1: Run Debug Script
```bash
debug_all_services.bat
```

### Step 2: Check Logs
```bash
# Backend logs
type logs\server.log

# Python API logs (if any)
# Check console output where you ran python simple_api_server.py
```

### Step 3: Test Individual Services
```bash
# Test backend only
test_server.bat

# Test frontend only
test_frontend.bat
```

### Step 4: Manual Testing
```bash
# Test Python API
curl -X POST http://localhost:8000/health

# Test Backend
curl -X GET http://localhost:3001/health

# Test Frontend
# Open http://localhost:3000 in browser
```

## 🔄 Restarting Services

If you need to restart everything:

1. **Stop all services** (Ctrl+C in each terminal)
2. **Run debug script** to confirm everything is stopped
3. **Start Python API first**: `python simple_api_server.py`
4. **Start Backend**: `cd frontend/server && node server.js`
5. **Start Frontend**: `cd frontend && npm start`

## 📝 Notes

- **Start order matters**: Always start Python API first, then Backend, then Frontend
- **Logs are your friend**: Check `logs\server.log` for detailed error messages
- **Port conflicts**: If ports are busy, stop the conflicting services
- **Dependencies**: Make sure to run `npm install` in both `frontend/` and `frontend/server/`

## 🎯 Success Indicators

When everything is working:
- ✅ Python API responds to http://localhost:8000/health
- ✅ Backend server shows "Server successfully started on port 3001"
- ✅ Frontend opens in browser at http://localhost:3000
- ✅ Camera feed shows in the web interface
- ✅ "Start Detection" button is enabled 
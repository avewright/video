# 🚀 Frontend Quick Start Guide

Get your nameplate detection frontend up and running in minutes!

## 📋 Prerequisites

- **Node.js** (v16 or higher) - [Download here](https://nodejs.org/)
- **Python** (v3.8 or higher) with your trained model
- **Inference Server** running on port 8000

## ⚡ Quick Setup

### Option 1: One-Command Start (Recommended)

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Start everything**:
   ```bash
   npm run start-all
   ```

   This single command will:
   - ✅ Check all prerequisites
   - 📦 Install dependencies automatically
   - 🚀 Start the backend server (port 3001)
   - 🎨 Start the React frontend (port 3000)
   - 🔍 Verify your inference server connection
   - 🤖 Test your model file

### Option 2: Manual Setup

If you prefer manual control:

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start both services**:
   ```bash
   npm run dev
   ```

   Or start them separately in different terminals:
   ```bash
   # Terminal 1: Backend
   npm run server
   
   # Terminal 2: Frontend  
   npm start
   ```

## 🌐 Access Your Application

Once started, open your browser to:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:3001/api/health
- **WebSocket**: ws://localhost:3001

## 📱 First Time Usage

1. **Allow Camera Access** when prompted by your browser
2. **Click "Start Detection"** to begin
3. **Point camera at a nameplate** and wait for detection
4. **Click "Extract Fields"** when a nameplate is detected
5. **View results** in the sidebar

## 🔧 Configuration

### Model File Location
Your model should be at: `best_nameplate_classifier.pth` in the root directory

### Inference Server
Expected at: `http://localhost:8000/inference`

### Custom Configuration
Edit `frontend/server/server.js` to modify:
- Inference endpoint URL
- Detection parameters
- Python service integration

## 🐛 Troubleshooting

### Common Issues

**Dependencies not installing?**
- Ensure Node.js v16+ is installed
- Try deleting `node_modules` and `package-lock.json`, then run `npm install`
- Check your network connection

**Camera not working?**
- Ensure you're using HTTPS (required for camera access)
- Check browser permissions
- Try a different browser

**Backend not starting?**
- Check if port 3001 is available
- Verify Node.js version: `node --version`
- Try manual setup: `npm install` then `npm run dev`

**Python detection failing?**
- Verify model file exists at: `../best_nameplate_classifier.pth`
- Check Python environment: `python --version`
- Test with: `cd python-integration && python test_service.py`

**Field extraction not working?**
- Ensure inference server is running on port 8000
- Check network connectivity
- Verify API endpoint format with: `curl http://localhost:8000/health`

### Performance Tips

- **Lower frame rate**: Modify interval in `src/App.js` (line ~75)
- **Reduce resolution**: Adjust webcam props in `src/App.js`
- **Optimize Python**: Use GPU acceleration if available

## 📊 System Status

Check system health at: http://localhost:3001/api/health

This shows:
- ✅ Backend server status
- 🤖 Python service availability
- 📂 Model file existence
- 🔗 Inference server connection

## 📚 Next Steps

1. **Customize the UI** - Edit `src/App.js` and `src/App.css`
2. **Add features** - Enhance detection logic in `server/server.js`
3. **Deploy** - See `README.md` for deployment instructions
4. **Integrate** - Connect with your existing systems

## 🆘 Need Help?

If you encounter issues:

1. **Try manual setup**:
   ```bash
   npm install
   npm run dev
   ```

2. **Check the logs** for error messages
3. **Verify prerequisites** are installed correctly
4. **Review** the full `README.md` for detailed documentation

## 🔧 Alternative Startup Commands

```bash
# Install dependencies only
npm install

# Start backend server only
npm run server

# Start frontend only  
npm start

# Start both with concurrency
npm run dev

# Full automated startup
npm run start-all
```

---

**Ready to detect some nameplates? Let's go! 🎯** 
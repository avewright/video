@echo off
echo 🚀 Starting Simple Nameplate Detector...
echo.

echo 📍 Starting API Server (Port 8000)...
start "API Server" cmd /c "python simple_api_server.py"

echo ⏳ Waiting for API to start...
timeout /t 5 /nobreak

echo 🎨 Starting React Frontend (Port 3000)...
start "React Frontend" cmd /c "cd frontend && npm start"

echo.
echo ✅ Both services are starting...
echo 📱 Frontend: http://localhost:3000
echo 🔌 API: http://localhost:8000
echo 📋 API Health: http://localhost:8000/health
echo.
echo Press any key to exit...
pause >nul 
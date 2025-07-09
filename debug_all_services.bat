@echo off
title Debug All Services
echo 🔍 Debugging all services with comprehensive logging...
echo.

echo 📍 Current directory: %CD%
echo ⏰ Time: %date% %time%
echo 💻 System: %OS%
echo.

echo ==========================================
echo 🐍 PYTHON API SERVER STATUS
echo ==========================================
echo.

echo 🔍 Checking if Python API is running on port 8000...
netstat -an | findstr :8000
if errorlevel 1 (
    echo ❌ Python API server NOT running on port 8000
    echo 🚀 Try starting it with: python simple_api_server.py
) else (
    echo ✅ Python API server IS running on port 8000
)

echo.
echo 🔍 Testing Python API health endpoint...
curl -s http://localhost:8000/health || echo ❌ Python API health check failed

echo.
echo ==========================================
echo 🛠️ BACKEND SERVER STATUS
echo ==========================================
echo.

echo 🔍 Checking if backend server is running on port 3001...
netstat -an | findstr :3001
if errorlevel 1 (
    echo ❌ Backend server NOT running on port 3001
) else (
    echo ✅ Backend server IS running on port 3001
)

echo.
echo 📂 Checking backend server files...
if exist "frontend\server\server.js" (
    echo ✅ server.js exists
) else (
    echo ❌ server.js NOT found
    goto :end
)

if exist "frontend\server\node_modules" (
    echo ✅ node_modules exists
) else (
    echo ❌ node_modules NOT found - run 'npm install' in frontend/server
)

echo.
echo 🔧 Testing backend server startup...
echo Starting backend server for 10 seconds to check for errors...
timeout /t 3 /nobreak > nul
cd frontend\server
echo node server.js
start /min cmd /c "node server.js & timeout /t 10 /nobreak & taskkill /f /im node.exe"
timeout /t 12 /nobreak > nul
cd ..\..

echo.
echo ==========================================
echo 🎨 FRONTEND STATUS
echo ==========================================
echo.

echo 🔍 Checking if frontend is running on port 3000...
netstat -an | findstr :3000
if errorlevel 1 (
    echo ❌ Frontend NOT running on port 3000
) else (
    echo ✅ Frontend IS running on port 3000
)

echo.
echo 📂 Checking frontend files...
if exist "frontend\package.json" (
    echo ✅ package.json exists
) else (
    echo ❌ package.json NOT found
    goto :end
)

if exist "frontend\node_modules" (
    echo ✅ node_modules exists
) else (
    echo ❌ node_modules NOT found - run 'npm install' in frontend
)

echo.
echo ==========================================
echo 📋 SUMMARY
echo ==========================================
echo.

echo Services that should be running:
echo 1. Python API Server (port 8000)
echo 2. Backend Server (port 3001)
echo 3. Frontend (port 3000)
echo.

echo Current status:
netstat -an | findstr ":8000\|:3001\|:3000"

echo.
echo 📂 Log files to check:
if exist "logs\server.log" (
    echo ✅ Backend server log: logs\server.log
) else (
    echo ❌ No backend server log found
)

:end
echo.
echo 🔧 To start services manually:
echo 1. Python API: python simple_api_server.py
echo 2. Backend: cd frontend\server ^&^& node server.js
echo 3. Frontend: cd frontend ^&^& npm start
echo.
pause 
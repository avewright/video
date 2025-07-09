@echo off
title Test Server Startup
echo 🚀 Testing server startup with detailed logging...
echo.

echo 📍 Current directory: %CD%
echo ⏰ Time: %date% %time%
echo.

echo 📂 Checking directories...
if exist "frontend\server" (
    echo ✅ frontend\server directory exists
) else (
    echo ❌ frontend\server directory NOT found
    pause
    exit /b 1
)

if exist "frontend\server\server.js" (
    echo ✅ server.js file exists
) else (
    echo ❌ server.js file NOT found
    pause
    exit /b 1
)

echo.
echo 📦 Checking Node.js...
node --version
if errorlevel 1 (
    echo ❌ Node.js not found
    pause
    exit /b 1
)

echo.
echo 🔧 Starting server...
echo Press Ctrl+C to stop
echo ==========================================
echo.

cd frontend\server
node server.js

echo.
echo Server stopped or failed to start
pause 
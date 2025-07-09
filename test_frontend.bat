@echo off
title Test Frontend Startup
echo 🚀 Testing frontend startup with detailed logging...
echo.

echo 📍 Current directory: %CD%
echo ⏰ Time: %date% %time%
echo.

echo 📂 Checking directories...
if exist "frontend" (
    echo ✅ frontend directory exists
) else (
    echo ❌ frontend directory NOT found
    pause
    exit /b 1
)

if exist "frontend\package.json" (
    echo ✅ package.json file exists
) else (
    echo ❌ package.json file NOT found
    pause
    exit /b 1
)

echo.
echo 📦 Checking Node.js and npm...
node --version
npm --version
if errorlevel 1 (
    echo ❌ Node.js or npm not found
    pause
    exit /b 1
)

echo.
echo 🔧 Starting frontend...
echo Press Ctrl+C to stop
echo ==========================================
echo.

cd frontend
npm start

echo.
echo Frontend stopped or failed to start
pause 
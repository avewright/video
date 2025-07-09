@echo off
title Nameplate Detector - Full Logging
setlocal enabledelayedexpansion

:: Create logs directory
if not exist logs mkdir logs

:: Set log file with timestamp
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%" & set "YYYY=%dt:~0,4%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%" & set "Min=%dt:~10,2%" & set "Sec=%dt:~12,2%"
set "timestamp=%YYYY%-%MM%-%DD%_%HH%-%Min%-%Sec%"
set "LOGFILE=logs\startup_%timestamp%.log"

echo Starting Nameplate Detector with Full Logging > "%LOGFILE%"
echo Timestamp: %timestamp% >> "%LOGFILE%"
echo ========================================== >> "%LOGFILE%"
echo. >> "%LOGFILE%"

echo 🚀 Starting Nameplate Detector with Full Logging...
echo 📝 Log file: %LOGFILE%
echo ==========================================
echo.

:: Function to log and display
set "log_and_show=echo"

echo 📍 Current directory: %CD% | tee -a "%LOGFILE%"
echo ⏰ Time: %date% %time% | tee -a "%LOGFILE%"
echo 💻 System: %OS% | tee -a "%LOGFILE%"
echo. | tee -a "%LOGFILE%"

:: Check prerequisites
echo ==========================================
echo 🔍 CHECKING PREREQUISITES
echo ==========================================
echo.

echo 📦 Checking Node.js... | tee -a "%LOGFILE%"
node --version >> "%LOGFILE%" 2>&1
if errorlevel 1 (
    echo ❌ Node.js not found! | tee -a "%LOGFILE%"
    echo Please install Node.js from https://nodejs.org/ | tee -a "%LOGFILE%"
    goto :end
) else (
    echo ✅ Node.js found | tee -a "%LOGFILE%"
)

echo. | tee -a "%LOGFILE%"
echo 🐍 Checking Python... | tee -a "%LOGFILE%"
python --version >> "%LOGFILE%" 2>&1
if errorlevel 1 (
    echo ❌ Python not found! | tee -a "%LOGFILE%"
    echo Please install Python from https://python.org/ | tee -a "%LOGFILE%"
    goto :end
) else (
    echo ✅ Python found | tee -a "%LOGFILE%"
)

echo. | tee -a "%LOGFILE%"
echo 📂 Checking project structure... | tee -a "%LOGFILE%"
if exist "simple_api_server.py" (
    echo ✅ simple_api_server.py found | tee -a "%LOGFILE%"
) else (
    echo ❌ simple_api_server.py NOT found | tee -a "%LOGFILE%"
    goto :end
)

if exist "frontend\server\server.js" (
    echo ✅ frontend\server\server.js found | tee -a "%LOGFILE%"
) else (
    echo ❌ frontend\server\server.js NOT found | tee -a "%LOGFILE%"
    goto :end
)

if exist "frontend\package.json" (
    echo ✅ frontend\package.json found | tee -a "%LOGFILE%"
) else (
    echo ❌ frontend\package.json NOT found | tee -a "%LOGFILE%"
    goto :end
)

echo. | tee -a "%LOGFILE%"
echo 🔍 Checking for conflicting processes... | tee -a "%LOGFILE%"
netstat -ano | findstr :8000 >> "%LOGFILE%" 2>&1
if not errorlevel 1 (
    echo ⚠️  Port 8000 is already in use | tee -a "%LOGFILE%"
    echo Processes using port 8000: | tee -a "%LOGFILE%"
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do (
        tasklist /fi "PID eq %%a" >> "%LOGFILE%" 2>&1
    )
)

netstat -ano | findstr :3001 >> "%LOGFILE%" 2>&1
if not errorlevel 1 (
    echo ⚠️  Port 3001 is already in use | tee -a "%LOGFILE%"
    echo Processes using port 3001: | tee -a "%LOGFILE%"
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :3001') do (
        tasklist /fi "PID eq %%a" >> "%LOGFILE%" 2>&1
    )
)

netstat -ano | findstr :3000 >> "%LOGFILE%" 2>&1
if not errorlevel 1 (
    echo ⚠️  Port 3000 is already in use | tee -a "%LOGFILE%"
    echo Processes using port 3000: | tee -a "%LOGFILE%"
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :3000') do (
        tasklist /fi "PID eq %%a" >> "%LOGFILE%" 2>&1
    )
)

echo. | tee -a "%LOGFILE%"
echo ==========================================
echo 🐍 STARTING PYTHON API SERVER (PORT 8000)
echo ==========================================
echo.

echo 🚀 Starting Python API server... | tee -a "%LOGFILE%"
echo Command: python simple_api_server.py | tee -a "%LOGFILE%"
echo Working directory: %CD% | tee -a "%LOGFILE%"
echo. | tee -a "%LOGFILE%"

start "Python API Server" cmd /c "python simple_api_server.py > logs\python_api_%timestamp%.log 2>&1"

echo ⏳ Waiting for Python API to start... | tee -a "%LOGFILE%"
timeout /t 8 /nobreak > nul

echo 🔍 Testing Python API connection... | tee -a "%LOGFILE%"
curl -s http://localhost:8000/health >> "%LOGFILE%" 2>&1
if errorlevel 1 (
    echo ❌ Python API failed to start or not responding | tee -a "%LOGFILE%"
    echo Check logs\python_api_%timestamp%.log for details | tee -a "%LOGFILE%"
    goto :end
) else (
    echo ✅ Python API is running and responding | tee -a "%LOGFILE%"
)

echo. | tee -a "%LOGFILE%"
echo ==========================================
echo 🛠️  STARTING BACKEND SERVER (PORT 3001)
echo ==========================================
echo.

echo 🚀 Starting backend server... | tee -a "%LOGFILE%"
echo Command: cd frontend\server && node server.js | tee -a "%LOGFILE%"
echo Working directory: %CD%\frontend\server | tee -a "%LOGFILE%"
echo. | tee -a "%LOGFILE%"

cd frontend\server
start "Backend Server" cmd /c "node server.js > ..\..\logs\backend_%timestamp%.log 2>&1"
cd ..\..

echo ⏳ Waiting for backend server to start... | tee -a "%LOGFILE%"
timeout /t 8 /nobreak > nul

echo 🔍 Testing backend server connection... | tee -a "%LOGFILE%"
curl -s http://localhost:3001/health >> "%LOGFILE%" 2>&1
if errorlevel 1 (
    echo ❌ Backend server failed to start or not responding | tee -a "%LOGFILE%"
    echo Check logs\backend_%timestamp%.log for details | tee -a "%LOGFILE%"
    goto :end
) else (
    echo ✅ Backend server is running and responding | tee -a "%LOGFILE%"
)

echo. | tee -a "%LOGFILE%"
echo ==========================================
echo 🎨 STARTING FRONTEND (PORT 3000)
echo ==========================================
echo.

echo 🚀 Starting React frontend... | tee -a "%LOGFILE%"
echo Command: cd frontend && npm start | tee -a "%LOGFILE%"
echo Working directory: %CD%\frontend | tee -a "%LOGFILE%"
echo. | tee -a "%LOGFILE%"

cd frontend
start "React Frontend" cmd /c "npm start > ..\logs\frontend_%timestamp%.log 2>&1"
cd ..

echo ⏳ Waiting for frontend to start... | tee -a "%LOGFILE%"
timeout /t 10 /nobreak > nul

echo 🔍 Testing frontend connection... | tee -a "%LOGFILE%"
curl -s http://localhost:3000 >> "%LOGFILE%" 2>&1
if errorlevel 1 (
    echo ⚠️  Frontend may still be starting (this is normal) | tee -a "%LOGFILE%"
    echo Check logs\frontend_%timestamp%.log for details | tee -a "%LOGFILE%"
) else (
    echo ✅ Frontend is running and responding | tee -a "%LOGFILE%"
)

echo. | tee -a "%LOGFILE%"
echo ==========================================
echo 📊 FINAL STATUS CHECK
echo ==========================================
echo.

echo 🔍 Final port status: | tee -a "%LOGFILE%"
netstat -an | findstr ":8000\|:3001\|:3000" >> "%LOGFILE%"
netstat -an | findstr ":8000\|:3001\|:3000"

echo. | tee -a "%LOGFILE%"
echo ✅ All services should now be running!
echo 📱 Frontend: http://localhost:3000
echo 🔌 Backend: http://localhost:3001
echo 🐍 Python API: http://localhost:8000
echo.
echo 📝 Log files created:
echo    - Main log: %LOGFILE%
echo    - Python API: logs\python_api_%timestamp%.log
echo    - Backend: logs\backend_%timestamp%.log
echo    - Frontend: logs\frontend_%timestamp%.log
echo.
echo 🔄 To stop all services, close the command windows or press Ctrl+C
echo.

:end
echo. | tee -a "%LOGFILE%"
echo Script completed at %time% | tee -a "%LOGFILE%"
echo ========================================== | tee -a "%LOGFILE%"
echo.
echo 📋 If services failed to start, check the individual log files above
echo    for detailed error messages.
echo.
pause 
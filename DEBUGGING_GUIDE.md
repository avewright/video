# Debugging Guide - Capture Service Failures

## Overview

This guide explains how to use the comprehensive logging tools to capture exactly why services fail when they quit unexpectedly.

## 🔧 Available Debugging Tools

### 1. **start_with_logging.ps1** - PowerShell Full Startup with Logging
- **Purpose**: Start all services with comprehensive logging
- **Usage**: `powershell -ExecutionPolicy Bypass -File start_with_logging.ps1`
- **What it does**:
  - Checks all prerequisites (Node.js, Python, files)
  - Detects port conflicts
  - Starts services in correct order
  - Tests each service after starting
  - Creates separate log files for each service

### 2. **start_with_logging.bat** - Windows Batch Full Startup with Logging
- **Purpose**: Batch version of comprehensive startup
- **Usage**: `start_with_logging.bat`
- **What it does**:
  - Same as PowerShell version but using batch commands
  - Works without PowerShell execution policy issues

### 3. **monitor_services.ps1** - Real-time Service Monitor
- **Purpose**: Monitor running services and detect crashes
- **Usage**: `powershell -ExecutionPolicy Bypass -File monitor_services.ps1`
- **What it does**:
  - Continuously monitors all three services
  - Detects when services crash or stop
  - Logs crash details including PID, memory usage, etc.
  - Provides health checks for running services

### 4. **debug_all_services.bat** - Quick Status Check
- **Purpose**: Quick diagnostic of current service status
- **Usage**: `debug_all_services.bat`
- **What it does**:
  - Shows which services are running/not running
  - Checks for port conflicts
  - Provides step-by-step fix instructions

## 📊 How to Capture Service Failures

### Step 1: Start Services with Full Logging

```powershell
# Using PowerShell (recommended)
powershell -ExecutionPolicy Bypass -File start_with_logging.ps1

# OR using Batch
start_with_logging.bat
```

This will create log files in the `logs/` directory:
- `startup_YYYY-MM-DD_HH-mm-ss.log` - Main startup log
- `python_api_YYYY-MM-DD_HH-mm-ss.log` - Python API output
- `backend_YYYY-MM-DD_HH-mm-ss.log` - Backend server output
- `frontend_YYYY-MM-DD_HH-mm-ss.log` - Frontend output

### Step 2: Monitor for Crashes (Optional)

In a separate PowerShell window:
```powershell
powershell -ExecutionPolicy Bypass -File monitor_services.ps1
```

This will continuously watch for service crashes and log:
- Exact time of crash
- Process ID that crashed
- Memory usage before crash
- Related log files to check

### Step 3: When a Service Fails

If a service fails to start or crashes:

1. **Check the main startup log** first:
   ```
   type logs\startup_YYYY-MM-DD_HH-mm-ss.log
   ```

2. **Check the specific service log**:
   ```
   # For Python API issues
   type logs\python_api_YYYY-MM-DD_HH-mm-ss.log
   
   # For Backend server issues
   type logs\backend_YYYY-MM-DD_HH-mm-ss.log
   
   # For Frontend issues
   type logs\frontend_YYYY-MM-DD_HH-mm-ss.log
   ```

3. **Check the enhanced server log** (if backend server started):
   ```
   type logs\server.log
   ```

## 🎯 What to Look For in Logs

### Common Error Patterns

1. **"MODULE_NOT_FOUND" errors**:
   - Service is starting from wrong directory
   - Missing dependencies (`npm install` needed)

2. **"EADDRINUSE" errors**:
   - Port already in use by another process
   - Previous service instance still running

3. **"ENOENT" errors**:
   - Missing files or directories
   - Incorrect file paths

4. **Connection refused errors**:
   - Dependency service not running (e.g., backend needs Python API)
   - Service started but not responding

### Log File Locations

All logs are stored in the `logs/` directory:
```
logs/
├── startup_2025-01-15_14-30-00.log      # Main startup log
├── python_api_2025-01-15_14-30-00.log   # Python API output
├── backend_2025-01-15_14-30-00.log      # Backend server output
├── frontend_2025-01-15_14-30-00.log     # Frontend output
├── server.log                           # Enhanced backend server log
└── monitor_2025-01-15_14-30-00.log      # Service monitor log
```

## 🚀 Quick Troubleshooting Workflow

1. **Run quick diagnostic**:
   ```
   debug_all_services.bat
   ```

2. **If issues found, start with full logging**:
   ```
   powershell -ExecutionPolicy Bypass -File start_with_logging.ps1
   ```

3. **If services start but crash later, run monitor**:
   ```
   powershell -ExecutionPolicy Bypass -File monitor_services.ps1
   ```

4. **Check the relevant log files** for exact error messages

5. **Share the log files** with support/developers for analysis

## 📋 Example Log Analysis

### Successful Startup Log
```
[2025-01-15 14:30:00] [INFO] 🚀 Starting Nameplate Detector with Full Logging
[2025-01-15 14:30:01] [SUCCESS] ✅ Node.js found: v18.17.0
[2025-01-15 14:30:02] [SUCCESS] ✅ Python found: Python 3.11.0
[2025-01-15 14:30:03] [SUCCESS] ✅ Python API started successfully
[2025-01-15 14:30:04] [SUCCESS] ✅ Backend server started successfully
[2025-01-15 14:30:05] [SUCCESS] ✅ Frontend started successfully
```

### Failed Startup Log
```
[2025-01-15 14:30:00] [INFO] 🚀 Starting Nameplate Detector with Full Logging
[2025-01-15 14:30:01] [SUCCESS] ✅ Node.js found: v18.17.0
[2025-01-15 14:30:02] [SUCCESS] ✅ Python found: Python 3.11.0
[2025-01-15 14:30:03] [SUCCESS] ✅ Python API started successfully
[2025-01-15 14:30:04] [ERROR] ❌ Backend server failed to start. Check logs\backend_2025-01-15_14-30-00.log for details
[2025-01-15 14:30:04] [INFO] Last few lines of backend log:
[2025-01-15 14:30:04] [INFO]    Error: listen EADDRINUSE: address already in use :::3001
```

### Crash Detection Log
```
[2025-01-15 14:45:00] [CRITICAL] 💥 Backend Server CRASHED! (Crash #1)
[2025-01-15 14:45:00] [INFO]    Previous PID: 15428
[2025-01-15 14:45:00] [INFO]    Previous Status: Running
[2025-01-15 14:45:00] [INFO]    Current Status: Not Running
[2025-01-15 14:45:00] [INFO]    Process info: node - CPU: 0.5 - Memory: 58720256
```

## 💡 Tips for Effective Debugging

1. **Always start with the main startup log** - it shows the overall flow
2. **Check timestamps** - they help identify when exactly things went wrong
3. **Look for the first ERROR message** - subsequent errors are often consequences
4. **Check port conflicts** - very common cause of startup failures
5. **Monitor resource usage** - crashes often preceded by high memory/CPU usage

## 🔄 Clean Up Old Logs

To clean up old log files:
```powershell
# Remove logs older than 7 days
Get-ChildItem logs\* -Recurse | Where-Object {($_.LastWriteTime -lt (Get-Date).AddDays(-7))} | Remove-Item
```

This comprehensive logging system will capture exactly what happens when services fail, making it much easier to diagnose and fix issues. 
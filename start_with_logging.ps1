# Nameplate Detector - Full Logging PowerShell Script
param([switch]$Verbose)

# Create logs directory
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

# Set log file with timestamp
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$logFile = "logs\startup_$timestamp.log"

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestampedMessage = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] [$Level] $Message"
    Write-Host $timestampedMessage
    Add-Content -Path $logFile -Value $timestampedMessage
}

function Test-Port {
    param([int]$Port)
    try {
        $listener = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Any, $Port)
        $listener.Start()
        $listener.Stop()
        return $false  # Port is free
    } catch {
        return $true   # Port is in use
    }
}

function Test-Service {
    param([string]$Url, [string]$ServiceName)
    try {
        $response = Invoke-WebRequest -Uri $Url -Method GET -TimeoutSec 5 -UseBasicParsing
        Write-Log "✅ $ServiceName is responding (Status: $($response.StatusCode))" "SUCCESS"
        return $true
    } catch {
        Write-Log "❌ $ServiceName not responding: $($_.Exception.Message)" "ERROR"
        return $false
    }
}

function Kill-ProcessOnPort {
    param([int]$Port)
    try {
        $processes = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
        foreach ($process in $processes) {
            $pid = $process.OwningProcess
            Write-Log "Killing process $pid on port $Port" "INFO"
            Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
        }
    } catch {
        Write-Log "Could not kill processes on port $Port - $($_.Exception.Message)" "WARNING"
    }
}

# Start logging
Write-Log "Starting Nameplate Detector with Full Logging" "INFO"
Write-Log "Log file: $logFile" "INFO"
Write-Log "Current directory: $PWD" "INFO"
Write-Log "System: $env:OS" "INFO"
Write-Log "PowerShell version: $($PSVersionTable.PSVersion)" "INFO"

# Check prerequisites
Write-Log "==========================================" "INFO"
Write-Log "CHECKING PREREQUISITES" "INFO"
Write-Log "==========================================" "INFO"

# Check Node.js
Write-Log "Checking Node.js..." "INFO"
try {
    $nodeVersion = node --version
    Write-Log "SUCCESS - Node.js found: $nodeVersion" "SUCCESS"
} catch {
    Write-Log "ERROR - Node.js not found! Please install from https://nodejs.org/" "ERROR"
    Write-Log "Script terminated due to missing Node.js" "ERROR"
    exit 1
}

# Check Python
Write-Log "Checking Python..." "INFO"
try {
    $pythonVersion = python --version
    Write-Log "SUCCESS - Python found: $pythonVersion" "SUCCESS"
} catch {
    Write-Log "ERROR - Python not found! Please install from https://python.org/" "ERROR"
    Write-Log "Script terminated due to missing Python" "ERROR"
    exit 1
}

# Check project structure
Write-Log "Checking project structure..." "INFO"
$requiredFiles = @(
    "simple_api_server.py",
    "frontend\server\server.js",
    "frontend\package.json"
)

foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Log "SUCCESS - $file found" "SUCCESS"
    } else {
        Write-Log "ERROR - $file NOT found" "ERROR"
        Write-Log "Script terminated due to missing file: $file" "ERROR"
        exit 1
    }
}

# Check for conflicting processes
Write-Log "Checking for conflicting processes..." "INFO"
$ports = @(8000, 3001, 3000)
foreach ($port in $ports) {
    if (Test-Port $port) {
        Write-Log "WARNING - Port $port is already in use" "WARNING"
        try {
            $processes = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
            foreach ($process in $processes) {
                $processInfo = Get-Process -Id $process.OwningProcess -ErrorAction SilentlyContinue
                if ($processInfo) {
                    Write-Log "   Process using port $port - $($processInfo.ProcessName) (PID - $($processInfo.Id))" "WARNING"
                }
            }
        } catch {
            Write-Log "   Could not identify process using port $port" "WARNING"
        }
    } else {
        Write-Log "SUCCESS - Port $port is free" "SUCCESS"
    }
}

# Start Python API Server
Write-Log "==========================================" "INFO"
Write-Log "STARTING PYTHON API SERVER (PORT 8000)" "INFO"
Write-Log "==========================================" "INFO"

Write-Log "Starting Python API server..." "INFO"
Write-Log "Command: python simple_api_server.py" "INFO"
Write-Log "Working directory: $PWD" "INFO"

$pythonLogFile = "logs\python_api_$timestamp.log"
$pythonProcess = Start-Process -FilePath "python" -ArgumentList "simple_api_server.py" -RedirectStandardOutput $pythonLogFile -RedirectStandardError $pythonLogFile -PassThru -NoNewWindow

Write-Log "Waiting for Python API to start..." "INFO"
Start-Sleep -Seconds 8

Write-Log "Testing Python API connection..." "INFO"
if (Test-Service "http://localhost:8000/health" "Python API") {
    Write-Log "SUCCESS - Python API started successfully" "SUCCESS"
} else {
    Write-Log "ERROR - Python API failed to start. Check $pythonLogFile for details" "ERROR"
    Write-Log "Python API process still running: $($pythonProcess.HasExited -eq $false)" "INFO"
    if (Test-Path $pythonLogFile) {
        Write-Log "Last few lines of Python API log:" "INFO"
        Get-Content $pythonLogFile -Tail 10 | ForEach-Object { Write-Log "   $_" "INFO" }
    }
    exit 1
}

# Start Backend Server
Write-Log "==========================================" "INFO"
Write-Log "STARTING BACKEND SERVER (PORT 3001)" "INFO"
Write-Log "==========================================" "INFO"

Write-Log "Starting backend server..." "INFO"
Write-Log "Command: node server.js" "INFO"
Write-Log "Working directory: $PWD\frontend\server" "INFO"

$backendLogFile = "logs\backend_$timestamp.log"
Push-Location "frontend\server"
$backendProcess = Start-Process -FilePath "node" -ArgumentList "server.js" -RedirectStandardOutput "..\..\$backendLogFile" -RedirectStandardError "..\..\$backendLogFile" -PassThru -NoNewWindow
Pop-Location

Write-Log "Waiting for backend server to start..." "INFO"
Start-Sleep -Seconds 8

Write-Log "Testing backend server connection..." "INFO"
if (Test-Service "http://localhost:3001/health" "Backend Server") {
    Write-Log "SUCCESS - Backend server started successfully" "SUCCESS"
} else {
    Write-Log "ERROR - Backend server failed to start. Check $backendLogFile for details" "ERROR"
    Write-Log "Backend process still running: $($backendProcess.HasExited -eq $false)" "INFO"
    if (Test-Path $backendLogFile) {
        Write-Log "Last few lines of backend log:" "INFO"
        Get-Content $backendLogFile -Tail 10 | ForEach-Object { Write-Log "   $_" "INFO" }
    }
    exit 1
}

# Start Frontend
Write-Log "==========================================" "INFO"
Write-Log "STARTING FRONTEND (PORT 3000)" "INFO"
Write-Log "==========================================" "INFO"

Write-Log "Starting React frontend..." "INFO"
Write-Log "Command: npm start" "INFO"
Write-Log "Working directory: $PWD\frontend" "INFO"

$frontendLogFile = "logs\frontend_$timestamp.log"
Push-Location "frontend"
$frontendProcess = Start-Process -FilePath "npm" -ArgumentList "start" -RedirectStandardOutput "..\$frontendLogFile" -RedirectStandardError "..\$frontendLogFile" -PassThru -NoNewWindow
Pop-Location

Write-Log "Waiting for frontend to start..." "INFO"
Start-Sleep -Seconds 15

Write-Log "Testing frontend connection..." "INFO"
if (Test-Service "http://localhost:3000" "Frontend") {
    Write-Log "SUCCESS - Frontend started successfully" "SUCCESS"
} else {
    Write-Log "WARNING - Frontend may still be starting (this is normal)" "WARNING"
    Write-Log "Check $frontendLogFile for details" "INFO"
    if (Test-Path $frontendLogFile) {
        Write-Log "Last few lines of frontend log:" "INFO"
        Get-Content $frontendLogFile -Tail 10 | ForEach-Object { Write-Log "   $_" "INFO" }
    }
}

# Final status check
Write-Log "==========================================" "INFO"
Write-Log "FINAL STATUS CHECK" "INFO"
Write-Log "==========================================" "INFO"

Write-Log "Final port status:" "INFO"
$finalPorts = Get-NetTCPConnection | Where-Object { $_.LocalPort -in @(8000, 3001, 3000) } | Select-Object LocalPort, State, OwningProcess
foreach ($port in $finalPorts) {
    Write-Log "   Port $($port.LocalPort) - $($port.State) (PID - $($port.OwningProcess))" "INFO"
}

Write-Log "SUCCESS - All services should now be running!" "SUCCESS"
Write-Log "Frontend - http://localhost:3000" "INFO"
Write-Log "Backend - http://localhost:3001" "INFO"
Write-Log "Python API - http://localhost:8000" "INFO"
Write-Log "Log files created:" "INFO"
Write-Log "   - Main log: $logFile" "INFO"
Write-Log "   - Python API: $pythonLogFile" "INFO"
Write-Log "   - Backend: $backendLogFile" "INFO"
Write-Log "   - Frontend: $frontendLogFile" "INFO"

Write-Log "To stop all services, close the process windows or use Ctrl+C" "INFO"
Write-Log "Script completed successfully" "SUCCESS"

Write-Host "`nPress any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 
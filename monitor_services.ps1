# Service Monitor - Watches for crashes and logs detailed information
param([int]$CheckInterval = 10)

# Create logs directory
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$monitorLog = "logs\monitor_$timestamp.log"

function Write-MonitorLog {
    param([string]$Message, [string]$Level = "INFO")
    $timestampedMessage = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] [$Level] $Message"
    Write-Host $timestampedMessage
    Add-Content -Path $monitorLog -Value $timestampedMessage
}

function Get-ServiceStatus {
    $services = @{}
    
    # Check Python API (port 8000)
    try {
        $pythonProc = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
        if ($pythonProc) {
            $services["Python API"] = @{
                Port = 8000
                PID = $pythonProc.OwningProcess
                Status = "Running"
                Process = Get-Process -Id $pythonProc.OwningProcess -ErrorAction SilentlyContinue
            }
        } else {
            $services["Python API"] = @{
                Port = 8000
                PID = $null
                Status = "Not Running"
                Process = $null
            }
        }
    } catch {
        $services["Python API"] = @{
            Port = 8000
            PID = $null
            Status = "Error"
            Process = $null
            Error = $_.Exception.Message
        }
    }
    
    # Check Backend Server (port 3001)
    try {
        $backendProc = Get-NetTCPConnection -LocalPort 3001 -ErrorAction SilentlyContinue
        if ($backendProc) {
            $services["Backend Server"] = @{
                Port = 3001
                PID = $backendProc.OwningProcess
                Status = "Running"
                Process = Get-Process -Id $backendProc.OwningProcess -ErrorAction SilentlyContinue
            }
        } else {
            $services["Backend Server"] = @{
                Port = 3001
                PID = $null
                Status = "Not Running"
                Process = $null
            }
        }
    } catch {
        $services["Backend Server"] = @{
            Port = 3001
            PID = $null
            Status = "Error"
            Process = $null
            Error = $_.Exception.Message
        }
    }
    
    # Check Frontend (port 3000)
    try {
        $frontendProc = Get-NetTCPConnection -LocalPort 3000 -ErrorAction SilentlyContinue
        if ($frontendProc) {
            $services["Frontend"] = @{
                Port = 3000
                PID = $frontendProc.OwningProcess
                Status = "Running"
                Process = Get-Process -Id $frontendProc.OwningProcess -ErrorAction SilentlyContinue
            }
        } else {
            $services["Frontend"] = @{
                Port = 3000
                PID = $null
                Status = "Not Running"
                Process = $null
            }
        }
    } catch {
        $services["Frontend"] = @{
            Port = 3000
            PID = $null
            Status = "Error"
            Process = $null
            Error = $_.Exception.Message
        }
    }
    
    return $services
}

function Test-ServiceHealth {
    param([string]$ServiceName, [string]$Url)
    try {
        $response = Invoke-WebRequest -Uri $Url -Method GET -TimeoutSec 3 -UseBasicParsing
        return $true
    } catch {
        Write-MonitorLog "❌ $ServiceName health check failed: $($_.Exception.Message)" "ERROR"
        return $false
    }
}

# Start monitoring
Write-MonitorLog "🔍 Starting service monitor" "INFO"
Write-MonitorLog "📝 Monitor log: $monitorLog" "INFO"
Write-MonitorLog "🔄 Check interval: $CheckInterval seconds" "INFO"
Write-MonitorLog "==========================================", "INFO"

$previousStatus = @{}
$crashCount = @{}

try {
    while ($true) {
        $currentStatus = Get-ServiceStatus
        
        foreach ($serviceName in $currentStatus.Keys) {
            $current = $currentStatus[$serviceName]
            $previous = $previousStatus[$serviceName]
            
            # Check for status changes
            if ($previous -and $previous.Status -ne $current.Status) {
                if ($current.Status -eq "Not Running" -and $previous.Status -eq "Running") {
                    # Service crashed or stopped
                    if (-not $crashCount.ContainsKey($serviceName)) {
                        $crashCount[$serviceName] = 0
                    }
                    $crashCount[$serviceName]++
                    
                    Write-MonitorLog "💥 $serviceName CRASHED! (Crash #$($crashCount[$serviceName]))" "CRITICAL"
                    Write-MonitorLog "   Previous PID: $($previous.PID)" "INFO"
                    Write-MonitorLog "   Previous Status: $($previous.Status)" "INFO"
                    Write-MonitorLog "   Current Status: $($current.Status)" "INFO"
                    
                    # Get crash details
                    if ($previous.Process) {
                        Write-MonitorLog "   Process info: $($previous.Process.ProcessName) - CPU: $($previous.Process.CPU) - Memory: $($previous.Process.WorkingSet64)" "INFO"
                    }
                    
                    # Check for related log files
                    $logPattern = "logs\*$timestamp*"
                    $logFiles = Get-ChildItem -Path $logPattern -ErrorAction SilentlyContinue
                    if ($logFiles) {
                        Write-MonitorLog "   Related log files to check:" "INFO"
                        foreach ($log in $logFiles) {
                            Write-MonitorLog "     - $($log.Name)" "INFO"
                        }
                    }
                    
                } elseif ($current.Status -eq "Running" -and $previous.Status -eq "Not Running") {
                    # Service restarted
                    Write-MonitorLog "✅ $serviceName RESTARTED (PID: $($current.PID))" "SUCCESS"
                }
            }
            
            # Health checks for running services
            if ($current.Status -eq "Running") {
                $healthy = $false
                switch ($serviceName) {
                    "Python API" { $healthy = Test-ServiceHealth $serviceName "http://localhost:8000/health" }
                    "Backend Server" { $healthy = Test-ServiceHealth $serviceName "http://localhost:3001/health" }
                    "Frontend" { $healthy = Test-ServiceHealth $serviceName "http://localhost:3000" }
                }
                
                if (-not $healthy) {
                    Write-MonitorLog "⚠️  $serviceName is running but not responding to health checks" "WARNING"
                }
            }
        }
        
        # Store current status for next iteration
        $previousStatus = $currentStatus
        
        # Log periodic status (every 10 checks)
        if ((Get-Date).Second % 60 -eq 0) {
            Write-MonitorLog "📊 Status check:" "INFO"
            foreach ($serviceName in $currentStatus.Keys) {
                $service = $currentStatus[$serviceName]
                Write-MonitorLog "   $serviceName (Port $($service.Port)): $($service.Status)" "INFO"
            }
        }
        
        Start-Sleep -Seconds $CheckInterval
    }
} catch {
    Write-MonitorLog "❌ Monitor crashed: $($_.Exception.Message)" "CRITICAL"
    Write-MonitorLog "Stack trace: $($_.Exception.StackTrace)" "ERROR"
} finally {
    Write-MonitorLog "🔄 Service monitor stopped" "INFO"
} 
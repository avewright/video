#!/bin/bash

# Nameplate Detector - Stop All Services
# This script stops all running services cleanly

set -e

echo "🛑 Stopping Nameplate Detector Services..."
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to kill process if running
kill_process() {
    local pid=$1
    local name=$2
    
    if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
        echo -e "${BLUE}🛑 Stopping $name (PID: $pid)...${NC}"
        kill $pid
        sleep 2
        
        # Force kill if still running
        if kill -0 $pid 2>/dev/null; then
            echo -e "${YELLOW}⚠️  Force stopping $name...${NC}"
            kill -9 $pid 2>/dev/null || true
        fi
        
        echo -e "${GREEN}✅ $name stopped${NC}"
    else
        echo -e "${YELLOW}⚠️  $name was not running${NC}"
    fi
}

# Read PIDs from file if it exists
if [ -f ".service_pids" ]; then
    source .service_pids
    
    kill_process "$FRONTEND_PID" "Frontend"
    kill_process "$BACKEND_PID" "Backend Server"
    kill_process "$API_PID" "API Server"
    
    # Remove PID file
    rm -f .service_pids
else
    echo -e "${YELLOW}⚠️  No PID file found. Attempting to find and stop services...${NC}"
    
    # Try to find processes by port
    API_PID=$(lsof -ti:8000 2>/dev/null || echo "")
    BACKEND_PID=$(lsof -ti:3001 2>/dev/null || echo "")
    FRONTEND_PID=$(lsof -ti:3000 2>/dev/null || echo "")
    
    kill_process "$API_PID" "API Server (Port 8000)"
    kill_process "$BACKEND_PID" "Backend Server (Port 3001)"
    kill_process "$FRONTEND_PID" "Frontend (Port 3000)"
fi

# Also try to kill any remaining nameplate processes
echo -e "${BLUE}🔍 Checking for remaining nameplate processes...${NC}"
REMAINING_PIDS=$(pgrep -f "nameplate" 2>/dev/null || echo "")
if [ -n "$REMAINING_PIDS" ]; then
    echo -e "${YELLOW}⚠️  Found remaining nameplate processes: $REMAINING_PIDS${NC}"
    echo "$REMAINING_PIDS" | xargs kill 2>/dev/null || true
fi

# Clean up any remaining node processes on our ports
for port in 3000 3001; do
    PID=$(lsof -ti:$port 2>/dev/null || echo "")
    if [ -n "$PID" ]; then
        echo -e "${YELLOW}⚠️  Cleaning up remaining process on port $port (PID: $PID)${NC}"
        kill $PID 2>/dev/null || true
    fi
done

echo -e "${GREEN}🎉 All services stopped successfully!${NC}"
echo ""
echo -e "${BLUE}📝 Service logs are still available in logs/directory:${NC}"
echo "  logs/api.log"
echo "  logs/backend.log"
echo "  logs/frontend.log" 
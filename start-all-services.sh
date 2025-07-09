#!/bin/bash

# Nameplate Detector - Start All Services
# This script starts all services in the correct order

set -e

echo "🚀 Starting Nameplate Detector Services..."
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        echo -e "${YELLOW}Warning: Port $port is already in use${NC}"
        return 1
    fi
    return 0
}

# Function to wait for service to be ready
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1
    
    echo -e "${BLUE}Waiting for $service_name to be ready...${NC}"
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s $url > /dev/null 2>&1; then
            echo -e "${GREEN}✅ $service_name is ready!${NC}"
            return 0
        fi
        echo -e "${YELLOW}⏳ Attempt $attempt/$max_attempts - waiting for $service_name...${NC}"
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo -e "${RED}❌ $service_name failed to start within timeout${NC}"
    return 1
}

# Check if virtual environment exists
if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
    echo -e "${YELLOW}⚠️  No virtual environment found. Creating one...${NC}"
    python -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
fi

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Install package if needed
if ! python -c "import nameplate_detector" 2>/dev/null; then
    echo -e "${YELLOW}📦 Installing nameplate detector package...${NC}"
    pip install -e .
    echo -e "${GREEN}✅ Package installed${NC}"
fi

# Check ports
echo -e "${BLUE}🔍 Checking ports...${NC}"
check_port 8000 || echo -e "${YELLOW}Will try to start API server anyway${NC}"
check_port 3001 || echo -e "${YELLOW}Will try to start backend server anyway${NC}"
check_port 3000 || echo -e "${YELLOW}Will try to start frontend anyway${NC}"

# Create logs directory
mkdir -p logs

echo -e "${BLUE}🔧 Starting API Server (Port 8000)...${NC}"
# Start API server in background
nohup nameplate-detector api --host 0.0.0.0 --port 8000 > logs/api.log 2>&1 &
API_PID=$!
echo "API Server PID: $API_PID"

# Wait for API to be ready
if wait_for_service "http://localhost:8000/health" "API Server"; then
    echo -e "${GREEN}✅ API Server started successfully${NC}"
else
    echo -e "${RED}❌ Failed to start API Server${NC}"
    kill $API_PID 2>/dev/null || true
    exit 1
fi

echo -e "${BLUE}🔧 Starting Backend Server (Port 3001)...${NC}"
# Start backend server in background
cd frontend/server
nohup npm start > ../../logs/backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend Server PID: $BACKEND_PID"
cd ../..

# Wait for backend to be ready
if wait_for_service "http://localhost:3001" "Backend Server"; then
    echo -e "${GREEN}✅ Backend Server started successfully${NC}"
else
    echo -e "${RED}❌ Failed to start Backend Server${NC}"
    kill $API_PID $BACKEND_PID 2>/dev/null || true
    exit 1
fi

echo -e "${BLUE}🔧 Starting Frontend (Port 3000)...${NC}"
# Start frontend in background
cd frontend
nohup npm start > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"
cd ..

# Wait a bit for frontend to start
sleep 5

echo -e "${GREEN}🎉 All services started successfully!${NC}"
echo "========================================"
echo -e "${BLUE}📱 Frontend:${NC} http://localhost:3000"
echo -e "${BLUE}🔧 Backend:${NC} http://localhost:3001"
echo -e "${BLUE}🔌 API:${NC} http://localhost:8000"
echo -e "${BLUE}📊 API Health:${NC} http://localhost:8000/health"
echo -e "${BLUE}📋 API Docs:${NC} http://localhost:8000/docs"
echo ""
echo -e "${YELLOW}📝 Logs are in the logs/ directory${NC}"
echo -e "${YELLOW}🛑 To stop all services, run: ./stop-all-services.sh${NC}"
echo ""

# Save PIDs for stopping later
cat > .service_pids << EOF
API_PID=$API_PID
BACKEND_PID=$BACKEND_PID
FRONTEND_PID=$FRONTEND_PID
EOF

echo -e "${GREEN}✅ Services are running in the background${NC}"
echo -e "${BLUE}🔍 Monitor logs with:${NC}"
echo "  tail -f logs/api.log"
echo "  tail -f logs/backend.log"  
echo "  tail -f logs/frontend.log" 
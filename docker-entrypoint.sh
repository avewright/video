#!/bin/bash
set -e

# Function to start API service
start_api() {
    echo "Starting Nameplate Detection API..."
    exec nameplate-api
}

# Function to start frontend backend
start_backend() {
    echo "Starting Frontend Backend..."
    cd /app/frontend/server
    exec node server.js
}

# Function to start full application
start_full() {
    echo "Starting Full Application..."
    
    # Start API in background
    nameplate-api &
    API_PID=$!
    
    # Start frontend backend
    cd /app/frontend/server
    node server.js &
    BACKEND_PID=$!
    
    # Wait for both services
    wait $API_PID $BACKEND_PID
}

# Function to run tests
run_tests() {
    echo "Running tests..."
    exec pytest tests/ -v
}

# Function to run development server
start_dev() {
    echo "Starting Development Server..."
    exec python -m nameplate_detector.api.server --reload
}

# Main command handler
case "${1:-api}" in
    api)
        start_api
        ;;
    backend)
        start_backend
        ;;
    full)
        start_full
        ;;
    test)
        run_tests
        ;;
    dev)
        start_dev
        ;;
    *)
        echo "Usage: $0 {api|backend|full|test|dev}"
        echo "  api     - Start API service only"
        echo "  backend - Start frontend backend only"
        echo "  full    - Start full application stack"
        echo "  test    - Run tests"
        echo "  dev     - Start development server"
        exit 1
        ;;
esac 
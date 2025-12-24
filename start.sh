#!/bin/bash

# RAG Chatbot Startup Script
# This script starts both the backend and frontend servers

set -e

echo "Starting RAG Chatbot..."
echo ""

# Check if dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
    echo "Installing frontend dependencies..."
    cd frontend && npm install && cd ..
fi

if [ ! -d "backend/.venv" ]; then
    echo "Setting up Python virtual environment..."
    cd backend
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    cd ..
else
    source backend/.venv/bin/activate
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start backend
echo "Starting backend on http://localhost:8000..."
cd backend
source .venv/bin/activate
uvicorn main:app --reload --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 2

# Start frontend
echo "Starting frontend on http://localhost:3000..."
cd frontend
npm run dev -- --hostname 127.0.0.1 --port 3000 &
FRONTEND_PID=$!
cd ..

echo ""
echo "✓ Backend running at http://localhost:8000"
echo "✓ Frontend running at http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Wait for any background process to exit
wait

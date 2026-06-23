#!/bin/bash

# Exit on error
set -e

echo "=========================================================="
echo "      Starting TrafficVioLens Auto-Setup & Runner         "
echo "=========================================================="

# Check for Tesseract OCR
if ! command -v tesseract &> /dev/null; then
    echo "⚠️  WARNING: Tesseract OCR is not installed or not in PATH."
    echo "   License Plate Recognition will fail."
    echo "   Please install it: "
    echo "   - Ubuntu/Debian: sudo apt install tesseract-ocr tesseract-ocr-eng"
    echo "   - macOS: brew install tesseract"
    echo ""
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Set up Python virtual environment
echo "📦 Setting up Python virtual environment..."
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cd ..

# Set up Node dependencies
echo "📦 Installing Node dependencies..."
cd frontend
npm install
cd ..

# Start Backend in the background
echo "🚀 Starting backend server on http://localhost:8000..."
cd backend
source .venv/bin/activate
python -m uvicorn main:app --port 8000 &
BACKEND_PID=$!
cd ..

# Trap Ctrl+C to kill the backend when exiting
cleanup() {
    echo "Stopping servers..."
    kill $BACKEND_PID 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Start Frontend in the foreground
echo "🚀 Starting React frontend server on http://localhost:5173..."
cd frontend
npm run dev

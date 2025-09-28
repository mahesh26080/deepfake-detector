#!/bin/bash

# Deepfake Detection AI - Quick Start Script
echo "🚀 Starting Deepfake Detection AI"
echo "================================="

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3 and try again"
    exit 1
fi

echo "✅ Using Python: $(python3 --version)"

# Install requirements if needed
echo "📦 Checking dependencies..."
python3 -c "import tensorflow, flask, cv2" 2>/dev/null || {
    echo "📥 Installing requirements..."
    pip3 install -r requirements.txt
}

# Create directories
echo "📁 Setting up directories..."
python3 -c "from config import create_directories; create_directories()" 2>/dev/null || {
    echo "⚠️  Warning: Could not create directories automatically"
    mkdir -p uploads static/results models processed_data demo_images demo_videos
}

# Start the application
echo "🌐 Starting web application..."
echo "📍 The application will be available at: http://localhost:5000"
echo "🛑 Press Ctrl+C to stop the application"
echo ""

python3 app.py


#!/bin/bash

# Deepfake Detection AI - Application Runner
# This script sets up and runs the Flask web application

echo "Deepfake Detection AI - Starting Application"
echo "============================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

echo "Using Python: $(python3 --version)"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements if needed
if [ ! -f "venv/pyvenv.cfg" ] || [ ! -d "venv/lib" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
fi

# Create necessary directories
echo "Setting up directories..."
python3 -c "from config import create_directories; create_directories()"

# Run the application
echo "Starting Flask application..."
echo "The application will be available at: http://localhost:5000"
echo "Press Ctrl+C to stop the application"
echo ""

python3 app.py

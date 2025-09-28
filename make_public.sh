#!/bin/bash
# Make the deepfake detection system publicly accessible using ngrok

echo "🌐 Making Deepfake Detection System Publicly Accessible"
echo "======================================================"
echo ""

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null; then
    echo "❌ ngrok not found. Installing..."
    
    # Detect OS and install ngrok
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install ngrok/ngrok/ngrok
        else
            echo "Please install Homebrew first, then run: brew install ngrok/ngrok/ngrok"
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
        echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
        sudo apt update && sudo apt install ngrok
    else
        echo "Please install ngrok manually from: https://ngrok.com/download"
        exit 1
    fi
fi

# Check if ngrok is authenticated
if ! ngrok config check &> /dev/null; then
    echo "🔐 Please authenticate ngrok first:"
    echo "1. Go to https://dashboard.ngrok.com/get-started/your-authtoken"
    echo "2. Copy your authtoken"
    echo "3. Run: ngrok config add-authtoken YOUR_TOKEN"
    echo ""
    read -p "Press Enter after setting up ngrok authentication..."
fi

# Start the Flask app in background
echo "🚀 Starting Flask application..."
python3 app_production.py &
FLASK_PID=$!

# Wait a moment for Flask to start
sleep 3

# Start ngrok tunnel
echo "🌐 Creating public tunnel..."
ngrok http 5000 --log=stdout &
NGROK_PID=$!

# Wait for ngrok to start
sleep 5

# Get the public URL
echo "🔍 Getting public URL..."
PUBLIC_URL=$(curl -s http://localhost:4040/api/tunnels | python3 -c "
import sys, json
data = json.load(sys.stdin)
if data['tunnels']:
    print(data['tunnels'][0]['public_url'])
else:
    print('Error: No tunnels found')
")

if [[ $PUBLIC_URL == http* ]]; then
    echo ""
    echo "🎉 SUCCESS! Your deepfake detection system is now public!"
    echo "========================================================"
    echo "🌐 Public URL: $PUBLIC_URL"
    echo ""
    echo "📱 Anyone can now access your app at the URL above"
    echo "🔗 Share this URL with others to let them use your system"
    echo ""
    echo "📊 Features available:"
    echo "   ✅ Upload images (JPG, PNG)"
    echo "   ✅ Upload videos (MP4, AVI, MOV, etc.)"
    echo "   ✅ Get real-time deepfake detection results"
    echo "   ✅ Detailed confidence scores and analysis"
    echo ""
    echo "🛑 To stop the public access, press Ctrl+C"
    echo ""
    
    # Keep the script running
    wait
else
    echo "❌ Failed to get public URL. Please check ngrok status."
    kill $FLASK_PID 2>/dev/null
    kill $NGROK_PID 2>/dev/null
    exit 1
fi

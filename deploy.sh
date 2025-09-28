#!/bin/bash
# Universal deployment script for deepfake detection system

echo "🌐 Deepfake Detection System - Public Deployment"
echo "================================================"
echo ""

echo "Choose your deployment platform:"
echo "1. Heroku (Easiest - Free tier available)"
echo "2. Railway (Modern - Free tier available)"
echo "3. Render (Simple - Free tier available)"
echo "4. Docker (Local or any platform)"
echo "5. Show deployment guide"
echo ""

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo "🚀 Deploying to Heroku..."
        ./deploy_heroku.sh
        ;;
    2)
        echo "🚀 Deploying to Railway..."
        ./deploy_railway.sh
        ;;
    3)
        echo "📖 Render Deployment Instructions:"
        echo "1. Push your code to GitHub"
        echo "2. Go to https://render.com"
        echo "3. Connect GitHub and create new Web Service"
        echo "4. Use these settings:"
        echo "   - Build Command: pip install -r requirements_production.txt"
        echo "   - Start Command: gunicorn app_production:app --bind 0.0.0.0:\$PORT"
        echo "   - Python Version: 3.12"
        ;;
    4)
        echo "🐳 Building Docker image..."
        docker build -t deepfake-detector .
        echo "✅ Docker image built successfully!"
        echo "Run with: docker run -p 5000:5000 deepfake-detector"
        echo "Or use: docker-compose up"
        ;;
    5)
        echo "📖 Opening deployment guide..."
        if command -v open &> /dev/null; then
            open DEPLOYMENT_GUIDE.md
        elif command -v xdg-open &> /dev/null; then
            xdg-open DEPLOYMENT_GUIDE.md
        else
            echo "Please open DEPLOYMENT_GUIDE.md manually"
        fi
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        ;;
esac

echo ""
echo "🎉 Deployment process completed!"
echo "📖 For detailed instructions, see DEPLOYMENT_GUIDE.md"

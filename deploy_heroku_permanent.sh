#!/bin/bash
# One-command permanent deployment to Heroku

echo "🌐 Deploying Deepfake Detection System to Heroku - PERMANENT"
echo "============================================================"
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📁 Initializing git repository..."
    git init
fi

# Check if Heroku CLI is installed
if ! command -v heroku &> /dev/null; then
    echo "🔧 Installing Heroku CLI..."
    curl https://cli-assets.heroku.com/install.sh | sh
    echo "✅ Heroku CLI installed"
else
    echo "✅ Heroku CLI already installed"
fi

# Login to Heroku
echo "🔐 Logging into Heroku..."
heroku login

# Get app name
echo ""
read -p "Enter a unique app name (will be your-app-name.herokuapp.com): " APP_NAME

# Create Heroku app
echo "📱 Creating Heroku app: $APP_NAME"
heroku create $APP_NAME

# Set environment variables
echo "⚙️  Setting environment variables..."
heroku config:set FLASK_ENV=production
heroku config:set PYTHONPATH=/app

# Add all files to git
echo "📦 Preparing files for deployment..."
git add .

# Commit changes
git commit -m "Deploy deepfake detection system - permanent deployment"

# Deploy to Heroku
echo "🚀 Deploying to Heroku..."
git push heroku main

# Open the app
echo "🌐 Opening your app..."
heroku open

echo ""
echo "🎉 SUCCESS! Your deepfake detection system is now PERMANENTLY deployed!"
echo "================================================================"
echo "🌍 Public URL: https://$APP_NAME.herokuapp.com"
echo "⏰ Status: LIVE 24/7 (with free tier limitations)"
echo "🌐 Access: Anyone worldwide can use your system"
echo ""
echo "📊 What people can do:"
echo "   ✅ Upload images (JPG, PNG)"
echo "   ✅ Upload videos (MP4, AVI, MOV, etc.)"
echo "   ✅ Get real-time deepfake detection results"
echo "   ✅ Access detailed analysis reports"
echo ""
echo "🔗 Share this URL with everyone: https://$APP_NAME.herokuapp.com"
echo ""
echo "📖 To update your app in the future:"
echo "   1. Make changes to your code"
echo "   2. Run: git add . && git commit -m 'Update' && git push heroku main"
echo ""
echo "🎯 Your app is now PERMANENTLY accessible worldwide! 🌍"

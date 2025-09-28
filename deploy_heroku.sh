#!/bin/bash
# Deploy to Heroku

echo "🚀 Deploying Deepfake Detection System to Heroku"
echo "================================================"

# Check if Heroku CLI is installed
if ! command -v heroku &> /dev/null; then
    echo "❌ Heroku CLI not found. Please install it first:"
    echo "   https://devcenter.heroku.com/articles/heroku-cli"
    exit 1
fi

# Login to Heroku
echo "🔐 Logging into Heroku..."
heroku login

# Create Heroku app
echo "📱 Creating Heroku app..."
read -p "Enter your app name (must be unique): " APP_NAME
heroku create $APP_NAME

# Set environment variables
echo "⚙️  Setting environment variables..."
heroku config:set FLASK_ENV=production
heroku config:set PYTHONPATH=/app

# Deploy
echo "📦 Deploying to Heroku..."
git add .
git commit -m "Deploy deepfake detection system"
git push heroku main

# Open the app
echo "🌐 Opening your app..."
heroku open

echo "✅ Deployment complete!"
echo "🔗 Your app is now live at: https://$APP_NAME.herokuapp.com"

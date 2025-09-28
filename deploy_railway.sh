#!/bin/bash
# Deploy to Railway

echo "🚀 Deploying Deepfake Detection System to Railway"
echo "================================================"

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Please install it first:"
    echo "   npm install -g @railway/cli"
    exit 1
fi

# Login to Railway
echo "🔐 Logging into Railway..."
railway login

# Initialize Railway project
echo "📱 Initializing Railway project..."
railway init

# Deploy
echo "📦 Deploying to Railway..."
railway up

echo "✅ Deployment complete!"
echo "🔗 Your app will be available at the URL shown above"

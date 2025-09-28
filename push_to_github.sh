#!/bin/bash
# Script to push deepfake detection system to GitHub

echo "🌐 Pushing Deepfake Detection System to GitHub"
echo "=============================================="
echo ""

echo "📋 Before running this script:"
echo "1. Go to https://github.com"
echo "2. Create a new repository named 'deepfake-detector'"
echo "3. Make it PUBLIC (required for free deployment)"
echo "4. Don't add README, .gitignore, or license"
echo ""

read -p "Have you created the GitHub repository? (y/n): " created

if [ "$created" != "y" ]; then
    echo "❌ Please create the GitHub repository first, then run this script again."
    echo "🔗 Go to: https://github.com/new"
    exit 1
fi

echo ""
echo "🔗 Enter your GitHub repository URL:"
echo "   It should look like: https://github.com/YOUR_USERNAME/deepfake-detector.git"
read -p "GitHub URL: " github_url

if [ -z "$github_url" ]; then
    echo "❌ Please provide the GitHub URL"
    exit 1
fi

echo ""
echo "📦 Adding GitHub remote..."
git remote add origin $github_url

echo "🚀 Pushing code to GitHub..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 SUCCESS! Your code is now on GitHub!"
    echo "🔗 Repository: $github_url"
    echo ""
    echo "📋 Next steps:"
    echo "1. Go to https://render.com"
    echo "2. Sign up with GitHub"
    echo "3. Create new Web Service"
    echo "4. Connect your repository"
    echo "5. Use these settings:"
    echo "   - Build Command: pip install -r requirements_production.txt"
    echo "   - Start Command: gunicorn app_production:app --bind 0.0.0.0:\$PORT"
    echo "   - Python Version: 3.12"
    echo ""
    echo "🌍 Your app will be live at: https://your-app-name.onrender.com"
else
    echo "❌ Failed to push to GitHub. Please check your repository URL and try again."
fi

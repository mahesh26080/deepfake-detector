#!/bin/bash
# Permanent deployment script for deepfake detection system

echo "🌐 Deepfake Detection System - Permanent Deployment"
echo "=================================================="
echo ""

echo "Choose your permanent deployment platform:"
echo "1. Heroku (Easiest - Free forever)"
echo "2. Railway (Modern - Free tier)"
echo "3. Render (Simple - Free tier)"
echo "4. Google Cloud Run (Scalable - Free tier)"
echo "5. AWS Elastic Beanstalk (Enterprise)"
echo "6. DigitalOcean App Platform (Performance)"
echo "7. Vercel (Serverless - Free tier)"
echo "8. Fly.io (Global Edge - Free tier)"
echo "9. Show detailed guide"
echo ""

read -p "Enter your choice (1-9): " choice

case $choice in
    1)
        echo "🚀 Deploying to Heroku (Free Forever)..."
        echo ""
        echo "Step 1: Installing Heroku CLI..."
        if ! command -v heroku &> /dev/null; then
            curl https://cli-assets.heroku.com/install.sh | sh
            echo "✅ Heroku CLI installed"
        else
            echo "✅ Heroku CLI already installed"
        fi
        
        echo ""
        echo "Step 2: Please login to Heroku..."
        heroku login
        
        echo ""
        echo "Step 3: Creating Heroku app..."
        read -p "Enter a unique app name (will be your-app-name.herokuapp.com): " APP_NAME
        heroku create $APP_NAME
        
        echo ""
        echo "Step 4: Setting environment variables..."
        heroku config:set FLASK_ENV=production
        heroku config:set PYTHONPATH=/app
        
        echo ""
        echo "Step 5: Deploying to Heroku..."
        git add .
        git commit -m "Deploy deepfake detection system"
        git push heroku main
        
        echo ""
        echo "Step 6: Opening your app..."
        heroku open
        
        echo ""
        echo "🎉 SUCCESS! Your app is now live at: https://$APP_NAME.herokuapp.com"
        echo "🌍 This URL will work FOREVER and is accessible worldwide!"
        ;;
        
    2)
        echo "🚀 Deploying to Railway..."
        echo ""
        echo "Installing Railway CLI..."
        if ! command -v railway &> /dev/null; then
            npm install -g @railway/cli
        fi
        
        echo "Logging into Railway..."
        railway login
        
        echo "Initializing project..."
        railway init
        
        echo "Deploying..."
        railway up
        
        echo "🎉 SUCCESS! Your app is now live on Railway!"
        ;;
        
    3)
        echo "📖 Render Deployment Instructions:"
        echo ""
        echo "1. Push your code to GitHub:"
        echo "   git init"
        echo "   git add ."
        echo "   git commit -m 'Deploy deepfake detection'"
        echo "   git remote add origin https://github.com/yourusername/deepfake-detector.git"
        echo "   git push -u origin main"
        echo ""
        echo "2. Go to https://render.com"
        echo "3. Sign up with GitHub"
        echo "4. Click 'New +' → 'Web Service'"
        echo "5. Connect your repository"
        echo "6. Use these settings:"
        echo "   - Build Command: pip install -r requirements_production.txt"
        echo "   - Start Command: gunicorn app_production:app --bind 0.0.0.0:\$PORT"
        echo "   - Python Version: 3.12"
        echo ""
        echo "🎉 Your app will be live at: https://your-app.onrender.com"
        ;;
        
    4)
        echo "🚀 Deploying to Google Cloud Run..."
        echo ""
        echo "Installing Google Cloud CLI..."
        if ! command -v gcloud &> /dev/null; then
            curl https://sdk.cloud.google.com | bash
            exec -l $SHELL
        fi
        
        echo "Initializing Google Cloud..."
        gcloud init
        
        echo "Building and deploying..."
        read -p "Enter your Google Cloud project ID: " PROJECT_ID
        gcloud builds submit --tag gcr.io/$PROJECT_ID/deepfake-detector
        gcloud run deploy deepfake-detector \
          --image gcr.io/$PROJECT_ID/deepfake-detector \
          --platform managed \
          --region us-central1 \
          --allow-unauthenticated
        
        echo "🎉 SUCCESS! Your app is now live on Google Cloud Run!"
        ;;
        
    5)
        echo "🚀 Deploying to AWS Elastic Beanstalk..."
        echo ""
        echo "Installing AWS CLI..."
        if ! command -v aws &> /dev/null; then
            curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
            sudo installer -pkg AWSCLIV2.pkg -target /
        fi
        
        echo "Installing EB CLI..."
        pip install awsebcli
        
        echo "Initializing..."
        eb init
        
        echo "Creating environment..."
        eb create production
        
        echo "Deploying..."
        eb deploy
        
        echo "🎉 SUCCESS! Your app is now live on AWS!"
        ;;
        
    6)
        echo "📖 DigitalOcean Deployment Instructions:"
        echo ""
        echo "1. Create app.yaml file:"
        echo "   name: deepfake-detector"
        echo "   services:"
        echo "   - name: web"
        echo "     source_dir: /"
        echo "     github:"
        echo "       repo: yourusername/deepfake-detector"
        echo "       branch: main"
        echo "     run_command: gunicorn app_production:app --bind 0.0.0.0:8080"
        echo "     environment_slug: python"
        echo "     instance_count: 1"
        echo "     instance_size_slug: basic-xxs"
        echo "     http_port: 8080"
        echo "     routes:"
        echo "     - path: /"
        echo ""
        echo "2. Go to https://cloud.digitalocean.com"
        echo "3. Create new app"
        echo "4. Connect GitHub repository"
        echo "5. Deploy"
        echo ""
        echo "🎉 Your app will be live at: https://your-app.ondigitalocean.app"
        ;;
        
    7)
        echo "🚀 Deploying to Vercel..."
        echo ""
        echo "Installing Vercel CLI..."
        if ! command -v vercel &> /dev/null; then
            npm install -g vercel
        fi
        
        echo "Logging into Vercel..."
        vercel login
        
        echo "Deploying..."
        vercel --prod
        
        echo "🎉 SUCCESS! Your app is now live on Vercel!"
        ;;
        
    8)
        echo "🚀 Deploying to Fly.io..."
        echo ""
        echo "Installing Fly CLI..."
        if ! command -v flyctl &> /dev/null; then
            brew install flyctl
        fi
        
        echo "Logging into Fly.io..."
        flyctl auth login
        
        echo "Launching app..."
        flyctl launch
        
        echo "Deploying..."
        flyctl deploy
        
        echo "🎉 SUCCESS! Your app is now live on Fly.io!"
        ;;
        
    9)
        echo "📖 Opening detailed deployment guide..."
        if command -v open &> /dev/null; then
            open PERMANENT_DEPLOYMENT.md
        elif command -v xdg-open &> /dev/null; then
            xdg-open PERMANENT_DEPLOYMENT.md
        else
            echo "Please open PERMANENT_DEPLOYMENT.md manually"
        fi
        ;;
        
    *)
        echo "❌ Invalid choice. Please run the script again."
        ;;
esac

echo ""
echo "🎉 Deployment process completed!"
echo "🌍 Your deepfake detection system is now permanently accessible worldwide!"
echo "📖 For detailed instructions, see PERMANENT_DEPLOYMENT.md"

# 🌐 Deepfake Detection System - Public Deployment Guide

This guide will help you deploy your deepfake detection system so everyone can access it publicly.

## 🚀 Quick Deployment Options

### Option 1: Heroku (Easiest - Free Tier Available)

1. **Install Heroku CLI**:
   ```bash
   # macOS
   brew install heroku/brew/heroku
   
   # Windows
   # Download from: https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Deploy**:
   ```bash
   chmod +x deploy_heroku.sh
   ./deploy_heroku.sh
   ```

3. **Manual Heroku Deployment**:
   ```bash
   # Login to Heroku
   heroku login
   
   # Create app
   heroku create your-app-name
   
   # Deploy
   git add .
   git commit -m "Deploy deepfake detection"
   git push heroku main
   
   # Open app
   heroku open
   ```

### Option 2: Railway (Modern Platform - Free Tier)

1. **Install Railway CLI**:
   ```bash
   npm install -g @railway/cli
   ```

2. **Deploy**:
   ```bash
   chmod +x deploy_railway.sh
   ./deploy_railway.sh
   ```

### Option 3: Render (Simple & Reliable)

1. **Connect GitHub Repository**:
   - Push your code to GitHub
   - Go to [render.com](https://render.com)
   - Connect your GitHub account
   - Create new "Web Service"
   - Select your repository

2. **Configure Build Settings**:
   - **Build Command**: `pip install -r requirements_production.txt`
   - **Start Command**: `gunicorn app_production:app --bind 0.0.0.0:$PORT`
   - **Python Version**: 3.12

### Option 4: Google Cloud Run (Scalable)

1. **Install Google Cloud CLI**:
   ```bash
   # macOS
   brew install google-cloud-sdk
   
   # Initialize
   gcloud init
   ```

2. **Deploy**:
   ```bash
   # Build and push to Google Container Registry
   gcloud builds submit --tag gcr.io/PROJECT-ID/deepfake-detector
   
   # Deploy to Cloud Run
   gcloud run deploy deepfake-detector \
     --image gcr.io/PROJECT-ID/deepfake-detector \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

### Option 5: AWS Elastic Beanstalk

1. **Install EB CLI**:
   ```bash
   pip install awsebcli
   ```

2. **Deploy**:
   ```bash
   # Initialize
   eb init
   
   # Create environment
   eb create production
   
   # Deploy
   eb deploy
   ```

## 🐳 Docker Deployment (Any Platform)

### Local Docker
```bash
# Build image
docker build -t deepfake-detector .

# Run container
docker run -p 5000:5000 deepfake-detector
```

### Docker Compose
```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down
```

## 🔧 Configuration for Production

### Environment Variables
Set these in your deployment platform:

```bash
FLASK_ENV=production
PYTHONPATH=/app
PORT=5000
```

### File Size Limits
- **Heroku**: 100MB max file size
- **Railway**: 100MB max file size  
- **Render**: 100MB max file size
- **Google Cloud Run**: 32MB max file size (increase in settings)

## 📊 Performance Optimization

### For High Traffic:
1. **Increase Workers** (in Procfile):
   ```
   web: gunicorn app_production:app --bind 0.0.0.0:$PORT --workers 4 --timeout 120
   ```

2. **Add Caching**:
   ```python
   from flask_caching import Cache
   cache = Cache(app, config={'CACHE_TYPE': 'simple'})
   ```

3. **Use CDN** for static files

## 🔒 Security Considerations

### Add Security Headers:
```python
from flask_talisman import Talisman
Talisman(app)
```

### Rate Limiting:
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)
```

## 📈 Monitoring & Analytics

### Add Health Check:
```python
@app.route('/health')
def health():
    return {'status': 'healthy', 'model_loaded': True}
```

### Add Logging:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

## 🎯 Recommended Deployment Strategy

### For Beginners:
1. **Start with Heroku** (easiest setup)
2. **Use the free tier** for testing
3. **Upgrade to paid** when you need more resources

### For Production:
1. **Use Google Cloud Run** or **AWS Elastic Beanstalk**
2. **Set up monitoring** and logging
3. **Configure auto-scaling**
4. **Add CDN** for better performance

## 🚨 Important Notes

1. **Model Size**: The trained model is ~50MB, ensure your platform supports this
2. **Memory**: Requires at least 1GB RAM for optimal performance
3. **Timeout**: Set timeout to at least 120 seconds for video processing
4. **File Cleanup**: The app automatically cleans up uploaded files after processing

## 🔗 After Deployment

1. **Test your deployment**:
   ```bash
   curl https://your-app-url.herokuapp.com/health
   ```

2. **Share the URL** with others:
   - Your app will be accessible at: `https://your-app-name.herokuapp.com`
   - Anyone can upload images/videos and get detection results

3. **Monitor usage**:
   - Check logs in your platform's dashboard
   - Monitor resource usage
   - Set up alerts for errors

## 🆘 Troubleshooting

### Common Issues:
1. **Model not loading**: Ensure `models/robust_trained_model.h5` is included in deployment
2. **Memory errors**: Reduce file size limits or increase memory allocation
3. **Timeout errors**: Increase timeout settings
4. **SSL errors**: Most platforms handle SSL automatically

### Getting Help:
- Check platform-specific documentation
- Monitor application logs
- Test locally with Docker first

---

## 🎉 Success!

Once deployed, your deepfake detection system will be publicly accessible and anyone can:
- Upload images and videos
- Get real-time detection results
- Access the web interface from anywhere
- Use the API for integration

**Your app will be live at**: `https://your-app-name.herokuapp.com` (or your chosen platform URL)

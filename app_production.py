import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import tensorflow as tf
from robust_detector import RobustDeepfakeDetector
import json
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# Initialize the robust deepfake detector with trained model
model_path = "models/robust_trained_model.h5"
detector = RobustDeepfakeDetector(model_path)
if os.path.exists(model_path):
    detector.mark_as_trained()
    print("✓ Loaded trained robust model")
else:
    print("⚠️  Using untrained model - training will happen on first use")

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Determine file type and process accordingly
            file_ext = filename.rsplit('.', 1)[1].lower()
            
            if file_ext in ['jpg', 'jpeg', 'png']:
                result = detector.detect_image(filepath)
            else:
                result = detector.detect_video(filepath)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'result': result
            })
            
        except Exception as e:
            # Clean up uploaded file on error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({
                'success': False,
                'error': f'Processing failed: {str(e)}'
            })
    
    return jsonify({'success': False, 'error': 'Invalid file type'})

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': os.path.exists(model_path)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

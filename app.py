from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import tensorflow as tf
from robust_detector import RobustDeepfakeDetector
import json

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
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Detect if it's a video or image
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                result = detector.detect_video(filepath)
            else:
                result = detector.detect_image(filepath)
            
            # Save result
            result_filename = f"result_{filename.rsplit('.', 1)[0]}.json"
            result_path = os.path.join('static/results', result_filename)
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            return jsonify({
                'success': True,
                'result': result,
                'result_file': result_filename
            })
            
        except Exception as e:
            return jsonify({'error': f'Detection failed: {str(e)}'}), 500
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/results/<filename>')
def get_result(filename):
    return send_from_directory('static/results', filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

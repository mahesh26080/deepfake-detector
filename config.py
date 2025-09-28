"""
Configuration file for Deepfake Detection AI
"""

import os

# Model Configuration
MODEL_CONFIG = {
    'input_size': (224, 224),
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 50,
    'threshold': 0.5,
    'sample_frames': 30  # For video analysis
}

# Flask Configuration
FLASK_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True,
    'max_content_length': 100 * 1024 * 1024,  # 100MB
    'upload_folder': 'uploads'
}

# File Upload Configuration
UPLOAD_CONFIG = {
    'allowed_extensions': {'mp4', 'avi', 'mov', 'mkv', 'webm', 'jpg', 'jpeg', 'png'},
    'max_file_size': 100 * 1024 * 1024,  # 100MB
    'temp_cleanup': True  # Clean up uploaded files after processing
}

# Model Paths
MODEL_PATHS = {
    'base_model': 'models/base_model.h5',
    'trained_model': 'models/trained_model.h5',
    'best_model': 'models/best_model.h5'
}

# Data Paths
DATA_PATHS = {
    'raw_data': 'data',
    'processed_data': 'processed_data',
    'results': 'static/results',
    'uploads': 'uploads'
}

# Detection Configuration
DETECTION_CONFIG = {
    'video_frame_interval': 30,  # Extract every 30th frame
    'max_video_frames': 50,      # Maximum frames to analyze
    'confidence_threshold': 0.5,  # Threshold for fake detection
    'enable_temporal_analysis': True
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/deepfake_detection.log'
}

# Create necessary directories
def create_directories():
    """Create all necessary directories"""
    directories = [
        'models',
        'data/real/images',
        'data/real/videos', 
        'data/fake/images',
        'data/fake/videos',
        'processed_data/train/real',
        'processed_data/train/fake',
        'processed_data/val/real',
        'processed_data/val/fake',
        'processed_data/test/real',
        'processed_data/test/fake',
        'static/results',
        'uploads',
        'logs',
        'demo_images',
        'demo_videos'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Initialize directories on import
create_directories()


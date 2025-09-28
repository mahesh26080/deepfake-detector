#!/usr/bin/env python3
"""
Setup script for Deepfake Detection AI
This script helps set up the environment and install dependencies
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✓ Python version: {sys.version}")
    return True

def install_requirements():
    """Install required packages"""
    print("\nInstalling requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    directories = [
        "uploads",
        "static/results",
        "models",
        "processed_data",
        "demo_images",
        "demo_videos"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def check_gpu_support():
    """Check if GPU support is available"""
    print("\nChecking GPU support...")
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ GPU support available: {len(gpus)} GPU(s) detected")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
        else:
            print("⚠ No GPU detected - training will use CPU (slower)")
        return len(gpus) > 0
    except ImportError:
        print("⚠ TensorFlow not installed yet")
        return False

def test_installation():
    """Test if the installation works"""
    print("\nTesting installation...")
    try:
        from deepfake_detector import DeepfakeDetector
        detector = DeepfakeDetector()
        print("✓ Deepfake detector initialized successfully")
        
        # Test basic functionality
        import cv2
        import numpy as np
        
        # Create a test image
        test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite("test_image.jpg", test_img)
        
        # Test detection
        result = detector.detect_image("test_image.jpg")
        if "error" not in result:
            print("✓ Image detection test passed")
        else:
            print(f"⚠ Image detection test failed: {result['error']}")
        
        # Clean up test file
        if os.path.exists("test_image.jpg"):
            os.remove("test_image.jpg")
        
        return True
    except Exception as e:
        print(f"✗ Installation test failed: {e}")
        return False

def create_sample_data():
    """Create sample data for testing"""
    print("\nCreating sample data...")
    try:
        from data_preparation import DataPreparator
        preparator = DataPreparator()
        preparator.create_sample_dataset(num_real=100, num_fake=100)
        print("✓ Sample dataset created")
        return True
    except Exception as e:
        print(f"⚠ Could not create sample data: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run the web application:")
    print("   python app.py")
    print("   Then open http://localhost:5000 in your browser")
    print("\n2. Or run the demo script:")
    print("   python demo.py --full-demo")
    print("\n3. To train your own model:")
    print("   python train_model.py")
    print("\n4. For more information, see README.md")
    print("\n" + "="*60)

def main():
    """Main setup function"""
    print("Deepfake Detection AI - Setup Script")
    print("="*40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("Failed to install requirements. Please check the error messages above.")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Check GPU support
    check_gpu_support()
    
    # Test installation
    if not test_installation():
        print("Installation test failed. Please check the error messages above.")
        sys.exit(1)
    
    # Create sample data
    create_sample_data()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()


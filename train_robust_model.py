#!/usr/bin/env python3
"""
Training script for the robust deepfake detection model
"""

import os
import sys
from robust_detector import RobustDeepfakeDetector

def main():
    print("🚀 Training Robust Deepfake Detection Model")
    print("=" * 50)
    
    # Initialize the robust detector
    detector = RobustDeepfakeDetector()
    
    # Train the model with sample data
    print("📚 Creating sample dataset and training model...")
    history = detector.train_quick_model()
    
    # Mark the model as trained
    detector.mark_as_trained()
    
    print("✅ Training completed!")
    print("🎯 The robust model should now provide much better detection results")
    print("🔄 Restart the web application to use the robust model")
    
    # Test the robust model
    print("\n🧪 Testing the robust model...")
    
    # Create test images
    import cv2
    import numpy as np
    
    # Create a "real" looking test image
    test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    cv2.circle(test_img, (112, 112), 50, (255, 255, 255), -1)
    cv2.imwrite("test_real.jpg", test_img)
    
    # Test detection
    result = detector.detect_image("test_real.jpg")
    print(f"Test result: {result}")
    
    # Clean up test file
    if os.path.exists("test_real.jpg"):
        os.remove("test_real.jpg")

if __name__ == "__main__":
    main()

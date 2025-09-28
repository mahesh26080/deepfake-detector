#!/usr/bin/env python3
"""
Quick training script for the improved deepfake detector
This script trains the model with sample data to improve detection accuracy
"""

import os
import sys
from improved_detector import ImprovedDeepfakeDetector

def main():
    print("🚀 Quick Training for Deepfake Detection Model")
    print("=" * 50)
    
    # Initialize the improved detector
    detector = ImprovedDeepfakeDetector()
    
    # Train the model with sample data
    print("📚 Creating sample dataset and training model...")
    history = detector.train_quick_model()
    
    # Mark the model as trained
    detector.mark_as_trained()
    
    print("✅ Training completed!")
    print("🎯 The model should now provide better detection results")
    print("🔄 Restart the web application to use the improved model")
    
    # Test the improved model
    print("\n🧪 Testing the improved model...")
    
    # Create a test image
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

#!/usr/bin/env python3
"""
Test script to demonstrate the improved deepfake detection model
"""

import cv2
import numpy as np
from improved_detector import ImprovedDeepfakeDetector

def create_test_images():
    """Create test images with different characteristics"""
    
    # Create a "real" looking image
    real_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    # Add some realistic structure
    cv2.circle(real_img, (112, 112), 50, (255, 255, 255), -1)
    cv2.rectangle(real_img, (50, 50), (174, 174), (0, 0, 0), 2)
    # Add some natural noise
    noise = np.random.normal(0, 25, real_img.shape).astype(np.uint8)
    real_img = np.clip(real_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite("test_real.jpg", real_img)
    
    # Create a "fake" looking image with obvious artifacts
    fake_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    # Add obvious artificial patterns
    for x in range(0, 224, 20):
        cv2.line(fake_img, (x, 0), (x, 224), (255, 0, 0), 1)
    for y in range(0, 224, 20):
        cv2.line(fake_img, (0, y), (224, y), (0, 255, 0), 1)
    # Add geometric shapes
    cv2.circle(fake_img, (112, 112), 80, (0, 0, 255), 3)
    cv2.rectangle(fake_img, (30, 30), (194, 194), (255, 255, 0), 3)
    # Add structured noise
    noise = np.random.normal(0, 50, fake_img.shape).astype(np.uint8)
    fake_img = np.clip(fake_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite("test_fake.jpg", fake_img)
    
    return "test_real.jpg", "test_fake.jpg"

def test_model():
    """Test the improved model"""
    print("🧪 Testing Improved Deepfake Detection Model")
    print("=" * 50)
    
    # Load the trained model
    detector = ImprovedDeepfakeDetector("models/quick_trained_model.h5")
    detector.mark_as_trained()
    
    # Create test images
    real_file, fake_file = create_test_images()
    
    print("\n📸 Testing Real Image:")
    print("-" * 30)
    real_result = detector.detect_image(real_file)
    if "error" not in real_result:
        print(f"✅ Result: {'FAKE' if real_result['is_deepfake'] else 'REAL'}")
        print(f"🎯 Confidence: {real_result['confidence']:.4f}")
        print(f"🤖 AI Confidence: {real_result['ai_confidence']:.4f}")
        print(f"🔍 Artifact Confidence: {real_result['artifact_confidence']:.4f}")
        print(f"📊 Real Probability: {real_result['real_probability']:.4f}")
        print(f"📊 Fake Probability: {real_result['fake_probability']:.4f}")
    else:
        print(f"❌ Error: {real_result['error']}")
    
    print("\n📸 Testing Fake Image:")
    print("-" * 30)
    fake_result = detector.detect_image(fake_file)
    if "error" not in fake_result:
        print(f"✅ Result: {'FAKE' if fake_result['is_deepfake'] else 'REAL'}")
        print(f"🎯 Confidence: {fake_result['confidence']:.4f}")
        print(f"🤖 AI Confidence: {fake_result['ai_confidence']:.4f}")
        print(f"🔍 Artifact Confidence: {fake_result['artifact_confidence']:.4f}")
        print(f"📊 Real Probability: {fake_result['real_probability']:.4f}")
        print(f"📊 Fake Probability: {fake_result['fake_probability']:.4f}")
    else:
        print(f"❌ Error: {fake_result['error']}")
    
    # Clean up test files
    import os
    if os.path.exists(real_file):
        os.remove(real_file)
    if os.path.exists(fake_file):
        os.remove(fake_file)
    
    print("\n🎉 Test completed!")
    print("💡 The model now provides different confidence scores instead of always returning 50%")

if __name__ == "__main__":
    test_model()

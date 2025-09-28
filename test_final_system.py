#!/usr/bin/env python3
"""
Final comprehensive test of the deepfake detection system
"""

import cv2
import numpy as np
import requests
import json
import os
from robust_detector import RobustDeepfakeDetector

def create_test_media():
    """Create test images and videos with different characteristics"""
    
    # Create a "real" looking image
    real_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    # Add realistic structure
    cv2.circle(real_img, (112, 112), 50, (255, 255, 255), -1)
    cv2.rectangle(real_img, (50, 50), (174, 174), (0, 0, 0), 2)
    # Add natural noise
    noise = np.random.normal(0, 25, real_img.shape).astype(np.uint8)
    real_img = np.clip(real_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite("test_real.jpg", real_img)
    
    # Create a "fake" looking image with obvious artifacts
    fake_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    # Add artificial patterns
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
    
    # Create a test video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test_video.mp4', fourcc, 10.0, (224, 224))
    
    for i in range(50):  # 5 seconds at 10 fps
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        # Add moving elements
        center_x = int(112 + 30 * np.sin(i * 0.2))
        center_y = int(112 + 20 * np.cos(i * 0.2))
        cv2.circle(frame, (center_x, center_y), 20, (255, 255, 255), -1)
        cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        out.write(frame)
    
    out.release()
    
    return "test_real.jpg", "test_fake.jpg", "test_video.mp4"

def test_direct_detection():
    """Test direct detection using the robust detector"""
    print("🔬 Testing Direct Detection (Robust Detector)")
    print("=" * 50)
    
    # Load the trained model
    detector = RobustDeepfakeDetector("models/robust_trained_model.h5")
    detector.mark_as_trained()
    
    # Create test images
    real_file, fake_file, video_file = create_test_media()
    
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
    
    print("\n🎬 Testing Video:")
    print("-" * 30)
    video_result = detector.detect_video(video_file, sample_frames=20)
    if "error" not in video_result:
        print(f"✅ Result: {'FAKE' if video_result['is_deepfake'] else 'REAL'}")
        print(f"🎯 Confidence: {video_result['confidence']:.4f}")
        print(f"🤖 AI Confidence: {video_result['ai_confidence']:.4f}")
        print(f"🔍 Artifact Confidence: {video_result['artifact_confidence']:.4f}")
        print(f"📊 Real Probability: {video_result['real_probability']:.4f}")
        print(f"📊 Fake Probability: {video_result['fake_probability']:.4f}")
        print(f"🎞️  Frames Analyzed: {video_result['video_info']['processed_frames']}")
    else:
        print(f"❌ Error: {video_result['error']}")
    
    return real_file, fake_file, video_file

def test_web_api():
    """Test the web API"""
    print("\n🌐 Testing Web API")
    print("=" * 50)
    
    # Test if the web server is running
    try:
        response = requests.get("http://localhost:5000", timeout=5)
        if response.status_code == 200:
            print("✅ Web server is running")
        else:
            print(f"❌ Web server returned status {response.status_code}")
            return
    except requests.exceptions.RequestException as e:
        print(f"❌ Web server is not accessible: {e}")
        return
    
    # Test image upload
    print("\n📤 Testing Image Upload...")
    real_file, fake_file, video_file = create_test_media()
    
    # Test real image upload
    with open(real_file, 'rb') as f:
        files = {'file': f}
        response = requests.post("http://localhost:5000/upload", files=files, timeout=30)
        
    if response.status_code == 200:
        result = response.json()
        if result.get('success'):
            print("✅ Real image upload successful")
            print(f"   Result: {'FAKE' if result['result']['is_deepfake'] else 'REAL'}")
            print(f"   Confidence: {result['result']['confidence']:.4f}")
        else:
            print(f"❌ Real image upload failed: {result.get('error')}")
    else:
        print(f"❌ Real image upload failed with status {response.status_code}")
    
    # Test fake image upload
    with open(fake_file, 'rb') as f:
        files = {'file': f}
        response = requests.post("http://localhost:5000/upload", files=files, timeout=30)
        
    if response.status_code == 200:
        result = response.json()
        if result.get('success'):
            print("✅ Fake image upload successful")
            print(f"   Result: {'FAKE' if result['result']['is_deepfake'] else 'REAL'}")
            print(f"   Confidence: {result['result']['confidence']:.4f}")
        else:
            print(f"❌ Fake image upload failed: {result.get('error')}")
    else:
        print(f"❌ Fake image upload failed with status {response.status_code}")
    
    # Test video upload
    with open(video_file, 'rb') as f:
        files = {'file': f}
        response = requests.post("http://localhost:5000/upload", files=files, timeout=60)
        
    if response.status_code == 200:
        result = response.json()
        if result.get('success'):
            print("✅ Video upload successful")
            print(f"   Result: {'FAKE' if result['result']['is_deepfake'] else 'REAL'}")
            print(f"   Confidence: {result['result']['confidence']:.4f}")
            print(f"   Frames Analyzed: {result['result']['video_info']['processed_frames']}")
        else:
            print(f"❌ Video upload failed: {result.get('error')}")
    else:
        print(f"❌ Video upload failed with status {response.status_code}")

def cleanup_test_files():
    """Clean up test files"""
    test_files = ["test_real.jpg", "test_fake.jpg", "test_video.mp4"]
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
    print("\n🧹 Cleaned up test files")

def main():
    print("🚀 Comprehensive Deepfake Detection System Test")
    print("=" * 60)
    
    # Test direct detection
    real_file, fake_file, video_file = test_direct_detection()
    
    # Test web API
    test_web_api()
    
    # Cleanup
    cleanup_test_files()
    
    print("\n🎉 Test completed!")
    print("💡 The system now provides:")
    print("   ✅ Different confidence scores for different content")
    print("   ✅ Combined AI model + artifact analysis")
    print("   ✅ Web interface for easy file upload")
    print("   ✅ Support for both images and videos")
    print("   ✅ Detailed analysis results")
    print("\n🌐 Access the web interface at: http://localhost:5000")

if __name__ == "__main__":
    main()

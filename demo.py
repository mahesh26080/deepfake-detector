#!/usr/bin/env python3
"""
Demo script for Deepfake Detection AI
This script demonstrates the core functionality without the web interface
"""

import os
import cv2
import numpy as np
from deepfake_detector import DeepfakeDetector
import matplotlib.pyplot as plt
from PIL import Image
import argparse

def create_demo_images():
    """Create demo images for testing"""
    print("Creating demo images...")
    
    # Create demo directory
    os.makedirs("demo_images", exist_ok=True)
    
    # Create a "real" looking image
    real_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    # Add some realistic structure
    cv2.circle(real_img, (112, 112), 50, (255, 255, 255), -1)
    cv2.rectangle(real_img, (50, 50), (174, 174), (0, 0, 0), 2)
    # Add some natural noise
    noise = np.random.normal(0, 25, real_img.shape).astype(np.uint8)
    real_img = np.clip(real_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite("demo_images/real_demo.jpg", real_img)
    
    # Create a "fake" looking image
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
    cv2.imwrite("demo_images/fake_demo.jpg", fake_img)
    
    print("Demo images created in 'demo_images' directory")

def create_demo_video():
    """Create a demo video for testing"""
    print("Creating demo video...")
    
    os.makedirs("demo_videos", exist_ok=True)
    
    # Video properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('demo_videos/demo_video.mp4', fourcc, 10.0, (224, 224))
    
    # Create frames
    for i in range(100):  # 10 seconds at 10 fps
        # Create a frame that changes over time
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Add moving elements
        center_x = int(112 + 50 * np.sin(i * 0.1))
        center_y = int(112 + 30 * np.cos(i * 0.1))
        cv2.circle(frame, (center_x, center_y), 30, (255, 255, 255), -1)
        
        # Add some text
        cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        out.write(frame)
    
    out.release()
    print("Demo video created: 'demo_videos/demo_video.mp4'")

def run_detection_demo():
    """Run detection on demo files"""
    print("Initializing Deepfake Detector...")
    detector = DeepfakeDetector()
    
    print("\n" + "="*50)
    print("DEEPFAKE DETECTION DEMO")
    print("="*50)
    
    # Test on demo images
    print("\n1. Testing on Demo Images:")
    print("-" * 30)
    
    demo_files = [
        ("demo_images/real_demo.jpg", "Real Demo Image"),
        ("demo_images/fake_demo.jpg", "Fake Demo Image")
    ]
    
    for file_path, description in demo_files:
        if os.path.exists(file_path):
            print(f"\nAnalyzing: {description}")
            result = detector.detect_image(file_path)
            
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Result: {'FAKE' if result['is_deepfake'] else 'REAL'}")
                print(f"Confidence: {result['confidence']:.4f}")
                print(f"Real Probability: {result['real_probability']:.4f}")
                print(f"Fake Probability: {result['fake_probability']:.4f}")
        else:
            print(f"File not found: {file_path}")
    
    # Test on demo video
    print("\n2. Testing on Demo Video:")
    print("-" * 30)
    
    video_path = "demo_videos/demo_video.mp4"
    if os.path.exists(video_path):
        print(f"\nAnalyzing: Demo Video")
        result = detector.detect_video(video_path, sample_frames=20)
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Result: {'FAKE' if result['is_deepfake'] else 'REAL'}")
            print(f"Average Confidence: {result['average_confidence']:.4f}")
            print(f"Confidence Std: {result['confidence_std']:.4f}")
            print(f"Real Probability: {result['real_probability']:.4f}")
            print(f"Fake Probability: {result['fake_probability']:.4f}")
            
            if 'video_info' in result:
                info = result['video_info']
                print(f"Video Duration: {info['duration_seconds']:.1f} seconds")
                print(f"Frames Analyzed: {info['processed_frames']}/{info['total_frames']}")
                print(f"FPS: {info['fps']:.1f}")
    else:
        print(f"Video not found: {video_path}")

def visualize_results():
    """Create visualizations of the detection results"""
    print("\n3. Creating Visualizations:")
    print("-" * 30)
    
    detector = DeepfakeDetector()
    
    # Load demo images
    real_img = cv2.imread("demo_images/real_demo.jpg")
    fake_img = cv2.imread("demo_images/fake_demo.jpg")
    
    if real_img is not None and fake_img is not None:
        # Analyze images
        real_result = detector.detect_image("demo_images/real_demo.jpg")
        fake_result = detector.detect_image("demo_images/fake_demo.jpg")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Real image
        axes[0, 0].imshow(cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title(f"Real Image\nConfidence: {real_result.get('confidence', 0):.4f}")
        axes[0, 0].axis('off')
        
        # Fake image
        axes[0, 1].imshow(cv2.cvtColor(fake_img, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title(f"Fake Image\nConfidence: {fake_result.get('confidence', 0):.4f}")
        axes[0, 1].axis('off')
        
        # Confidence comparison
        labels = ['Real Image', 'Fake Image']
        confidences = [real_result.get('confidence', 0), fake_result.get('confidence', 0)]
        colors = ['green' if c < 0.5 else 'red' for c in confidences]
        
        bars = axes[1, 0].bar(labels, confidences, color=colors, alpha=0.7)
        axes[1, 0].set_title('Detection Confidence')
        axes[1, 0].set_ylabel('Confidence Score')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Threshold')
        axes[1, 0].legend()
        
        # Add confidence values on bars
        for bar, conf in zip(bars, confidences):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{conf:.3f}', ha='center', va='bottom')
        
        # Probability distribution
        real_prob = real_result.get('real_probability', 0)
        fake_prob = real_result.get('fake_probability', 0)
        
        axes[1, 1].pie([real_prob, fake_prob], labels=['Real', 'Fake'], 
                      colors=['green', 'red'], autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Real Image Probabilities')
        
        plt.tight_layout()
        plt.savefig('demo_results.png', dpi=300, bbox_inches='tight')
        print("Visualization saved as 'demo_results.png'")
        plt.show()

def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description='Deepfake Detection Demo')
    parser.add_argument('--create-demo', action='store_true', 
                       help='Create demo images and videos')
    parser.add_argument('--run-detection', action='store_true',
                       help='Run detection on demo files')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    parser.add_argument('--full-demo', action='store_true',
                       help='Run complete demo (create files, detect, visualize)')
    
    args = parser.parse_args()
    
    if args.full_demo or (not any([args.create_demo, args.run_detection, args.visualize])):
        # Run full demo
        create_demo_images()
        create_demo_video()
        run_detection_demo()
        visualize_results()
    else:
        if args.create_demo:
            create_demo_images()
            create_demo_video()
        
        if args.run_detection:
            run_detection_demo()
        
        if args.visualize:
            visualize_results()

if __name__ == "__main__":
    main()


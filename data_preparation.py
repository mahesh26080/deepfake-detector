import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import List, Tuple, Dict
import random
from tqdm import tqdm

class DataPreparator:
    def __init__(self, data_dir: str = "data", output_dir: str = "processed_data"):
        """
        Initialize data preparation utilities
        
        Args:
            data_dir: Directory containing raw data
            output_dir: Directory to save processed data
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "train" / "real").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "train" / "fake").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "val" / "real").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "val" / "fake").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "test" / "real").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "test" / "fake").mkdir(parents=True, exist_ok=True)
    
    def extract_frames_from_video(self, video_path: str, output_dir: str, 
                                 max_frames: int = 50, frame_interval: int = 30) -> List[str]:
        """
        Extract frames from a video file
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            max_frames: Maximum number of frames to extract
            frame_interval: Interval between frames to extract
            
        Returns:
            List of saved frame paths
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []
        
        frame_count = 0
        saved_frames = []
        frame_idx = 0
        
        while cap.isOpened() and len(saved_frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                # Resize frame to standard size
                frame_resized = cv2.resize(frame, (224, 224))
                
                # Save frame
                frame_filename = f"frame_{frame_count:04d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame_resized)
                saved_frames.append(frame_path)
                frame_count += 1
            
            frame_idx += 1
        
        cap.release()
        return saved_frames
    
    def process_video_dataset(self, real_videos_dir: str, fake_videos_dir: str, 
                            split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)):
        """
        Process video dataset and split into train/val/test
        
        Args:
            real_videos_dir: Directory containing real videos
            fake_videos_dir: Directory containing fake videos
            split_ratios: (train, val, test) ratios
        """
        real_videos = list(Path(real_videos_dir).glob("*.mp4")) + \
                     list(Path(real_videos_dir).glob("*.avi")) + \
                     list(Path(real_videos_dir).glob("*.mov"))
        
        fake_videos = list(Path(fake_videos_dir).glob("*.mp4")) + \
                     list(Path(fake_videos_dir).glob("*.avi")) + \
                     list(Path(fake_videos_dir).glob("*.mov"))
        
        print(f"Found {len(real_videos)} real videos and {len(fake_videos)} fake videos")
        
        # Split datasets
        random.shuffle(real_videos)
        random.shuffle(fake_videos)
        
        train_real = real_videos[:int(len(real_videos) * split_ratios[0])]
        val_real = real_videos[int(len(real_videos) * split_ratios[0]):
                              int(len(real_videos) * (split_ratios[0] + split_ratios[1]))]
        test_real = real_videos[int(len(real_videos) * (split_ratios[0] + split_ratios[1])):]
        
        train_fake = fake_videos[:int(len(fake_videos) * split_ratios[0])]
        val_fake = fake_videos[int(len(fake_videos) * split_ratios[0]):
                              int(len(fake_videos) * (split_ratios[0] + split_ratios[1]))]
        test_fake = fake_videos[int(len(fake_videos) * (split_ratios[0] + split_ratios[1])):]
        
        # Process each split
        for split, real_list, fake_list in [("train", train_real, train_fake),
                                          ("val", val_real, val_fake),
                                          ("test", test_real, test_fake)]:
            print(f"\nProcessing {split} set...")
            
            # Process real videos
            for video_path in tqdm(real_list, desc=f"Processing real {split}"):
                output_dir = self.output_dir / split / "real"
                self.extract_frames_from_video(str(video_path), str(output_dir))
            
            # Process fake videos
            for video_path in tqdm(fake_list, desc=f"Processing fake {split}"):
                output_dir = self.output_dir / split / "fake"
                self.extract_frames_from_video(str(video_path), str(output_dir))
    
    def process_image_dataset(self, real_images_dir: str, fake_images_dir: str,
                            split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)):
        """
        Process image dataset and split into train/val/test
        
        Args:
            real_images_dir: Directory containing real images
            fake_images_dir: Directory containing fake images
            split_ratios: (train, val, test) ratios
        """
        real_images = list(Path(real_images_dir).glob("*.jpg")) + \
                     list(Path(real_images_dir).glob("*.jpeg")) + \
                     list(Path(real_images_dir).glob("*.png"))
        
        fake_images = list(Path(fake_images_dir).glob("*.jpg")) + \
                     list(Path(fake_images_dir).glob("*.jpeg")) + \
                     list(Path(fake_images_dir).glob("*.png"))
        
        print(f"Found {len(real_images)} real images and {len(fake_images)} fake images")
        
        # Split datasets
        random.shuffle(real_images)
        random.shuffle(fake_images)
        
        train_real = real_images[:int(len(real_images) * split_ratios[0])]
        val_real = real_images[int(len(real_images) * split_ratios[0]):
                              int(len(real_images) * (split_ratios[0] + split_ratios[1]))]
        test_real = real_images[int(len(real_images) * (split_ratios[0] + split_ratios[1])):]
        
        train_fake = fake_images[:int(len(fake_images) * split_ratios[0])]
        val_fake = fake_images[int(len(fake_images) * split_ratios[0]):
                              int(len(fake_images) * (split_ratios[0] + split_ratios[1]))]
        test_fake = fake_images[int(len(fake_images) * (split_ratios[0] + split_ratios[1])):]
        
        # Copy images to appropriate directories
        for split, real_list, fake_list in [("train", train_real, train_fake),
                                          ("val", val_real, val_fake),
                                          ("test", test_real, test_fake)]:
            print(f"\nProcessing {split} set...")
            
            # Copy real images
            for img_path in tqdm(real_list, desc=f"Copying real {split}"):
                img = cv2.imread(str(img_path))
                if img is not None:
                    img_resized = cv2.resize(img, (224, 224))
                    output_path = self.output_dir / split / "real" / img_path.name
                    cv2.imwrite(str(output_path), img_resized)
            
            # Copy fake images
            for img_path in tqdm(fake_list, desc=f"Copying fake {split}"):
                img = cv2.imread(str(img_path))
                if img is not None:
                    img_resized = cv2.resize(img, (224, 224))
                    output_path = self.output_dir / split / "fake" / img_path.name
                    cv2.imwrite(str(output_path), img_resized)
    
    def create_data_generators(self, batch_size: int = 32, image_size: Tuple[int, int] = (224, 224)):
        """
        Create data generators for training
        
        Args:
            batch_size: Batch size for training
            image_size: Size to resize images to
            
        Returns:
            Tuple of (train_gen, val_gen, test_gen)
        """
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        # No augmentation for validation and test
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_gen = train_datagen.flow_from_directory(
            self.output_dir / "train",
            target_size=image_size,
            batch_size=batch_size,
            class_mode='binary',
            shuffle=True
        )
        
        val_gen = val_test_datagen.flow_from_directory(
            self.output_dir / "val",
            target_size=image_size,
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        test_gen = val_test_datagen.flow_from_directory(
            self.output_dir / "test",
            target_size=image_size,
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        return train_gen, val_gen, test_gen
    
    def create_sample_dataset(self, num_real: int = 100, num_fake: int = 100):
        """
        Create a small sample dataset for testing purposes
        
        Args:
            num_real: Number of real samples to generate
            num_fake: Number of fake samples to generate
        """
        print("Creating sample dataset...")
        
        # Create sample real images (random noise with some structure)
        for i in tqdm(range(num_real), desc="Generating real samples"):
            # Create a more realistic looking image
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # Add some structure to make it look more realistic
            cv2.circle(img, (112, 112), 50, (255, 255, 255), -1)
            cv2.rectangle(img, (50, 50), (174, 174), (0, 0, 0), 2)
            
            # Add some noise
            noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Save to train/val/test splits
            split = "train" if i < num_real * 0.7 else "val" if i < num_real * 0.85 else "test"
            filename = f"real_sample_{i:04d}.jpg"
            cv2.imwrite(str(self.output_dir / split / "real" / filename), img)
        
        # Create sample fake images (more artificial looking)
        for i in tqdm(range(num_fake), desc="Generating fake samples"):
            # Create a more artificial looking image
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # Add obvious artificial patterns
            for x in range(0, 224, 20):
                cv2.line(img, (x, 0), (x, 224), (255, 0, 0), 1)
            for y in range(0, 224, 20):
                cv2.line(img, (0, y), (224, y), (0, 255, 0), 1)
            
            # Add some geometric shapes
            cv2.circle(img, (112, 112), 80, (0, 0, 255), 3)
            cv2.rectangle(img, (30, 30), (194, 194), (255, 255, 0), 3)
            
            # Add more structured noise
            noise = np.random.normal(0, 50, img.shape).astype(np.uint8)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Save to train/val/test splits
            split = "train" if i < num_fake * 0.7 else "val" if i < num_fake * 0.85 else "test"
            filename = f"fake_sample_{i:04d}.jpg"
            cv2.imwrite(str(self.output_dir / split / "fake" / filename), img)
        
        print(f"Sample dataset created with {num_real} real and {num_fake} fake samples")

if __name__ == "__main__":
    # Example usage
    preparator = DataPreparator()
    
    # Create sample dataset for testing
    preparator.create_sample_dataset(num_real=200, num_fake=200)
    
    print("Data preparation completed!")
    print(f"Processed data saved to: {preparator.output_dir}")


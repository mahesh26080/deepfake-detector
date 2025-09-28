import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
from typing import Dict, List, Tuple
import json

class DeepfakeDetector:
    def __init__(self, model_path: str = None):
        """
        Initialize the Deepfake Detector
        
        Args:
            model_path: Path to pre-trained model weights (optional)
        """
        self.model = None
        self.input_size = (224, 224)
        self.load_or_create_model(model_path)
    
    def load_or_create_model(self, model_path: str = None):
        """Load existing model or create a new one"""
        if model_path and os.path.exists(model_path):
            try:
                self.model = tf.keras.models.load_model(model_path)
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Failed to load model: {e}")
                self.create_model()
        else:
            self.create_model()
    
    def create_model(self):
        """Create a CNN model using transfer learning with EfficientNet"""
        try:
            # Load pre-trained EfficientNetB0
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.input_size, 3)
            )
        except Exception as e:
            print(f"Warning: Could not load pre-trained weights: {e}")
            print("Creating model with random weights...")
            # Load EfficientNetB0 without pre-trained weights
            base_model = EfficientNetB0(
                weights=None,
                include_top=False,
                input_shape=(*self.input_size, 3)
            )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom layers for binary classification
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        
        # Create the model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("Created new model with EfficientNetB0 backbone")
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a single frame for model input
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Preprocessed frame ready for model input
        """
        # Resize frame to model input size
        frame_resized = cv2.resize(frame, self.input_size)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values to [0, 1]
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        
        # Add batch dimension
        frame_batch = np.expand_dims(frame_normalized, axis=0)
        
        return frame_batch
    
    def detect_image(self, image_path: str) -> Dict:
        """
        Detect deepfake in a single image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing detection results
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Could not load image"}
            
            # Preprocess image
            processed_image = self.preprocess_frame(image)
            
            # Make prediction
            prediction = self.model.predict(processed_image, verbose=0)
            confidence = float(prediction[0][0])
            
            # Determine if it's a deepfake (threshold = 0.5)
            is_deepfake = confidence > 0.5
            
            return {
                "type": "image",
                "file_path": image_path,
                "is_deepfake": bool(is_deepfake),
                "confidence": confidence,
                "real_probability": 1 - confidence,
                "fake_probability": confidence,
                "threshold": 0.5
            }
            
        except Exception as e:
            return {"error": f"Image detection failed: {str(e)}"}
    
    def detect_video(self, video_path: str, sample_frames: int = 30) -> Dict:
        """
        Detect deepfake in a video by sampling frames
        
        Args:
            video_path: Path to the video file
            sample_frames: Number of frames to sample for analysis
            
        Returns:
            Dictionary containing detection results
        """
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"error": "Could not open video"}
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            # Sample frames evenly throughout the video
            frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
            
            predictions = []
            processed_frames = 0
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Preprocess frame
                processed_frame = self.preprocess_frame(frame)
                
                # Make prediction
                prediction = self.model.predict(processed_frame, verbose=0)
                predictions.append(float(prediction[0][0]))
                processed_frames += 1
            
            cap.release()
            
            if not predictions:
                return {"error": "No frames could be processed"}
            
            # Calculate average confidence
            avg_confidence = np.mean(predictions)
            std_confidence = np.std(predictions)
            
            # Determine if video contains deepfakes
            is_deepfake = avg_confidence > 0.5
            
            # Calculate frame-by-frame results
            frame_results = []
            for i, (frame_idx, pred) in enumerate(zip(frame_indices, predictions)):
                frame_results.append({
                    "frame_number": int(frame_idx),
                    "timestamp": frame_idx / fps if fps > 0 else 0,
                    "confidence": float(pred),
                    "is_deepfake": bool(pred > 0.5)
                })
            
            return {
                "type": "video",
                "file_path": video_path,
                "is_deepfake": bool(is_deepfake),
                "average_confidence": float(avg_confidence),
                "confidence_std": float(std_confidence),
                "real_probability": float(1 - avg_confidence),
                "fake_probability": float(avg_confidence),
                "threshold": 0.5,
                "video_info": {
                    "total_frames": total_frames,
                    "processed_frames": processed_frames,
                    "fps": fps,
                    "duration_seconds": duration
                },
                "frame_analysis": frame_results
            }
            
        except Exception as e:
            return {"error": f"Video detection failed: {str(e)}"}
    
    def save_model(self, model_path: str):
        """Save the trained model"""
        if self.model:
            self.model.save(model_path)
            print(f"Model saved to {model_path}")
    
    def get_model_summary(self):
        """Get model architecture summary"""
        if self.model:
            return self.model.summary()
        return "No model loaded"

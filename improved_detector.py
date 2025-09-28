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

class ImprovedDeepfakeDetector:
    def __init__(self, model_path: str = None):
        """
        Initialize the Improved Deepfake Detector
        
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
            # Try to load pre-trained EfficientNetB0
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.input_size, 3)
            )
            print("✓ Loaded pre-trained EfficientNetB0")
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
    
    def train_quick_model(self):
        """Train a quick model with sample data"""
        print("Training quick model with sample data...")
        
        # Create sample data
        from data_preparation import DataPreparator
        preparator = DataPreparator()
        preparator.create_sample_dataset(num_real=200, num_fake=200)
        
        # Get data generators
        train_gen, val_gen, test_gen = preparator.create_data_generators(batch_size=16)
        
        # Train for a few epochs
        history = self.model.fit(
            train_gen,
            epochs=5,  # Quick training
            validation_data=val_gen,
            verbose=1
        )
        
        # Save the model
        model_path = "models/quick_trained_model.h5"
        os.makedirs("models", exist_ok=True)
        self.model.save(model_path)
        print(f"Quick model saved to {model_path}")
        
        return history
    
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
    
    def analyze_image_artifacts(self, image: np.ndarray) -> Dict:
        """
        Analyze image for common deepfake artifacts
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with artifact analysis results
        """
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Edge analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 2. Frequency domain analysis
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Check for high-frequency artifacts
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        high_freq_region = magnitude_spectrum[center_h-50:center_h+50, center_w-50:center_w+50]
        high_freq_score = np.mean(high_freq_region)
        
        # 3. Color channel analysis
        b, g, r = cv2.split(image)
        color_variance = np.var([np.var(b), np.var(g), np.var(r)])
        
        # 4. Texture analysis using Local Binary Pattern
        from skimage.feature import local_binary_pattern
        try:
            lbp = local_binary_pattern(gray, 8, 1, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
            lbp_entropy = -np.sum((lbp_hist / lbp_hist.sum()) * np.log2(lbp_hist / lbp_hist.sum() + 1e-10))
        except:
            lbp_entropy = 0
        
        # Combine features into a fake probability
        # Higher edge density, high frequency artifacts, and color variance suggest fake
        fake_score = 0.0
        
        # Edge density contribution (0-0.3)
        if edge_density > 0.1:
            fake_score += min(0.3, edge_density * 2)
        
        # High frequency contribution (0-0.4)
        if high_freq_score > 8:
            fake_score += min(0.4, (high_freq_score - 8) * 0.05)
        
        # Color variance contribution (0-0.2)
        if color_variance > 1000:
            fake_score += min(0.2, (color_variance - 1000) / 10000)
        
        # LBP entropy contribution (0-0.1)
        if lbp_entropy > 2.5:
            fake_score += min(0.1, (lbp_entropy - 2.5) * 0.1)
        
        return {
            'edge_density': edge_density,
            'high_freq_score': high_freq_score,
            'color_variance': color_variance,
            'lbp_entropy': lbp_entropy,
            'artifact_score': min(1.0, fake_score)
        }
    
    def detect_image(self, image_path: str) -> Dict:
        """
        Detect deepfake in a single image using both AI model and artifact analysis
        
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
            
            # AI model prediction
            processed_image = self.preprocess_frame(image)
            ai_prediction = self.model.predict(processed_image, verbose=0)
            ai_confidence = float(ai_prediction[0][0])
            
            # Artifact analysis
            artifacts = self.analyze_image_artifacts(image)
            artifact_confidence = artifacts['artifact_score']
            
            # Combine predictions (weighted average)
            # Give more weight to AI model if it's trained, otherwise rely more on artifacts
            if hasattr(self, '_is_trained') and self._is_trained:
                combined_confidence = 0.7 * ai_confidence + 0.3 * artifact_confidence
            else:
                combined_confidence = 0.3 * ai_confidence + 0.7 * artifact_confidence
            
            # Determine if it's a deepfake (threshold = 0.5)
            is_deepfake = combined_confidence > 0.5
            
            return {
                "type": "image",
                "file_path": image_path,
                "is_deepfake": bool(is_deepfake),
                "confidence": combined_confidence,
                "ai_confidence": ai_confidence,
                "artifact_confidence": artifact_confidence,
                "real_probability": 1 - combined_confidence,
                "fake_probability": combined_confidence,
                "threshold": 0.5,
                "artifacts": artifacts
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
            
            ai_predictions = []
            artifact_predictions = []
            processed_frames = 0
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # AI model prediction
                processed_frame = self.preprocess_frame(frame)
                ai_prediction = self.model.predict(processed_frame, verbose=0)
                ai_predictions.append(float(ai_prediction[0][0]))
                
                # Artifact analysis
                artifacts = self.analyze_image_artifacts(frame)
                artifact_predictions.append(artifacts['artifact_score'])
                
                processed_frames += 1
            
            cap.release()
            
            if not ai_predictions:
                return {"error": "No frames could be processed"}
            
            # Calculate average confidences
            avg_ai_confidence = np.mean(ai_predictions)
            avg_artifact_confidence = np.mean(artifact_predictions)
            std_ai_confidence = np.std(ai_predictions)
            std_artifact_confidence = np.std(artifact_predictions)
            
            # Combine predictions
            if hasattr(self, '_is_trained') and self._is_trained:
                combined_confidence = 0.7 * avg_ai_confidence + 0.3 * avg_artifact_confidence
            else:
                combined_confidence = 0.3 * avg_ai_confidence + 0.7 * avg_artifact_confidence
            
            # Determine if video contains deepfakes
            is_deepfake = combined_confidence > 0.5
            
            # Calculate frame-by-frame results
            frame_results = []
            for i, (frame_idx, ai_pred, artifact_pred) in enumerate(zip(frame_indices, ai_predictions, artifact_predictions)):
                frame_combined = 0.7 * ai_pred + 0.3 * artifact_pred if hasattr(self, '_is_trained') and self._is_trained else 0.3 * ai_pred + 0.7 * artifact_pred
                frame_results.append({
                    "frame_number": int(frame_idx),
                    "timestamp": frame_idx / fps if fps > 0 else 0,
                    "confidence": float(frame_combined),
                    "ai_confidence": float(ai_pred),
                    "artifact_confidence": float(artifact_pred),
                    "is_deepfake": bool(frame_combined > 0.5)
                })
            
            return {
                "type": "video",
                "file_path": video_path,
                "is_deepfake": bool(is_deepfake),
                "confidence": float(combined_confidence),
                "ai_confidence": float(avg_ai_confidence),
                "artifact_confidence": float(avg_artifact_confidence),
                "ai_confidence_std": float(std_ai_confidence),
                "artifact_confidence_std": float(std_artifact_confidence),
                "real_probability": float(1 - combined_confidence),
                "fake_probability": float(combined_confidence),
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
    
    def mark_as_trained(self):
        """Mark the model as trained"""
        self._is_trained = True

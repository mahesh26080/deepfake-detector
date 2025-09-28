import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
from datetime import datetime
from data_preparation import DataPreparator
from deepfake_detector import DeepfakeDetector
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Accuracy, Precision, Recall

class ModelTrainer:
    def __init__(self, data_dir: str = "processed_data", model_save_dir: str = "models"):
        """
        Initialize model trainer
        
        Args:
            data_dir: Directory containing processed data
            model_save_dir: Directory to save trained models
        """
        self.data_dir = data_dir
        self.model_save_dir = model_save_dir
        os.makedirs(model_save_dir, exist_ok=True)
        
        self.detector = DeepfakeDetector()
        self.history = None
        self.test_results = None
    
    def prepare_data(self, create_sample: bool = True):
        """Prepare data for training"""
        preparator = DataPreparator()
        
        if create_sample and not os.path.exists(self.data_dir):
            print("Creating sample dataset...")
            preparator.create_sample_dataset(num_real=500, num_fake=500)
            preparator.output_dir = self.data_dir
        
        # Get data generators
        train_gen, val_gen, test_gen = preparator.create_data_generators(
            batch_size=32,
            image_size=(224, 224)
        )
        
        return train_gen, val_gen, test_gen
    
    def train_model(self, epochs: int = 50, learning_rate: float = 0.001):
        """
        Train the deepfake detection model
        
        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
        """
        print("Preparing data...")
        train_gen, val_gen, test_gen = self.prepare_data()
        
        print(f"Training samples: {train_gen.samples}")
        print(f"Validation samples: {val_gen.samples}")
        print(f"Test samples: {test_gen.samples}")
        
        # Compile model with new learning rate
        self.detector.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=BinaryCrossentropy(),
            metrics=[Accuracy(), Precision(), Recall()]
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.model_save_dir, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        print("Starting training...")
        self.history = self.detector.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        final_model_path = os.path.join(self.model_save_dir, 'final_model.h5')
        self.detector.save_model(final_model_path)
        
        print(f"Training completed! Model saved to {final_model_path}")
        
        return self.history
    
    def evaluate_model(self, test_gen):
        """
        Evaluate the trained model on test data
        
        Args:
            test_gen: Test data generator
        """
        print("Evaluating model on test data...")
        
        # Get predictions
        predictions = self.detector.model.predict(test_gen, verbose=1)
        y_pred = (predictions > 0.5).astype(int).flatten()
        y_true = test_gen.classes
        
        # Calculate metrics
        test_loss, test_accuracy, test_precision, test_recall = self.detector.model.evaluate(
            test_gen, verbose=1
        )
        
        # Classification report
        class_names = ['Real', 'Fake']
        report = classification_report(
            y_true, y_pred, 
            target_names=class_names, 
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        self.test_results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': predictions.flatten().tolist(),
            'true_labels': y_true.tolist()
        }
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        
        return self.test_results
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 1].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, save_path: str = None):
        """Plot confusion matrix"""
        if self.test_results is None:
            print("No test results available")
            return
        
        cm = np.array(self.test_results['confusion_matrix'])
        class_names = ['Real', 'Fake']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix plot saved to {save_path}")
        
        plt.show()
    
    def save_results(self, results_path: str = None):
        """Save training and evaluation results"""
        if results_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = os.path.join(self.model_save_dir, f'training_results_{timestamp}.json')
        
        results = {
            'training_history': self.history.history if self.history else None,
            'test_results': self.test_results,
            'model_info': {
                'input_shape': self.detector.model.input_shape,
                'output_shape': self.detector.model.output_shape,
                'total_params': self.detector.model.count_params(),
                'trainable_params': sum([tf.keras.backend.count_params(w) for w in self.detector.model.trainable_weights])
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to {results_path}")
        return results_path

def main():
    """Main training function"""
    print("Starting Deepfake Detection Model Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Train model
    print("Training model...")
    history = trainer.train_model(epochs=30, learning_rate=0.001)
    
    # Prepare test data
    print("Preparing test data...")
    _, _, test_gen = trainer.prepare_data(create_sample=False)
    
    # Evaluate model
    print("Evaluating model...")
    test_results = trainer.evaluate_model(test_gen)
    
    # Plot results
    print("Generating plots...")
    trainer.plot_training_history(os.path.join(trainer.model_save_dir, 'training_history.png'))
    trainer.plot_confusion_matrix(os.path.join(trainer.model_save_dir, 'confusion_matrix.png'))
    
    # Save results
    results_path = trainer.save_results()
    
    print("Training completed successfully!")
    print(f"Model saved to: {trainer.model_save_dir}")
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()


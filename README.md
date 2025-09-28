# Deepfake Detection AI

A comprehensive deepfake detection system using deep learning, built with TensorFlow, OpenCV, and Flask. This project implements a CNN-based model using transfer learning with EfficientNetB0 for detecting manipulated images and videos.

## 🚀 Features

- **AI-Powered Detection**: Uses state-of-the-art CNN models with transfer learning
- **Multi-Format Support**: Analyzes both images (JPG, PNG) and videos (MP4, AVI, MOV, MKV, WebM)
- **Web Interface**: Modern, responsive web application for easy file upload and analysis
- **Video Analysis**: Frame-by-frame analysis with temporal consistency checking
- **Image Forensics**: Pixel-level artifact detection and analysis
- **Real-time Results**: Instant analysis with confidence scores and detailed reports
- **Training Pipeline**: Complete data preparation and model training utilities

## 🛠️ Tech Stack

- **Backend**: Python, Flask, TensorFlow 2.13
- **AI/ML**: EfficientNetB0, Transfer Learning, CNN
- **Computer Vision**: OpenCV, PIL
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Data Processing**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn

## 📁 Project Structure

```
deepfake-detection-ai/
├── app.py                      # Flask web application
├── deepfake_detector.py        # Core detection model
├── data_preparation.py         # Data preprocessing utilities
├── train_model.py             # Model training and evaluation
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── templates/
│   └── index.html             # Web interface template
├── static/
│   ├── css/
│   │   └── style.css          # Custom styles
│   ├── js/
│   │   └── app.js             # Frontend JavaScript
│   └── results/               # Analysis results storage
├── models/                    # Trained model storage
├── processed_data/            # Processed training data
└── uploads/                   # Temporary file uploads
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd deepfake-detection-ai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application

```bash
# Start the Flask web server
python3 app.py
```

The application will be available at `http://localhost:5000`

### 3. Train Your Own Model (Optional)

```bash
# Create sample dataset and train model
python3 train_model.py
```

## 🎯 Usage

### Web Interface

1. **Upload Media**: Drag and drop or click to upload images or videos
2. **Analysis**: The system automatically analyzes the uploaded content
3. **Results**: View detailed analysis results including:
   - Deepfake probability scores
   - Confidence levels
   - Frame-by-frame analysis (for videos)
   - Visual confidence indicators

### Programmatic Usage

```python
from deepfake_detector import DeepfakeDetector

# Initialize detector
detector = DeepfakeDetector()

# Detect in image
result = detector.detect_image("path/to/image.jpg")
print(f"Is deepfake: {result['is_deepfake']}")
print(f"Confidence: {result['confidence']}")

# Detect in video
result = detector.detect_video("path/to/video.mp4")
print(f"Is deepfake: {result['is_deepfake']}")
print(f"Average confidence: {result['average_confidence']}")
```

## 🧠 AI Model Details

### Architecture
- **Base Model**: EfficientNetB0 (pre-trained on ImageNet)
- **Transfer Learning**: Frozen base layers + custom classification head
- **Input Size**: 224x224x3 RGB images
- **Output**: Binary classification (Real/Fake) with confidence scores

### Training Process
1. **Data Preparation**: Frame extraction from videos, image preprocessing
2. **Data Augmentation**: Rotation, shifting, flipping, zooming
3. **Transfer Learning**: Fine-tuning on deepfake detection task
4. **Evaluation**: Cross-validation with multiple metrics

### Performance Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate for fake detection
- **Recall**: Sensitivity for fake detection
- **F1-Score**: Harmonic mean of precision and recall

## 📊 Data Requirements

### Training Data Structure
```
data/
├── real/
│   ├── images/          # Real images
│   └── videos/          # Real videos
└── fake/
    ├── images/          # Fake/manipulated images
    └── videos/          # Fake/manipulated videos
```

### Supported Formats
- **Images**: JPG, JPEG, PNG
- **Videos**: MP4, AVI, MOV, MKV, WebM

## 🔧 Configuration

### Model Parameters
- **Input Size**: 224x224 pixels
- **Batch Size**: 32 (configurable)
- **Learning Rate**: 0.001 (with decay)
- **Epochs**: 50 (with early stopping)

### Detection Thresholds
- **Default Threshold**: 0.5
- **Confidence Range**: 0.0 (Real) to 1.0 (Fake)

## 📈 Training Your Own Model

### 1. Prepare Data
```python
from data_preparation import DataPreparator

preparator = DataPreparator()
preparator.process_image_dataset("data/real/images", "data/fake/images")
preparator.process_video_dataset("data/real/videos", "data/fake/videos")
```

### 2. Train Model
```python
from train_model import ModelTrainer

trainer = ModelTrainer()
trainer.train_model(epochs=50, learning_rate=0.001)
```

### 3. Evaluate Performance
```python
# Model evaluation is automatically performed after training
# Results are saved to models/training_results_*.json
```

## 🎨 Web Interface Features

### Upload Area
- Drag & drop file upload
- File format validation
- File size limits (100MB)
- Visual feedback

### Results Display
- Real-time analysis results
- Confidence visualization
- Frame-by-frame breakdown (videos)
- Downloadable reports

### Responsive Design
- Mobile-friendly interface
- Bootstrap 5 styling
- Interactive elements
- Progress indicators

## 🔍 Detection Methods

### Image Analysis
- **Pixel-level Artifacts**: Detection of manipulation traces
- **Compression Artifacts**: Analysis of JPEG/PNG compression
- **Color Space Analysis**: RGB channel inconsistencies
- **Edge Detection**: Unnatural edge patterns

### Video Analysis
- **Temporal Consistency**: Frame-to-frame analysis
- **Motion Artifacts**: Unnatural movement patterns
- **Compression Analysis**: Video codec inconsistencies
- **Sampling Strategy**: Intelligent frame selection

## ⚠️ Limitations

- **Model Accuracy**: Performance depends on training data quality
- **Computational Requirements**: GPU recommended for training
- **File Size Limits**: Large videos may take longer to process
- **False Positives**: May occasionally misclassify authentic content

## 🚀 Future Enhancements

- [ ] Real-time video stream analysis
- [ ] Mobile app development
- [ ] Advanced ensemble methods
- [ ] Temporal consistency models
- [ ] Audio analysis integration
- [ ] Cloud deployment options

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- EfficientNet architecture by Google Research
- TensorFlow team for the deep learning framework
- OpenCV community for computer vision tools
- Flask team for the web framework

## 📞 Support

For questions, issues, or contributions, please:
1. Check the existing issues
2. Create a new issue with detailed description
3. Contact the maintainers

---

**Note**: This tool is for educational and research purposes. Always verify results through multiple methods for critical decisions.

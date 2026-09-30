# DeepGuard — Deep Learning-Based Deepfake Detection System

## 1. Project Title
DeepGuard — Deep Learning-Based Deepfake Detection System

## 2. Problem Statement
With the rise of sophisticated AI models, generating highly realistic fake images and videos (deepfakes) has become easier than ever. These deepfakes pose significant threats to society, including misinformation, identity theft, and fraud. A reliable and automated system is required to accurately detect deepfakes to mitigate these risks.

## 3. Motivation
As an AI and Data Science student, tackling real-world problems using computer vision and deep learning is a core interest. Building an end-to-end deepfake detection system demonstrates a comprehensive understanding of dataset processing, transfer learning, inference optimization, and model explainability.

## 4. Objectives
- Detect faces in images and videos.
- Preprocess images for neural network compatibility.
- Classify visual media as REAL or DEEPFAKE using a deep learning model.
- Provide a confidence score for predictions.
- Offer visual explainability using Grad-CAM.
- Deploy the solution in a modern, user-friendly Streamlit web interface.

## 5. Features
- **Image Detection**: Predict whether an uploaded image is real or manipulated.
- **Video Detection**: Sample frames from videos, analyze them, and aggregate predictions to classify the entire video.
- **Visual Explainability**: Display Grad-CAM heatmaps to highlight regions that contributed to the model's decision.
- **Hardware Acceleration**: Automatic fallback across CUDA, MPS (Apple Silicon), and CPU.

## 6. System Architecture
1. **Video Frame Extraction**: Extract a configurable number of frames from input videos.
2. **Face Detection**: Localize and crop faces using MTCNN.
3. **Preprocessing**: Resize faces to 224x224 and normalize using ImageNet statistics.
4. **Model Inference**: Pass preprocessed faces through a fine-tuned EfficientNet-B0 backbone.
5. **Aggregation (for videos)**: Average the prediction probabilities across sampled frames.
6. **Explainability**: Generate Grad-CAM heatmaps for specific frames/images.
7. **Streamlit UI**: Display results, confidence scores, and visual explanations in an interactive dashboard.

## 7. Technologies
- **Python 3.11+**
- **Deep Learning**: PyTorch, torchvision
- **Computer Vision**: OpenCV, Pillow, facenet-pytorch (MTCNN)
- **Data Science**: NumPy, Pandas, scikit-learn, Matplotlib, Seaborn
- **Explainability**: Grad-CAM
- **Web App**: Streamlit

## 8. Dataset
This project is designed to be trained on datasets like **FaceForensics++** or **Celeb-DF**.
*Note: Datasets must be downloaded independently and placed in the `data/raw` directory.*

## 9. Dataset Preparation
The dataset preparation pipeline involves:
1. Video level splitting into train, validation, and test sets to prevent data leakage.
2. Frame extraction.
3. Face detection and cropping.
4. Image normalization and resizing.

## 10. Model Architecture
- **Backbone**: EfficientNet-B0 (pretrained on ImageNet).
- **Head**: Custom fully connected layers modified for binary classification (REAL vs. DEEPFAKE).
- **Transfer Learning**: The backbone is initially frozen to train the classifier head, followed by potential fine-tuning.

## 11. Training Process
- Optimizer: AdamW
- Loss Function: CrossEntropyLoss
- Callbacks: Early Stopping, Model Checkpointing
- Imbalance handling: Weighted loss or WeightedRandomSampler.

## 12. Evaluation
Evaluated on an unseen test set using Accuracy, Precision, Recall, F1-score, and ROC-AUC.

## 13. Results
*To be added after training.*

## 14. Explainability
Uses **Grad-CAM** to highlight the regions (e.g., artifacts around the mouth or eyes) that the model focused on to make its prediction. This is an estimate and not definitive forensic proof.

## 15. Streamlit Application
A modern dashboard allowing users to upload media, view results, explore frame-by-frame analyses for videos, and understand the model architecture.

## 16. Installation
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# macOS / Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 17. Usage
*Usage instructions will be added once scripts are implemented.*

## 18. Project Structure
*(To be detailed)*

## 19. Limitations
- **Dataset Bias**: The model may not generalize to unseen manipulation techniques.
- **Compression Artifacts**: Heavy compression on social media might degrade performance.
- Not intended as definitive forensic evidence.

## 20. Future Improvements
- Implement temporal analysis (e.g., LSTM/Transformer) for video sequences.
- Enhance robustness against adversarial attacks.

## 21. Ethical Considerations
This project is an **educational deepfake detection system for research and demonstration**. It should not be treated as providing forensic certainty. We acknowledge the potential for false positives/negatives and the importance of responsible use.

## 22. References
- EfficientNet Paper
- Grad-CAM Paper
- FaceForensics++ Dataset

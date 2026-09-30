import pytest
import torch
import numpy as np
from PIL import Image
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.face_detection import FaceDetector
from src.model import DeepfakeClassifier
from src.inference import ImagePredictor

def test_config_loads():
    assert "dataset" in config
    assert "training" in config
    assert "model" in config

def test_face_detector_init():
    detector = FaceDetector(device=torch.device('cpu'))
    assert detector.device.type == 'cpu'

def test_face_detection_no_face():
    detector = FaceDetector(device=torch.device('cpu'))
    # Create a blank image (no face)
    img = Image.new("RGB", (800, 600), color="black")
    
    face = detector.detect_and_crop(img)
    assert face is None

def test_model_output_shape():
    model = DeepfakeClassifier(num_classes=2, pretrained=False)
    model.eval()
    
    # EfficientNet takes 224x224
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
        
    assert output.shape == (1, 2)

def test_image_predictor_initialization():
    # It should fallback to random predictions if checkpoint not found, but shouldn't crash
    predictor = ImagePredictor(model_path="non_existent_model.pth")
    assert predictor.model is not None

def test_image_predictor_inference():
    predictor = ImagePredictor(model_path="non_existent_model.pth")
    
    # Test with dummy image
    img = Image.new("RGB", (224, 224), color="white")
    
    # It will fail face detection
    result = predictor.predict_image(img)
    
    assert "error" in result
    assert "No face detected" in result["error"]

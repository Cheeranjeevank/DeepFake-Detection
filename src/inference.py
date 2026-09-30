import torch
from PIL import Image
import numpy as np
from pathlib import Path

from src.config import config
from src.utils import logger
from src.preprocessing import get_val_transforms
from src.face_detection import FaceDetector
from src.model import DeepfakeClassifier
from src.train import get_device

class ImagePredictor:
    def __init__(self, model_path=None):
        self.device = get_device()
        self.face_detector = FaceDetector(device=self.device)
        
        self.image_size = config["training"]["image_size"]
        self.transforms = get_val_transforms(self.image_size)
        self.classes = config["model"]["classes"]
        
        self.model = DeepfakeClassifier(num_classes=config["model"]["num_classes"])
        
        if model_path is None:
            model_path = config["training"]["best_model_path"]
            
        if Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info("Predictor initialized successfully.")
        else:
            logger.warning(f"Model checkpoint not found at {model_path}. Predictions will be random untrained outputs.")
            self.model = self.model.to(self.device)
            self.model.eval()

    def predict_image(self, image_path_or_pil):
        """
        Predicts whether an image is REAL or DEEPFAKE.
        """
        if isinstance(image_path_or_pil, (str, Path)):
            try:
                image = Image.open(image_path_or_pil).convert("RGB")
            except Exception as e:
                logger.error(f"Error opening image: {e}")
                return {"error": str(e)}
        else:
            image = image_path_or_pil

        # 1. Detect face (with fallback to center crop)
        face, face_detected = self.face_detector.detect_and_crop(image)

        # 2. Preprocess
        input_tensor = self.transforms(face).unsqueeze(0).to(self.device)

        # 3. Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        # 4. Format Output
        real_prob = float(probs[0])
        fake_prob = float(probs[1])

        if fake_prob > 0.5:
            prediction = "DEEPFAKE"
            confidence = fake_prob
        else:
            prediction = "REAL"
            confidence = real_prob

        result = {
            "prediction": prediction,
            "confidence": confidence,
            "real_probability": real_prob,
            "fake_probability": fake_prob,
            "face_crop": face,
            "face_detected": face_detected,
        }

        if not face_detected:
            result["warning"] = (
                "No face was detected — the full image (center-cropped) was analyzed instead. "
                "Results may be less accurate."
            )

        return result

if __name__ == "__main__":
    predictor = ImagePredictor()
    # Test with a dummy image
    dummy_img = Image.new("RGB", (800, 600))
    result = predictor.predict_image(dummy_img)
    print(f"Prediction result: {result}")

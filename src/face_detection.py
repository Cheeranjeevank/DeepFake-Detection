import torch
import cv2
import numpy as np
from PIL import Image
try:
    from facenet_pytorch import MTCNN
except ImportError:
    MTCNN = None
from src.utils import logger

class FaceDetector:
    def __init__(self, device=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
            
        if MTCNN is not None:
            # MTCNN must run on CPU — MPS has an adaptive pool bug for certain image sizes
            # (pytorch/pytorch#96056). The classifier model can still use MPS/CUDA.
            self.detector = MTCNN(keep_all=False, device=torch.device('cpu'))
        else:
            logger.error("facenet_pytorch not installed. Face detection will fail.")
            self.detector = None

    def detect_and_crop(self, image):
        """
        Detects the primary face and returns the cropped face image.
        image: PIL Image or NumPy array (RGB)
        Returns: (PIL Image of cropped face or fallback crop, face_detected: bool)
        """
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image

        if self.detector is not None:
            try:
                boxes, probs = self.detector.detect(image_pil)

                if boxes is not None and len(boxes) > 0:
                    box = boxes[0]
                    width, height = image_pil.size
                    x1, y1, x2, y2 = box

                    # Expand bounding box slightly for context
                    w = x2 - x1
                    h = y2 - y1
                    x1 = max(0, x1 - 0.2 * w)
                    y1 = max(0, y1 - 0.2 * h)
                    x2 = min(width, x2 + 0.2 * w)
                    y2 = min(height, y2 + 0.2 * h)

                    face = image_pil.crop((x1, y1, x2, y2))
                    return face, True

            except Exception as e:
                logger.warning(f"Face detection failed, using full image fallback: {e}")

        # ── Fallback: center-square crop of the full image ──
        logger.warning("No face detected — falling back to center-cropped full image.")
        width, height = image_pil.size
        side = min(width, height)
        left  = (width  - side) // 2
        top   = (height - side) // 2
        fallback = image_pil.crop((left, top, left + side, top + side))
        return fallback, False

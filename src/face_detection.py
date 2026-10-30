import torch
import numpy as np
from PIL import Image, ImageEnhance
try:
    from facenet_pytorch import MTCNN
except ImportError:
    MTCNN = None
from src.utils import logger


class FaceDetector:
    """
    Multi-strategy face detector using MTCNN with cascaded fallbacks.

    Detection strategy (in order):
      1. MTCNN standard pass (threshold 0.85)
      2. MTCNN relaxed pass (lower thresholds, catches harder faces)
      3. MTCNN on upscaled image (catches small/distant faces)
      4. MTCNN on contrast-enhanced image (helps with poor lighting)
      5. Center-square crop of the full image (last resort)
    """

    # Confidence thresholds for the two MTCNN passes
    _STRICT_THRESHOLDS  = [0.6, 0.7, 0.85]
    _RELAXED_THRESHOLDS = [0.3, 0.5, 0.6]

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

        if MTCNN is None:
            logger.error("facenet_pytorch not installed. Face detection will fail.")
            self._strict = None
            self._relaxed = None
            return

        # MTCNN must run on CPU — MPS has an adaptive pool bug for certain image sizes
        # (pytorch/pytorch#96056). The classifier model can still use MPS/CUDA.
        cpu = torch.device('cpu')

        # Pass 1 — strict: high-precision, fewer false positives
        self._strict = MTCNN(
            keep_all=True,
            device=cpu,
            thresholds=self._STRICT_THRESHOLDS,
            min_face_size=20,         # detect faces as small as 20px
            post_process=False,
        )
        # Pass 2 — relaxed: lower thresholds for harder cases
        self._relaxed = MTCNN(
            keep_all=True,
            device=cpu,
            thresholds=self._RELAXED_THRESHOLDS,
            min_face_size=15,
            post_process=False,
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _best_box(self, boxes, probs, min_conf: float = 0.5):
        """Return the highest-confidence box above min_conf, or None."""
        if boxes is None or len(boxes) == 0:
            return None, None
        best_idx = int(np.argmax(probs))
        if probs[best_idx] < min_conf:
            return None, None
        return boxes[best_idx], float(probs[best_idx])

    def _crop_box(self, image_pil: Image.Image, box, margin: float = 0.25):
        """
        Crop image_pil to the given MTCNN bounding box, adding a margin
        proportional to the box dimensions for contextual information.
        """
        w_img, h_img = image_pil.size
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1

        # Symmetric margin
        x1 = max(0,     x1 - margin * bw)
        y1 = max(0,     y1 - margin * bh)
        x2 = min(w_img, x2 + margin * bw)
        y2 = min(h_img, y2 + margin * bh)

        return image_pil.crop((x1, y1, x2, y2))

    def _try_detect(self, detector, image_pil: Image.Image, min_conf: float = 0.5):
        """Run one MTCNN detector pass and return (box, conf) or (None, None)."""
        try:
            boxes, probs = detector.detect(image_pil)
            return self._best_box(boxes, probs, min_conf)
        except Exception as e:
            logger.debug(f"MTCNN detect exception: {e}")
            return None, None

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect_and_crop(self, image, margin: float = 0.25):
        """
        Detects the primary face using a multi-strategy cascade and returns
        the cropped face image with contextual margin.

        Args:
            image:  PIL Image or NumPy array (RGB).
            margin: Fractional margin to add around the detected bounding box.

        Returns:
            (PIL Image of cropped face or fallback crop, face_detected: bool)
        """
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image

        if self._strict is None:
            return self._fallback(image_pil)

        w_img, h_img = image_pil.size

        # ── Strategy 1: strict MTCNN pass ────────────────────────────────────
        box, conf = self._try_detect(self._strict, image_pil, min_conf=0.85)
        if box is not None:
            logger.debug(f"Face detected (strict) conf={conf:.3f}")
            return self._crop_box(image_pil, box, margin), True

        # ── Strategy 2: relaxed MTCNN pass ───────────────────────────────────
        box, conf = self._try_detect(self._relaxed, image_pil, min_conf=0.50)
        if box is not None:
            logger.debug(f"Face detected (relaxed) conf={conf:.3f}")
            return self._crop_box(image_pil, box, margin), True

        # ── Strategy 3: upscale small images and retry ───────────────────────
        if max(w_img, h_img) < 640:
            scale = 640 / max(w_img, h_img)
            upscaled = image_pil.resize(
                (int(w_img * scale), int(h_img * scale)),
                Image.LANCZOS,
            )
            box, conf = self._try_detect(self._strict, upscaled, min_conf=0.80)
            if box is None:
                box, conf = self._try_detect(self._relaxed, upscaled, min_conf=0.45)
            if box is not None:
                # Scale box coordinates back to original size
                box = [v / scale for v in box]
                logger.debug(f"Face detected (upscale) conf={conf:.3f}")
                return self._crop_box(image_pil, box, margin), True

        # ── Strategy 4: contrast enhancement and retry ───────────────────────
        try:
            enhanced = ImageEnhance.Contrast(image_pil).enhance(1.5)
            box, conf = self._try_detect(self._relaxed, enhanced, min_conf=0.45)
            if box is not None:
                logger.debug(f"Face detected (enhanced) conf={conf:.3f}")
                return self._crop_box(image_pil, box, margin), True
        except Exception:
            pass

        # ── Strategy 5: center-square crop fallback ───────────────────────────
        return self._fallback(image_pil)

    def _fallback(self, image_pil: Image.Image):
        """Return center-square crop as a last resort."""
        logger.warning("No face detected — falling back to center-cropped full image.")
        w, h = image_pil.size
        side  = min(w, h)
        left  = (w - side) // 2
        top   = (h - side) // 2
        return image_pil.crop((left, top, left + side, top + side)), False

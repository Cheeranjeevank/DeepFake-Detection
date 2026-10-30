"""
DeepGuard FastAPI Backend — Real-Time Deepfake Detection
Uses InceptionNet + GRU (TensorFlow) trained on FaceForensics++ dataset.
Provides REST endpoints for image upload, video upload, and live webcam streaming.
"""
import sys
import io
import os
import base64
import tempfile
from pathlib import Path

os.environ['TF_USE_LEGACY_KERAS'] = '1'

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from facenet_pytorch import MTCNN as _MTCNN
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import logger

# MTCNN face detector (CPU)
_mtcnn = _MTCNN(keep_all=False, device=torch.device('cpu'), post_process=False)

app = FastAPI(title="DeepGuard API — Real-Time", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ── Constants ──────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 20
IMAGE_SIZE = 299  # InceptionV3 native input

# ── Model Loading ──────────────────────────────────────────────────────────────
_gru_model = None     # The trained GRU classifier  
_iv3_extractor = None # InceptionV3 feature extractor
_model_error = None

def get_models():
    global _gru_model, _iv3_extractor, _model_error
    if _gru_model is None and _model_error is None:
        try:
            import tensorflow as tf
            # 1. Load the trained GRU classifier
            gru_path = str(Path(__file__).resolve().parent.parent / "models" / "inceptionNet_model.h5")
            _gru_model = tf.keras.models.load_model(gru_path)
            logger.info(f"GRU model loaded: input={[i.shape for i in _gru_model.inputs]}")

            # 2. Build the InceptionV3 feature extractor (ImageNet weights)
            _iv3_extractor = tf.keras.applications.InceptionV3(
                include_top=False, pooling='avg',
                input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                weights='imagenet'
            )
            logger.info("InceptionV3 feature extractor ready.")
        except Exception as e:
            _model_error = str(e)
            logger.error(f"Failed to load models: {e}")
    return _gru_model, _iv3_extractor, _model_error

# ── Frame Buffer for real-time streaming (per session, server-side) ────────────
_realtime_buffer: list = []

# ── Helper Functions ───────────────────────────────────────────────────────────

def detect_and_crop_face(rgb_frame: np.ndarray):
    """
    Detect face using MTCNN and return cropped RGB numpy array, or None.
    Falls back to center crop if no face detected.
    """
    try:
        pil_img = Image.fromarray(rgb_frame)
        boxes, _ = _mtcnn.detect(pil_img)
        if boxes is not None and len(boxes) > 0:
            x1, y1, x2, y2 = [int(max(0, v)) for v in boxes[0]]
            face_img = rgb_frame[y1:y2, x1:x2]
            if face_img.size > 0:
                return face_img
    except Exception:
        pass
    return None

def preprocess_face(face_rgb: np.ndarray) -> np.ndarray:
    """Resize face to IMAGE_SIZE x IMAGE_SIZE."""
    return cv2.resize(face_rgb, (IMAGE_SIZE, IMAGE_SIZE))

def run_sequence_inference(frames_array: np.ndarray):
    """
    Two-stage inference:
    1. Extract InceptionV3 features for each of the 20 face frames.
    2. Pass the (20, 2048) sequence + mask to the GRU classifier.
    Returns (label, confidence_pct, fake_pct, real_pct).
    """
    import tensorflow as tf
    gru_model, iv3, _ = get_models()
    if gru_model is None or iv3 is None:
        return None, None, None, None

    n = len(frames_array)

    # Preprocess frames for InceptionV3
    frames_float = frames_array.astype(np.float32)
    preprocessed = tf.keras.applications.inception_v3.preprocess_input(frames_float)  # [-1, 1]

    # Extract features: (n, 2048)
    features = iv3.predict(preprocessed, verbose=0)

    # Pad / truncate to exactly SEQUENCE_LENGTH
    if n < SEQUENCE_LENGTH:
        pad = np.zeros((SEQUENCE_LENGTH - n, 2048), dtype=np.float32)
        features = np.vstack([features, pad])
        mask = np.array([1]*n + [0]*(SEQUENCE_LENGTH - n), dtype=np.float32)
    else:
        features = features[:SEQUENCE_LENGTH]
        mask = np.ones(SEQUENCE_LENGTH, dtype=np.float32)

    # Add batch dimension
    features_batch = np.expand_dims(features, axis=0)  # (1, 20, 2048)
    mask_batch = np.expand_dims(mask, axis=0)          # (1, 20)

    pred = gru_model.predict([features_batch, mask_batch], verbose=0)

    # Output is (1, 2) — softmax over [REAL, FAKE] or [FAKE, REAL]?
    # We'll treat argmax as FAKE (class 1 = FAKE based on FaceForensics++ convention)
    real_prob = float(pred[0][0])
    fake_prob = float(pred[0][1])
    label = "DEEPFAKE" if fake_prob > real_prob else "REAL"
    confidence = max(fake_prob, real_prob)
    return label, round(confidence * 100, 1), round(fake_prob * 100, 1), round(real_prob * 100, 1)


def extract_frames_from_video(video_path: str, num_frames: int = SEQUENCE_LENGTH):
    """Sample num_frames evenly-spaced frames with face detection from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return None
    step = max(1, total // num_frames)
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = detect_and_crop_face(rgb)
        if face is not None:
            try:
                frames.append(preprocess_face(face))
            except Exception:
                pass
    cap.release()
    # Pad if needed
    while 0 < len(frames) < num_frames:
        frames.append(frames[-1])
    return np.array(frames) if frames else None


def pil_to_b64(img: Image.Image, fmt="JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    index = static_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "DeepGuard API v2 running. Frontend not found."}


@app.get("/api/status")
async def status():
    _, _, err = get_models()
    if err:
        return {"status": "degraded", "message": f"Model failed to load: {err}"}
    return {"status": "ok", "message": "InceptionV3 + GRU model loaded and ready.", "model": "InceptionV3+GRU (FaceForensics++)"}


@app.post("/api/detect/image")
async def detect_image(file: UploadFile = File(...)):
    """
    Detect deepfake in a single uploaded image.
    Treats the image as a 20-frame sequence (same frame repeated) for compatibility.
    """
    _, _, err = get_models()
    if err:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {err}")

    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    rgb = np.array(pil_img)

    # Try face detection
    face = detect_and_crop_face(rgb)
    face_detected = face is not None

    if not face_detected:
        # Fallback: center crop
        h, w = rgb.shape[:2]
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        face = rgb[y0:y0+s, x0:x0+s]

    # Preprocess and build a 20-frame sequence from the single face
    face_resized = preprocess_face(face)
    frames = np.stack([face_resized] * SEQUENCE_LENGTH, axis=0)

    label, confidence, fake_pct, real_pct = run_sequence_inference(frames)
    if label is None:
        raise HTTPException(status_code=503, detail="Inference failed.")

    # Face crop as base64
    face_pil = Image.fromarray(face_resized)
    face_b64 = pil_to_b64(face_pil)

    return JSONResponse({
        "prediction": label,
        "confidence": confidence,
        "fake_probability": fake_pct,
        "real_probability": real_pct,
        "face_detected": face_detected,
        "face_crop_b64": face_b64,
        "heatmap_b64": None,
        "warning": None if face_detected else "No face detected — used full image center crop",
    })


@app.post("/api/detect/video")
async def detect_video(file: UploadFile = File(...)):
    """Detect deepfake across a video by sampling 20 face frames."""
    _, _, err = get_models()
    if err:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {err}")

    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        frames = extract_frames_from_video(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    if frames is None:
        raise HTTPException(status_code=422, detail="No faces detected in video.")

    label, confidence, fake_pct, real_pct = run_sequence_inference(frames)
    if label is None:
        raise HTTPException(status_code=503, detail="Inference failed.")

    return JSONResponse({
        "final_prediction": label,
        "confidence": confidence,
        "fake_probability": fake_pct,
        "real_probability": real_pct,
        "frames_analyzed": len(frames),
    })


# ── Real-Time Webcam Streaming ─────────────────────────────────────────────────

class FrameData(BaseModel):
    frame: str  # base64 data URL


@app.post("/api/realtime/frame")
async def analyze_realtime_frame(body: FrameData):
    """
    Accepts a base64-encoded webcam frame, detects the face, adds it to
    a rolling 20-frame buffer, and runs inference when the buffer is full.
    """
    global _realtime_buffer

    # Decode frame
    try:
        img_data = base64.b64decode(body.frame.split(',')[1])
        pil_img = Image.open(io.BytesIO(img_data)).convert("RGB")
        rgb = np.array(pil_img)
    except Exception as e:
        return JSONResponse({"error": f"Frame decode error: {e}"}, status_code=400)

    face = detect_and_crop_face(rgb)
    face_detected = face is not None

    if face_detected:
        try:
            _realtime_buffer.append(preprocess_face(face))
            if len(_realtime_buffer) > SEQUENCE_LENGTH:
                _realtime_buffer = _realtime_buffer[-SEQUENCE_LENGTH:]
        except Exception:
            pass

    if len(_realtime_buffer) >= SEQUENCE_LENGTH:
        frames_arr = np.array(_realtime_buffer[:SEQUENCE_LENGTH])
        label, confidence, fake_pct, real_pct = run_sequence_inference(frames_arr)
        return JSONResponse({
            "label": label,
            "confidence": confidence,
            "fake_probability": fake_pct,
            "real_probability": real_pct,
            "face_detected": face_detected,
            "buffer_size": len(_realtime_buffer),
            "ready": True,
        })

    return JSONResponse({
        "label": "SCANNING",
        "confidence": 0,
        "face_detected": face_detected,
        "buffer_size": len(_realtime_buffer),
        "ready": False,
    })


@app.post("/api/realtime/reset")
async def reset_buffer():
    global _realtime_buffer
    _realtime_buffer = []
    return JSONResponse({"status": "ok"})

"""
DeepGuard FastAPI Backend
Provides REST API endpoints for deepfake detection on images and videos.
"""
import sys
import io
import base64
import tempfile
import os
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference import ImagePredictor
from src.video_processor import VideoPredictor
from src.explainability import GradCAMExplainer
from src.utils import logger

app = FastAPI(title="DeepGuard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (HTML/CSS/JS frontend)
static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# --- Model loading (singleton) ---
_image_predictor = None
_video_predictor = None
_explainer = None
_models_error = None


def get_models():
    global _image_predictor, _video_predictor, _explainer, _models_error
    if _image_predictor is None and _models_error is None:
        try:
            _image_predictor = ImagePredictor()
            _video_predictor = VideoPredictor(predictor=_image_predictor)
            _explainer = GradCAMExplainer(_image_predictor.model, _image_predictor.device)
            logger.info("All models loaded successfully.")
        except Exception as e:
            _models_error = str(e)
            logger.error(f"Failed to load models: {e}")
    return _image_predictor, _video_predictor, _explainer, _models_error


def pil_to_base64(image: Image.Image, fmt="JPEG") -> str:
    """Convert PIL Image to base64 string."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# --- Routes ---

@app.get("/")
async def serve_frontend():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "DeepGuard API is running. Frontend not found."}


@app.get("/api/status")
async def status():
    """Check if models are loaded."""
    _, _, _, err = get_models()
    if err:
        return {"status": "degraded", "message": f"Models failed to load: {err}"}
    return {"status": "ok", "message": "All models loaded successfully."}


@app.post("/api/detect/image")
async def detect_image(file: UploadFile = File(...)):
    """
    Detect if an uploaded image is REAL or DEEPFAKE.
    Returns prediction, confidence, probabilities, and optional Grad-CAM heatmap.
    """
    image_predictor, _, explainer, err = get_models()

    if err:
        raise HTTPException(status_code=503, detail=f"Models unavailable: {err}")

    # Read and open image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # Run prediction
    result = image_predictor.predict_image(image)

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    # Prepare response
    response = {
        "prediction": result["prediction"],
        "confidence": round(result["confidence"] * 100, 2),
        "real_probability": round(result["real_probability"] * 100, 2),
        "fake_probability": round(result["fake_probability"] * 100, 2),
        "face_detected": result.get("face_detected", True),
        "warning": result.get("warning", None),
        "face_crop_b64": None,
        "heatmap_b64": None,
    }

    # Add face crop and Grad-CAM heatmap
    if result.get("face_crop"):
        face_crop = result["face_crop"]
        response["face_crop_b64"] = pil_to_base64(face_crop)

        if explainer is not None:
            try:
                target_class = 1 if result["prediction"] == "DEEPFAKE" else 0
                heatmap = explainer.generate_heatmap(face_crop, target_class=target_class)
                response["heatmap_b64"] = pil_to_base64(heatmap)
            except Exception as e:
                logger.warning(f"Grad-CAM failed: {e}")

    return JSONResponse(content=response)


@app.post("/api/detect/video")
async def detect_video(file: UploadFile = File(...)):
    """
    Detect if an uploaded video is REAL or DEEPFAKE by sampling frames.
    """
    _, video_predictor, _, err = get_models()

    if err:
        raise HTTPException(status_code=503, detail=f"Models unavailable: {err}")

    # Save video to temp file
    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = video_predictor.predict_video(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    # Prepare frame results (include base64 face crops for suspicious frames)
    frame_results_out = []
    for f in result["frame_results"]:
        face_b64 = None
        if f.get("face_crop") and f["prediction"] == "DEEPFAKE":
            try:
                face_b64 = pil_to_base64(f["face_crop"])
            except Exception:
                pass

        frame_results_out.append({
            "frame_num": f["frame_num"],
            "timestamp": round(f["timestamp"], 2),
            "prediction": f["prediction"],
            "fake_probability": round(f["fake_probability"] * 100, 2),
            "real_probability": round(f["real_probability"] * 100, 2),
            "face_crop_b64": face_b64,
        })

    return JSONResponse(content={
        "final_prediction": result["final_prediction"],
        "confidence": round(result["confidence"] * 100, 2),
        "frames_analyzed": result["frames_analyzed"],
        "suspicious_frames": result["suspicious_frames"],
        "mean_fake_probability": round(result["mean_fake_probability"] * 100, 2),
        "mean_real_probability": round(result["mean_real_probability"] * 100, 2),
        "frame_results": frame_results_out,
    })

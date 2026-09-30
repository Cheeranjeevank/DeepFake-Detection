# 🛡️ DeepGuard — Deepfake Detection System

An AI-powered deepfake detection web application using **EfficientNet-B0**, **MTCNN** face detection, and **Grad-CAM** explainability — wrapped in a modern dark-mode HTML/CSS/JS frontend powered by a **FastAPI** backend.

---

## 🚀 Quick Start (any system)

```bash
# 1. Clone the repo
git clone https://github.com/Cheeranjeevank/DeepFake-Detection.git
cd DeepFake-Detection

# 2. Run (auto-downloads model weights + creates venv on first run)
chmod +x run_web.sh
./run_web.sh

# 3. Open in browser
#    http://localhost:8000
```

> **`run_web.sh` does everything automatically:**  
> ✅ Downloads the trained model weights from GitHub Releases (16 MB)  
> ✅ Creates a Python virtual environment  
> ✅ Installs all dependencies  
> ✅ Starts the web server  

---

## 📋 Requirements

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| pip | any recent |
| curl or wget | pre-installed on macOS/Linux |
| RAM | ≥ 4 GB |
| GPU | Optional (CUDA/MPS) — CPU works fine |

> **Windows users:** Run the manual steps below instead of `run_web.sh`

---

## 🖥️ Manual Setup (Windows / advanced)

```bash
# Create & activate virtual environment
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download model weights
mkdir models
curl -L -o models/best_model.pth \
  https://github.com/Cheeranjeevank/DeepFake-Detection/releases/download/v1.0.0/best_model.pth

# Start the server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Open http://localhost:8000
```

---

## ✨ Features

| Feature | Details |
|---|---|
| 🖼️ **Image Detection** | Upload JPG/PNG → instant REAL/DEEPFAKE verdict |
| 🎥 **Video Analysis** | Samples 20 frames → per-frame results + aggregate |
| 🔍 **Grad-CAM Heatmaps** | Highlights which face regions triggered the prediction |
| 📊 **Probability Bars** | Granular real vs fake probabilities |
| ⚠️ **No-face Fallback** | Falls back to center-crop if no face is detected |
| ⚡ **Hardware Auto-detect** | Uses CUDA → MPS (Apple Silicon) → CPU automatically |

---

## 🏗️ System Architecture

```
Browser (HTML/CSS/JS)
        │  upload image/video
        ▼
FastAPI Backend (app/main.py)
        │
        ├── FaceDetector (MTCNN on CPU)
        │       └── detect & crop face
        ├── DeepfakeClassifier (EfficientNet-B0 on GPU/CPU)
        │       └── REAL / DEEPFAKE + probabilities
        └── GradCAMExplainer
                └── heatmap overlaid on face crop
```

---

## 📁 Project Structure

```
DeepFake-Detection/
├── app/
│   ├── main.py              ← FastAPI backend (REST API)
│   ├── streamlit_app.py     ← Original Streamlit UI (legacy)
│   └── static/
│       ├── index.html       ← Frontend
│       ├── style.css        ← Dark-mode design system
│       └── app.js           ← Upload, results, heatmap logic
├── src/
│   ├── model.py             ← EfficientNet-B0 classifier
│   ├── face_detection.py    ← MTCNN face detector
│   ├── inference.py         ← Image predictor
│   ├── video_processor.py   ← Video frame sampler
│   ├── explainability.py    ← Grad-CAM heatmap generator
│   ├── train.py             ← Training pipeline
│   ├── evaluate.py          ← Evaluation metrics
│   └── preprocessing.py     ← Transforms
├── models/
│   └── best_model.pth       ← ⬇️ Auto-downloaded by run_web.sh
├── config.yaml              ← All hyperparameters
├── requirements.txt         ← Python dependencies
└── run_web.sh               ← One-command launcher
```

---

## 🧠 Model Details

- **Backbone**: EfficientNet-B0 (pretrained on ImageNet)
- **Head**: Dropout(0.3) → Linear(1280 → 2)
- **Training**: AdamW, CrossEntropyLoss, Early Stopping
- **Input**: 224×224 RGB, ImageNet normalization
- **Classes**: `REAL` (0), `DEEPFAKE` (1)
- **Dataset**: Designed for FaceForensics++ / Celeb-DF / Real-vs-Fake

---

## ⚠️ Disclaimer

DeepGuard is an **educational research project**. It provides AI-based estimates and should **not** be treated as definitive forensic evidence. Results may vary based on image compression, lighting, and manipulation technique.

---

## 📄 References

- [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946)
- [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)
- [FaceForensics++ Dataset](https://github.com/ondyari/FaceForensics)

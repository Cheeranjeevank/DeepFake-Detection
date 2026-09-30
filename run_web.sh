#!/bin/bash
# DeepGuard Web App Launcher
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_PATH="models/best_model.pth"
MODEL_URL="https://github.com/Cheeranjeevank/DeepFake-Detection/releases/download/v1.0.0/best_model.pth"

# ── Auto-download model weights if missing ──
if [ ! -f "$MODEL_PATH" ]; then
  echo "⬇️  Model weights not found. Downloading from GitHub Releases..."
  mkdir -p models
  if command -v curl &>/dev/null; then
    curl -L --progress-bar -o "$MODEL_PATH" "$MODEL_URL"
  elif command -v wget &>/dev/null; then
    wget --show-progress -O "$MODEL_PATH" "$MODEL_URL"
  else
    echo "❌ Error: neither curl nor wget found. Please download manually:"
    echo "   $MODEL_URL"
    echo "   → save to: $MODEL_PATH"
    exit 1
  fi
  echo "✅ Model weights downloaded."
fi

# ── Check venv ──
if [ ! -f ".venv/bin/python" ]; then
  echo "📦 Virtual environment not found. Creating..."
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt --quiet
  echo "✅ Dependencies installed."
fi

echo ""
echo "🛡️  Starting DeepGuard Web App..."
echo "📍  Open in browser: http://localhost:8000"
echo ""

.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

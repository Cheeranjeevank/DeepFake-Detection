#!/bin/bash
# DeepGuard Web App Launcher
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🛡️  Starting DeepGuard Web App..."
echo "📍  http://localhost:8000"
echo ""

.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

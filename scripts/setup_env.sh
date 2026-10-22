#!/usr/bin/env bash
# DeepFusionGuard — local (CPU-only) environment setup.
# Safe to re-run: every step skips work that is already done.
# Usage (from the repo root):  bash scripts/setup_env.sh
set -euo pipefail

PY_VERSION="3.12"
VENV_DIR=".venv"

echo "== 1. System tools =="
python3 --version || echo "system python3 not found (fine: uv provides its own)"
git --version
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg missing -> installing with apt (needs sudo)"
    sudo apt update && sudo apt install -y ffmpeg
fi
ffmpeg -version | head -n 1

echo "== 2. uv (Python + package manager) =="
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo "== 3. Virtual env with Python ${PY_VERSION} =="
if [ ! -d "$VENV_DIR" ]; then
    uv venv --python "$PY_VERSION" "$VENV_DIR"
fi
"$VENV_DIR/bin/python" --version

echo "== 4. Packages (CPU-only torch) =="
VIRTUAL_ENV="$VENV_DIR" uv pip install -r requirements.txt --torch-backend cpu

echo "== 5. Environment report =="
"$VENV_DIR/bin/python" scripts/env_report.py --device auto

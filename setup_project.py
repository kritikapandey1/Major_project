"""
DeepFusionGuard - project folder setup (works on Windows, Mac, Linux).

Run from INSIDE your cloned repo folder (Major_project):
    python setup_project.py
Safe to re-run: it never overwrites files that already exist.
"""
from pathlib import Path

FOLDERS = [
    "data/raw",              # downloaded datasets (never pushed to GitHub)
    "data/processed",        # splits / label files
    "cache",                 # preprocessed crops, audio, transcripts, features
    "src/preprocessing",     # preprocess.py lives here
    "src/encoders",          # CLIP, wav2vec2+AASIST, RoBERTa, Fast-DetectGPT
    "src/fusion",            # consistency module + gated cross-attention
    "src/xai",               # Grad-CAM, SHAP / IG, gate contribution
    "src/utils",             # metrics (Acc, AUC, F1, EER), helpers
    "notebooks",             # Colab / Kaggle experiments
    "configs",               # YAML settings per experiment
    "outputs/checkpoints",
    "outputs/results",
    "outputs/figures",
    "app/backend",           # FastAPI (later phase)
    "app/frontend",          # React (later phase)
    "docs",                  # report, slides, weekly logs
]

REQUIREMENTS = """\
# Install PyTorch FIRST from https://pytorch.org/get-started/locally/
# (pick your CUDA version), then: pip install -r requirements.txt
# torch>=2.2
# torchvision
# torchaudio
transformers>=4.40
peft>=0.10
opencv-python>=4.9
facenet-pytorch>=2.5
librosa>=0.10
scikit-learn>=1.4
numpy>=1.26
pandas>=2.2
tqdm
pyyaml
captum>=0.7
shap>=0.45
grad-cam>=1.5
"""

GITIGNORE = """\
# data and generated files are too big for GitHub
data/raw/
cache/
outputs/checkpoints/
*.npz
*.wav
*.pt
*.pth
*.ckpt

# python
__pycache__/
*.pyc
.venv/
venv/
.ipynb_checkpoints/

# misc
.DS_Store
.env
"""

README = """\
# DeepFusionGuard
An Explainable Multimodal Deep Learning Approach with Cross-Modal Feature Fusion
for Robust Deepfake Detection (Group RG16, Major Project - I).

## Setup
1. Create a virtual environment: `python -m venv .venv` and activate it
2. Install PyTorch from pytorch.org, then `pip install -r requirements.txt`
3. Install FFmpeg (>= 6.0) and check with `ffmpeg -version`

## Structure
See folders under `src/`: preprocessing, encoders, fusion, xai, utils.
"""


def write_if_missing(path: Path, text: str):
    if path.exists():
        print(f"  skip (exists): {path}")
    else:
        path.write_text(text, encoding="utf-8")
        print(f"  created:       {path}")


def main():
    for folder in FOLDERS:
        p = Path(folder)
        p.mkdir(parents=True, exist_ok=True)
        (p / ".gitkeep").touch()          # lets Git track empty folders
    for pkg in ["src", "src/preprocessing", "src/encoders", "src/fusion",
                "src/xai", "src/utils"]:
        (Path(pkg) / "__init__.py").touch()

    write_if_missing(Path("requirements.txt"), REQUIREMENTS)
    write_if_missing(Path(".gitignore"), GITIGNORE)
    write_if_missing(Path("README.md"), README)
    print("\nDone. Next: move preprocess.py into src/preprocessing/")


if __name__ == "__main__":
    main()

"""Print a short environment report for DeepFusionGuard and flag common setup problems.

Usage:
    python scripts/env_report.py [--device {auto,cpu,cuda}]
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import os
import platform
import shutil
import subprocess
import sys
import time

EXPECTED_TORCH = "2.11"
PACKAGES = [
    "torch", "torchvision", "torchaudio", "transformers", "opencv-python-headless",
    "librosa", "soundfile", "faster-whisper", "numpy", "pandas", "scikit-learn",
    "pyyaml", "tqdm", "matplotlib", "psutil", "grad-cam", "captum",
    "fastapi", "uvicorn", "python-multipart", "httpx", "pytest",
]


def tool_version(cmd: str) -> str:
    """Return the first line of `<cmd> -version` / `--version`, or 'NOT FOUND'."""
    if shutil.which(cmd) is None:
        return "NOT FOUND"
    flag = "-version" if cmd == "ffmpeg" else "--version"
    out = subprocess.run([cmd, flag], capture_output=True, text=True).stdout
    return out.splitlines()[0] if out else "unknown"


def os_name() -> str:
    """Return PRETTY_NAME from /etc/os-release, falling back to platform info."""
    try:
        with open("/etc/os-release", encoding="utf-8") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    return line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    return platform.platform()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="auto = cuda if available else cpu")
    args = parser.parse_args()
    problems: list[str] = []

    print("=" * 60)
    print("DeepFusionGuard - environment report")
    print("=" * 60)
    on_colab = "COLAB_RELEASE_TAG" in os.environ or os.path.isdir("/content")
    on_kaggle = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
    where = "Colab" if on_colab else "Kaggle" if on_kaggle else "local"
    print(f"Platform      : {where}")
    print(f"OS            : {os_name()} (kernel {platform.release()})")
    print(f"Python        : {sys.version.split()[0]}  ({sys.executable})")
    if sys.version_info[:2] not in ((3, 12), (3, 13)):
        problems.append(f"Python is {sys.version.split()[0]}; project needs 3.12 (3.13 OK on Colab).")
    print(f"ffmpeg        : {tool_version('ffmpeg')}")
    print(f"git           : {tool_version('git')}")
    if shutil.which("ffmpeg") is None:
        hint = ("winget install Gyan.FFmpeg, then open a new terminal" if sys.platform == "win32"
                else "sudo apt install ffmpeg")
        problems.append(f"ffmpeg not found ({hint}).")

    try:
        import psutil
        vm = psutil.virtual_memory()
        disk = psutil.disk_usage(".")
        print(f"CPU           : {platform.processor() or platform.machine()} | "
              f"{psutil.cpu_count(logical=False)} cores / {psutil.cpu_count()} threads")
        print(f"RAM           : {vm.total / 2**30:.1f} GB total, {vm.available / 2**30:.1f} GB available")
        print(f"Disk (here)   : {disk.free / 2**30:.0f} GB free")
    except ImportError:
        problems.append("psutil missing - install requirements first.")

    print("-" * 60)
    try:
        import torch
        torch.set_num_threads(4)
        cuda_ok = torch.cuda.is_available()
        device = ("cuda" if cuda_ok else "cpu") if args.device == "auto" else args.device
        if device == "cuda" and not cuda_ok:
            problems.append("--device cuda requested but CUDA is not available.")
            device = "cpu"
        print(f"torch         : {torch.__version__}  (CUDA build: {torch.version.cuda})")
        print(f"CUDA available: {cuda_ok}")
        print(f"Device chosen : {device}")
        print(f"Torch threads : {torch.get_num_threads()}")
        if cuda_ok:
            props = torch.cuda.get_device_properties(0)
            print(f"GPU           : {props.name}, {props.total_memory / 2**30:.1f} GB VRAM, "
                  f"compute capability {props.major}.{props.minor}")
        if not torch.__version__.startswith(EXPECTED_TORCH):
            problems.append(f"torch {torch.__version__} found; project pins {EXPECTED_TORCH}.x.")
        try:
            a = torch.randn(512, 512, device=device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                for _ in range(20):
                    a @ a
            if device == "cuda":
                torch.cuda.synchronize()
            print(f"Smoke test    : 20x matmul(512x512) on {device} in {time.perf_counter() - t0:.3f} s")
        except RuntimeError as e:
            problems.append(f"torch smoke test failed on {device}: {e}. If it says 'no kernel image', "
                            "this torch CUDA build does not support this GPU (P100 needs cu126).")
    except ImportError:
        pass  # reported once in the package list below
    except OSError as e:  # Windows: a missing DLL makes `import torch` raise OSError
        problems.append(f"torch is installed but failed to load: {e}. On Windows, install the "
                        "Microsoft Visual C++ Redistributable (winget install Microsoft.VCRedist.2015+.x64).")

    try:
        import cv2
        yunet = hasattr(cv2, "FaceDetectorYN")
        print(f"OpenCV        : {cv2.__version__} (YuNet face detector: {yunet})")
        if not yunet:
            problems.append("cv2.FaceDetectorYN missing - face cropping will not work.")
    except ImportError:
        problems.append("OpenCV (cv2) not importable.")

    print("-" * 60)
    installed = {d.metadata["Name"].lower() for d in md.distributions()}
    for pkg in PACKAGES:
        try:
            print(f"  {pkg:24} {md.version(pkg)}")
        except md.PackageNotFoundError:
            print(f"  {pkg:24} MISSING")
            problems.append(f"{pkg} missing.")
    if "opencv-python" in installed and "opencv-python-headless" in installed:
        problems.append("opencv-python and opencv-python-headless are both installed; "
                        "uninstall opencv-python (they share the cv2 folder).")

    print("=" * 60)
    if problems:
        print("PROBLEMS FOUND:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

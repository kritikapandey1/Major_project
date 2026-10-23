"""Guard the agreed repo layout so files don't silently go missing."""

from __future__ import annotations

import importlib

import pytest

from src.utils.config import REPO_ROOT

EXPECTED_FILES = [
    "configs/default.yaml", "requirements.txt", "README.md", ".gitignore", "docs/SETUP.md",
    "notebooks/colab_setup.ipynb", "scripts/setup_env.sh", "scripts/setup_env.ps1",
    "scripts/setup_env.bat", ".gitattributes", "scripts/env_report.py",
    "scripts/show_config.py", "data/README.md", "AGENTS.md",
    "docs/agent_prompts/README.md", "docs/agent_prompts/00_kickoff_verify.md",
    "docs/agent_prompts/02_preprocessing.md", "docs/agent_prompts/03_features.md",
    "docs/agent_prompts/04_model_modules.md", "docs/agent_prompts/05_metrics_training.md",
    "docs/agent_prompts/06_colab.md",
    "src/preprocess/video_split.py", "src/preprocess/frames.py", "src/preprocess/face_crop.py",
    "src/preprocess/audio.py", "src/preprocess/transcript.py",
    "src/encoders/image_clip.py", "src/encoders/video_temporal.py", "src/encoders/audio_w2v.py",
    "src/encoders/text_encoder.py", "src/encoders/fast_detectgpt.py",
    "src/consistency/sync.py", "src/consistency/emotion.py", "src/consistency/authorship.py",
    "src/fusion/cross_attention.py", "src/fusion/gated_fusion.py", "src/fusion/classifier.py",
    "src/xai/gradcam.py", "src/xai/audio_saliency.py",
    "src/data/datasets.py", "src/data/splits.py",
    "src/train/train_unimodal.py", "src/train/train_fusion.py", "src/train/metrics.py",
    "app/backend/__init__.py",
]
PACKAGES = ["src", "src.utils", "src.preprocess", "src.encoders", "src.consistency",
            "src.fusion", "src.xai", "src.data", "src.train"]


@pytest.mark.parametrize("rel", EXPECTED_FILES)
def test_file_exists(rel):
    assert (REPO_ROOT / rel).is_file(), f"missing {rel}"


@pytest.mark.parametrize("pkg", PACKAGES)
def test_package_imports(pkg):
    importlib.import_module(pkg)


def test_data_folder_is_gitignored():
    lines = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "data/*" in lines and "!data/README.md" in lines


def test_line_ending_rules():
    """bash scripts must stay LF (Linux/Colab); Windows scripts CRLF. Enforced by .gitattributes."""
    rules = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "*.sh     text eol=lf" in rules and "*.ps1    text eol=crlf" in rules
    assert b"\r\n" not in (REPO_ROOT / "scripts" / "setup_env.sh").read_bytes()


def test_agents_md_within_antigravity_limit():
    """Antigravity caps each rules file at 12,000 characters; longer files get cut off."""
    assert len((REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")) < 12_000

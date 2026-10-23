"""Tests for src/utils/config.py (profiles, env substitution, overrides, paths, validation)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.config import (REPO_ROOT, ConfigError, deep_merge, get_path, load_config,
                              resolve_device)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Make tests independent of the machine they run on (laptop vs Colab)."""
    for var in ("DFG_PROFILE", "COLAB_RELEASE_TAG", "DFG_LOCAL_ROOT", "DFG_DRIVE_ROOT"):
        monkeypatch.delenv(var, raising=False)


def test_local_profile_defaults():
    cfg = load_config(profile="local")
    assert cfg["profile"] == "local"
    assert cfg["preprocess"]["num_frames"] == 16
    assert cfg["preprocess"]["sample_rate"] == 16000
    assert cfg["models"]["audio"] == "facebook/wav2vec2-base"
    assert cfg["models"]["dims"]["audio"] == 768


def test_colab_profile_overrides_models_and_paths(monkeypatch):
    monkeypatch.setenv("DFG_DRIVE_ROOT", "/tmp/fake_drive/DFG")
    cfg = load_config(profile="colab")
    assert cfg["models"]["audio"] == "facebook/wav2vec2-xls-r-300m"
    assert cfg["models"]["dims"]["audio"] == 1024          # dims follow the bigger model
    assert cfg["models"]["dims"]["image"] == 512           # untouched keys survive the merge
    # Compare as Path objects: on Windows these print with backslashes.
    assert Path(cfg["paths"]["persist_root"]) == Path("/tmp/fake_drive/DFG")
    assert Path(cfg["paths"]["work_root"]) == Path("/content/dfg_work")   # env default used


def test_auto_profile_detection(monkeypatch):
    assert load_config()["profile"] == "local"
    monkeypatch.setenv("COLAB_RELEASE_TAG", "release-colab_20261020")
    assert load_config()["profile"] == "colab"
    monkeypatch.setenv("DFG_PROFILE", "local")               # explicit env var wins
    assert load_config()["profile"] == "local"


def test_overrides_are_typed():
    cfg = load_config(profile="local", overrides=["train.batch_size=8", "fusion.dropout=0.2",
                                                  "preprocess.transcript.model=tiny"])
    assert cfg["train"]["batch_size"] == 8 and isinstance(cfg["train"]["batch_size"], int)
    assert cfg["fusion"]["dropout"] == pytest.approx(0.2)
    assert cfg["preprocess"]["transcript"]["model"] == "tiny"


@pytest.mark.parametrize("bad", ["train.batchsize=8", "nosuchsection.x=1", "train.batch_size"])
def test_bad_override_gives_clear_error(bad):
    with pytest.raises(ConfigError):
        load_config(profile="local", overrides=[bad])


def test_device_flag_and_validation():
    assert load_config(profile="local", device="cpu")["device"] == "cpu"
    with pytest.raises(ConfigError, match="divisible"):
        load_config(profile="local", overrides=["fusion.num_heads=3"])
    with pytest.raises(ConfigError, match="positive integer"):
        load_config(profile="local", overrides=["preprocess.num_frames=0"])


def test_unknown_profile():
    with pytest.raises(ConfigError, match="unknown profile"):
        load_config(profile="kaggle")


def test_paths_are_absolute_and_named():
    cfg = load_config(profile="local")
    assert get_path(cfg, "work_root") == REPO_ROOT / "data"
    assert get_path(cfg, "features") == REPO_ROOT / "data" / "features"
    assert get_path(cfg, "checkpoints") == REPO_ROOT / "data" / "runs" / "checkpoints"
    assert all(Path(cfg["paths"][k]).is_absolute() for k in ("work_root", "persist_root", "samples"))
    with pytest.raises(ConfigError, match="unknown path name"):
        get_path(cfg, "featurez")


def test_get_path_create(tmp_path):
    cfg = load_config(profile="local", overrides=[f"paths.work_root={tmp_path.as_posix()}"])
    p = get_path(cfg, "processed", create=True)
    assert p.is_dir() and p == tmp_path / "processed"


def test_missing_env_without_default(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    text = (REPO_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    cfg_file.write_text(text.replace("project: DeepFusionGuard", "project: ${env:DFG_NOT_SET}"),
                        encoding="utf-8")
    with pytest.raises(ConfigError, match="DFG_NOT_SET"):
        load_config(cfg_file, profile="local")


def test_deep_merge_does_not_mutate():
    base = {"a": {"b": 1, "c": 2}}
    out = deep_merge(base, {"a": {"b": 9}})
    assert out == {"a": {"b": 9, "c": 2}} and base == {"a": {"b": 1, "c": 2}}


def test_resolve_device():
    torch = pytest.importorskip("torch")
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") == ("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        with pytest.raises(ConfigError, match="no CUDA GPU"):
            resolve_device("cuda")


def test_rooted_paths_kept_on_any_os():
    """A path with a root (Colab-style /content/..., or C:\\...) must not be joined onto the repo."""
    from pathlib import PureWindowsPath
    assert PureWindowsPath("/content/dfg_work").anchor            # rooted, no drive letter
    assert PureWindowsPath("C:/Users/me/data").anchor
    assert not PureWindowsPath("data/runs").anchor                 # relative -> under repo root
    cfg = load_config(profile="local", overrides=["paths.work_root=/content/dfg_work"])
    assert Path(cfg["paths"]["work_root"]) == Path("/content/dfg_work")

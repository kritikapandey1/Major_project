# DeepFusionGuard

**An Explainable Multimodal Deep Learning Approach with Cross-Modal Feature Fusion for Robust Deepfake Detection**
Major Project-I, Group RG16, Jaypee University of Information Technology (JUIT)

One system that detects deepfakes in **images, video, audio and text**, fuses the evidence across modalities, and explains every decision.

## How it works

1. **Preprocessing.** FFmpeg splits video and audio → 16 evenly spaced frames → face and mouth crops (224×224) → 16 kHz mono audio → speech transcript.
2. **Frozen pre-trained encoders.** CLIP ViT-B/16 handles frames, with a small temporal transformer for video. wav2vec 2.0 handles audio. RoBERTa embeds the text, and Fast-DetectGPT gives an AI-text score.
3. **Cross-modal consistency vector** `c = [s_sync, s_emo, s_auth]`:
   - lip–speech sync
   - face–voice emotion mismatch
   - AI-written-script score
4. **Modality-gated cross-attention fusion.** Each available modality gets a gate weight. The weights become the *contribution %* shown to the user, and missing modalities get weight 0.
5. **Explainability.** Grad-CAM on frames, saliency on the log-mel spectrogram, token curvature for text, and gate weights. All are checked with deletion/insertion faithfulness tests.
6. **App.** A FastAPI backend with a simple dashboard.

Metrics: Accuracy, **AUC** (primary), F1, EER. All splits are **identity-disjoint**.

## Status

| Step | Content | Status |
|---|---|---|
| 0 | Environment, pinned requirements, Colab setup notebook | done |
| 1 | Repo structure, config system, README | done |
| 2 | Preprocessing pipeline | next |
| 3 | Feature extraction + caching | |
| 4 | Core model modules (projection, temporal, cross-attention, gating, classifier) | |
| 5 | Metrics + toy training loop | |

## Setup

Full team procedure (laptop, Google Drive, Colab, daily git loop): **[docs/SETUP.md](docs/SETUP.md)**.

Working with an AI coding agent (Antigravity, etc.): `AGENTS.md` holds the project rules, and
**[docs/agent_prompts/](docs/agent_prompts/README.md)** has one prompt per implementation step, with the
results each step must show.

Quick start on a laptop (CPU-only), Ubuntu:
```bash
git clone https://github.com/kritikapandey1/Major_project.git && cd Major_project
bash scripts/setup_env.sh            # Python 3.12 via uv, CPU torch, all pinned packages
.venv/bin/python -m pytest           # run the tests
```
Windows 10/11 (PowerShell):
```powershell
git config --global core.autocrlf false
git clone https://github.com/kritikapandey1/Major_project.git; cd Major_project
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1   # or double-click scripts\setup_env.bat
.venv\Scripts\python -m pytest
```
On Windows, run the other scripts the same way: `.venv\Scripts\python scripts\show_config.py`.
On Colab, open `notebooks/colab_setup.ipynb`, choose a T4 GPU and select *Run all*.

## Where things run

| Machine | Used for |
|---|---|
| Laptop, Ubuntu or Windows (i5-7200U, 8 GB RAM, no usable GPU) | Coding, unit tests, preprocessing a few sample clips, inference demos |
| Colab free (T4) / Kaggle | Dataset preprocessing, feature extraction, training |
| Google Drive | Persistent storage: archives, cached features, checkpoints, logs |

## Configuration

All paths and hyperparameters are in **`configs/default.yaml`**. Nothing is hard-coded.

There are two profiles:
- `local`: small models, laptop paths under `data/`.
- `colab`: the original-plan models, with fast local disk plus Drive.

The profile is picked automatically, or you can force it with `--profile` or the `DFG_PROFILE` environment variable.

Every script accepts the same flags:

| Flag | Meaning |
|---|---|
| `--device {auto,cpu,cuda}` | `auto` = cuda if available, else cpu |
| `--profile {local,colab}` | which profile to use |
| `--set key=value` | override any config value (repeatable), e.g. `--set train.batch_size=8` |
| `--config path.yaml` | use a different config file |
| `--log-level` | DEBUG / INFO / WARNING / ERROR |

**Model choices per profile** (the laptop uses smaller models because of RAM and CPU):

| Component | `local` (laptop) | `colab` (original plan) |
|---|---|---|
| Image/video encoder | CLIP ViT-B/16 | CLIP ViT-B/16 |
| Audio encoder | wav2vec2-base (95M) | wav2vec2 XLS-R 300M |
| Speech-to-text | faster-whisper base (int8) | faster-whisper small (fp16) |
| Fast-DetectGPT model | GPT-2 | GPT-Neo-2.7B |
| Text encoder | RoBERTa-base | RoBERTa-base |
| Face detection | OpenCV YuNet | OpenCV YuNet |

MTCNN (facenet-pytorch) was dropped because it pins `torch<2.3` and `numpy<2`. YuNet ships with OpenCV and gives mouth-corner landmarks.

## Scripts

| Command | What it does | Available from |
|---|---|---|
| `bash scripts/setup_env.sh` | Create `.venv` and install everything (CPU) | Step 0 |
| `scripts\setup_env.ps1` / `setup_env.bat` | Same, for Windows | Step 0 |
| `python scripts/env_report.py [--device auto]` | Environment report: Python, torch, GPU, RAM, packages | Step 0 |
| `python scripts/show_config.py [--profile colab] [--dump]` | Show the resolved config, device and paths | Step 1 |
| `python -m pytest` | Run all unit tests | Step 1 |
| `python scripts/preprocess.py --input data/samples/x.mp4` | Preprocess one video | Step 2 |

## Repository layout

```
configs/default.yaml   all paths, model names and hyperparameters (local + colab profiles)
data/                  not committed (see data/README.md); data/samples/ for test clips
docs/SETUP.md          team setup and daily workflow
src/utils/             config loader, shared CLI flags, seeding, logging
src/preprocess/        video_split, frames, face_crop, audio, transcript
src/encoders/          image_clip, video_temporal, audio_w2v, text_encoder, fast_detectgpt
src/consistency/       sync, emotion (stub), authorship
src/fusion/            cross_attention, gated_fusion, classifier
src/xai/               gradcam (stub), audio_saliency (stub)
src/data/              dataset classes, identity-disjoint splits
src/train/             train_unimodal, train_fusion, metrics
app/backend/           FastAPI app (later)
notebooks/             Colab notebooks
scripts/               command-line entry points
tests/                 pytest unit tests
```

## Datasets

None are downloaded automatically. Folder names are set in `configs/default.yaml` → `datasets`.

| Dataset | Modality | Access |
|---|---|---|
| 140K Real & Fake Faces (Kaggle) | image | open |
| HC3 (Hugging Face) | text | open |
| ASVspoof 2019 / 2021 LA | audio | open |
| In-the-Wild | audio | open |
| FaceForensics++, Celeb-DF v2 | video | request form |
| FakeAVCeleb, LAV-DF | audio + video | request form |
| AV-Deepfake1M++, Deepfake-Eval-2024 | mixed | test only |

## Team conventions

- Commit after each completed step with a message like `Step N: …`, and record `git rev-parse --short HEAD` in the weekly log.
- Library code uses `logging`, type hints and docstrings. Seeds come from the config (`seed: 42`).

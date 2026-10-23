# data/ (not committed)

Everything in this folder except this README is ignored by git.

| Folder | Contents | Created by |
|---|---|---|
| `samples/` | 1-3 tiny test videos for local runs | you (copy files in) |
| `raw/` | downloaded datasets, one sub-folder each (names in `configs/default.yaml` -> `datasets`) | you |
| `processed/<clip_id>/` | frames, face/mouth crops, audio.wav, transcript, metadata.json | `scripts/preprocess.py` (Step 2) |
| `features/<clip_id>/` | cached encoder outputs (`.npy`) | feature extraction (Step 3) |
| `splits/` | identity-disjoint split files | Step 5 |
| `runs/` | local checkpoints and logs (`paths.persist_root`) | training (Step 5) |
| `model_cache/` | small helper models (YuNet face detector) | Step 2 |

On Colab these live under `/content/dfg_work` (fast, temporary) and Google Drive `DeepFusionGuard/` (persistent).

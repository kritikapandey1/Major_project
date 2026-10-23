# Prompt 05 - Step 5: Metrics, identity-disjoint splits, dataset, toy training loop

Follow AGENTS.md. First produce the implementation plan and WAIT for my OK, then implement.
This step must run end-to-end on the laptop CPU in a few minutes, using TOY synthetic features.
Toy results are pipeline checks only: label every printed/plotted toy result
"TOY DATA - not a project result".

## Read first
AGENTS.md, configs/default.yaml (`train`, `splits`), Step 3 feature layout
(data/features/<clip_id>/manifest.json), Step 4 modules (src/fusion/), placeholders in src/train/ and src/data/.

## Implement

### src/train/metrics.py (must import without torch)
`compute_metrics(y_true, y_score, threshold=0.5) -> dict` with keys
`accuracy, auc, f1, eer, eer_threshold, n, n_pos` (y_score = probability of FAKE, label 1 = fake).
- AUC: `sklearn.metrics.roc_auc_score`. EER: from `roc_curve`: `fnr = 1 - tpr`, take the point
  where `|fnr - fpr|` is smallest, EER = mean of fpr and fnr there (also return its threshold).
- Only one class in y_true: auc and eer = NaN with a logged warning, no crash.
- Inputs as lists, numpy or torch tensors; validates equal length and values in {0, 1}.

### src/data/splits.py
`identity_disjoint_split(df, identity_cols: list[str], ratios: dict, seed: int) -> pd.Series` (split
name per row). Groups: a clip with several identities (e.g. face swap source + target) links
them; build connected groups with union-find over all identity columns, then assign whole groups
to train/val/test (greedy by group size, seeded) so the clip counts approximate the ratios.
`check_disjoint(df, identity_cols, split_col) -> None` raises ValueError naming an identity found in two splits.
`save_splits(...)` writes `get_path(cfg, "splits")/<name>.csv`.

### src/data/datasets.py
`CachedFeatureDataset(index_csv, features_root, cfg, split=None)`: one row per clip
(`clip_id, label, identity..., split`). `__getitem__` returns the Step 4 input dict for ONE sample
(no batch dim) + `label` + `clip_id`, loading .npy files from `features/<clip_id>/`.
Missing files -> that modality absent in `present` (zeros in its tensors), never a crash.
Detectgpt NaN -> text authorship treated as missing. Use `torch.from_numpy`, float32.
`collate_fn` stacks samples into the batched dict.

### scripts/make_toy_data.py
Writes `data/toy/features/<clip_id>/*.npy` + `data/toy/index.csv` with the SAME file names and shapes
as Step 3 (face_clip [16,512], mouth_clip, face_mask, audio_aligned [16,768], text_roberta [768],
detectgpt.json). Defaults: 240 clips, 48 identities, 50 % fake, seed from config.
Planted signal: fake clips add a fixed random direction (strength `--signal`, default 0.5) to face
and audio features, and mouth/audio are de-synchronised (shuffled frames) for 50 % of fakes.
Randomly mark 15 % of modalities missing (delete the file). Prints "TOY DATA" in its output.

### src/train/train_unimodal.py (+ thin CLI `scripts/train_unimodal.py`)
- Args: common args + `--modality {video,audio,text}`, `--index data/toy/index.csv`,
  `--features data/toy/features`, `--run-name`, `--resume`.
- Model: projector (+ TemporalTransformer for video) -> masked mean pool -> Classifier (Step 4 modules).
- Loss `BCEWithLogitsLoss`; AdamW (lr, weight_decay from config); `train.epochs`; batch `train.batch_size`.
- Splits: if the index has no split column, create identity-disjoint splits (seeded) and save them.
- Each epoch: train loss, then validation metrics (compute_metrics). Log a console row and append a
  CSV row to `get_path(cfg, "logs")/<run>.csv` with header
  `epoch,train_loss,val_loss,val_acc,val_auc,val_f1,val_eer,lr,seconds`.
- Checkpoint best val AUC to `get_path(cfg, "checkpoints")/<run>/best.pt` and `last.pt` every epoch
  (model, optimizer, epoch, metrics, config). `--resume` continues from last.pt (needed on Colab,
  where sessions die). Early stopping with `train.early_stopping_patience`.
- At the end: load best.pt, evaluate on test, print and save `<run>_test_metrics.json`.
- Respect `--device`; `set_seed`; DataLoader `num_workers` from config (0 is fine on Windows if
  workers cause problems; explain).

## Tests
- metrics: perfect scores -> acc 1, auc 1, eer 0; inverted -> auc 0; 10 000 random scores (seeded)
  -> auc within 0.5 +- 0.03 and eer within 0.5 +- 0.03; a hand-computed small example for EER;
  single class -> NaN without exception.
- splits: no identity in two splits; linked identities (A swapped onto B) end in the same split;
  ratios within +- 10 % of the target on 500 synthetic clips; same seed -> same split.
- dataset: toy clip with deleted audio file -> present = [1, 0, 1]; batch shapes from collate_fn.
- training smoke test: 2 epochs on 40 toy clips runs, writes the CSV (2 rows + header) and best.pt;
  `--resume` continues at epoch 3.

## Demo commands to run and show
```
<venv python> -m pytest -rs
<venv python> scripts/make_toy_data.py
<venv python> scripts/train_unimodal.py --modality video --run-name toy_video --device cpu
<venv python> scripts/train_unimodal.py --modality audio --run-name toy_audio --device cpu
<venv python> scripts/train_unimodal.py --modality text  --run-name toy_text  --device cpu
<venv python> scripts/train_unimodal.py --modality video --run-name toy_video --resume --set train.epochs=5
```
Show: the console epoch table, the CSV contents, the test metrics JSON for each modality, the split
sizes and the identity-overlap check result. Optionally save a loss/AUC plot (matplotlib) to
`data/runs/logs/<run>.png` and open it.

## Acceptance criteria
- [ ] All tests pass.
- [ ] Each toy run finishes on this laptop CPU in < 3 minutes (report real wall time).
- [ ] TOY video and audio runs reach val AUC > 0.85 (the planted signal is learnable); TOY text has
      no planted signal, so its test AUC should be roughly chance (between 0.3 and 0.7 on this small
      set), which shows the loop is not leaking labels. If text scores > 0.8, find the leak before continuing.
- [ ] CSV log + best.pt + last.pt + test metrics JSON exist at the configured paths.
- [ ] Resume continues from the saved epoch (show epoch numbers).
- [ ] Re-running the same command with the same seed gives the same metrics (show both).
- [ ] Every toy result is labelled "TOY DATA - not a project result".

STOP after the RESULTS REPORT. Proposed commit:
`"Step 5: metrics (Acc/AUC/F1/EER), identity-disjoint splits, cached-feature dataset, unimodal training loop"`

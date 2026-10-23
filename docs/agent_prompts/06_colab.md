# Prompt 06 - Step 6: Colab readiness (same code on a T4 GPU)

Follow AGENTS.md. First produce the implementation plan and WAIT for my OK, then implement.
You (the agent) edit files locally; the USER runs the notebook on Colab and pastes the outputs back.
So the deliverable is code + exact instructions, and the RESULTS REPORT has two parts:
local checks (you run them) and Colab checks (filled in from the outputs the user pastes).

## Read first
AGENTS.md, notebooks/colab_setup.ipynb (17 cells: settings, GPU check, Drive mount, clone/pull,
install, env report, show_config, staging helpers stage_in/stage_out, round-trip test, "Next steps"),
docs/SETUP.md (Part C), scripts from Steps 2, 3 and 5.

## Implement

### 1. scripts/make_subset.py - dataset subset index (no downloads)
- Args: common args + `--dataset <key from cfg["datasets"]>`, `--n <clips per class>`, `--out <csv>`.
- Scans `get_path(cfg, "raw") / datasets.<key>.dir` and writes an index CSV
  `clip_id, path, label, identities..., dataset` with a balanced, seeded sample.
- Label + identity rules from each dataset's documented folder layout. Implement these three and
  confirm them against the dataset's own README when the data arrives (say so in the docstring):
  - FaceForensics++: `original_sequences/**/videos/*.mp4` -> real; `manipulated_sequences/<method>/**/videos/*.mp4`
    -> fake; identities from file names (`000.mp4` -> id 000; `000_003.mp4` -> target 000, source 003).
  - Celeb-DF v2: `Celeb-real/`, `YouTube-real/` -> real; `Celeb-synthesis/` -> fake;
    `id0_id16_0000.mp4` -> identities id0 + id16.
  - FakeAVCeleb: top folder `RealVideo-RealAudio` -> real, the other three -> fake; identity = the `idXXXXX` folder.
  - Unknown dataset -> clear error listing supported keys.
- Unit tests build tiny FAKE folder trees in tmp_path (empty files) for each layout and check labels,
  identities, balance and that identity_disjoint_split (Step 5) keeps linked identities together.

### 2. Notebook cells (append after "Next steps"; keep existing cells unchanged)
Each new section: a markdown cell explaining it in 2-3 lines + one code cell. Commands use
`!python scripts/...` from the repo folder, `--device cuda`, profile colab (already set by env var).
- **7. Stage inputs**: `stage_in("samples.tar")` from `archives/raw/` (the user uploads a tar of
  their sample clips to Drive once; print exact instructions if it is missing).
- **8. Preprocess**: `scripts/preprocess.py --input <LOCAL_ROOT>/samples --device cuda`; print the summary.
- **9. Extract features**: `scripts/extract_features.py --all --device cuda`; print the encoder table.
  Show `torch.cuda.max_memory_allocated()` per encoder in the table when on cuda (add this to the
  Step 3 script if missing, guarded by `device == "cuda"`).
- **10. Save to Drive**: `stage_out` processed and features folders as `.tar` into
  `archives/processed/` and `features/` on Drive; print sizes.
- **11. Toy training on GPU**: `make_toy_data.py` then `train_unimodal.py --modality video --device cuda
  --set train.batch_size=32`; logs/checkpoints go to Drive (persist_root); then the same command with
  `--resume` to prove resuming works after a disconnect.
- **12. CPU vs GPU timing**: read the timings from manifests/metadata and print a small table
  (encoder | laptop s/clip (user types the laptop numbers from Step 3) | Colab s/clip).
- Optional **13. Dataset subset** (only runs if the dataset folder exists on local disk after
  `stage_in`): `make_subset.py --dataset ffpp --n 25`, then preprocess + extract on that index.

### 3. scripts/clear_notebook_outputs.py
Pure-Python (json only) tool that strips outputs and execution counts from .ipynb files, so
notebooks are committed clean. Run it on notebooks/ before every commit; add a test.

### 4. docs/SETUP.md
Add Part C.7 "Running the pipeline on Colab" (how to make `samples.tar`, upload it to
`DeepFusionGuard/archives/raw/`, run cells 7-12, where results land on Drive).

## Local checks (you run these)
```
<venv python> -m pytest -rs
<venv python> scripts/make_subset.py --help
<venv python> scripts/clear_notebook_outputs.py notebooks/colab_setup.ipynb
<venv python> -c "import json; nb = json.load(open('notebooks/colab_setup.ipynb', encoding='utf-8')); print(len(nb['cells']), 'cells, valid JSON')"
```

## Colab checks (the user runs; you list exactly what to paste back)
Runtime T4 -> Run all. Ask the user to paste the outputs of cells 5 (env report), 8, 9, 10, 11, 12.

## Acceptance criteria
Local:
- [ ] All tests pass, including make_subset tests on fake folder trees.
- [ ] Notebook valid JSON/nbformat, outputs cleared, existing cells unchanged (show a cell list).
Colab (from the user's pasted outputs):
- [ ] env report: GPU Tesla T4, torch 2.11.0 CUDA build, All checks passed.
- [ ] preprocess + extract ran with `--device cuda`; feature shapes identical to the laptop run.
- [ ] Features and logs present on Drive after the session (paths shown).
- [ ] Toy GPU training finished and `--resume` continued from the saved epoch.
- [ ] CPU vs GPU timing table filled with real numbers.

STOP after the RESULTS REPORT (local part), give the user the Colab instructions, and wait for the
pasted outputs before completing the Colab part. Proposed commit:
`"Step 6: Colab readiness (pipeline cells, subset index builder, clean-notebook tool)"`

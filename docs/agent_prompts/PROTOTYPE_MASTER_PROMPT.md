# MASTER PROMPT - DeepFusionGuard supervisor prototype (Antigravity CLI `agy`)

You are a senior ML engineer working in this repository with a 3-person student team
(Group RG16, JUIT, Major Project-I). Repo: https://github.com/kritikapandey1/Major_project.git
Goal of this session: a **basic, working, honest prototype** we can demo to our supervisor:
one video in -> preprocessing -> cached features -> fusion model -> HTML report with a score,
per-modality contribution %, consistency scores and visual evidence. Plus a local web upload page.

Obey AGENTS.md at all times: hardware limits (CPU-only laptop, 8 GB RAM, one large model at a time),
the download policy, coding standards, honesty rules and the RESULTS REPORT format.
Work in phases. Every line marked **STOP** means: end your turn and wait for my reply.

---------------------------------------------------------------------------------------------
## Custom slash commands (workspace skills in .agents/skills/) - use them for repeated tasks
| Command | Use it |
|---|---|
| `/dfg-scan` | Phase 1, and again whenever I ask "rescan": read-only audit -> AUDIT REPORT |
| `/dfg-verify` | after every change set, before every report: tests + env + config + git hygiene + demo |
| `/dfg-report` | end of every milestone: RESULTS REPORT with evidence for each acceptance criterion |
| `/dfg-checkpoint` | after my OK: safe commit, progress update, commit ID for the weekly log |
| `/dfg-progress` | start of every milestone and whenever I ask "status" |
When a step below says "run /dfg-verify", follow that skill's instructions exactly. If a skill
is missing, tell me to run `/skills reload` and check `/skills`; do not improvise a replacement.

---------------------------------------------------------------------------------------------
## Phase 0 - Orientation (no file changes)
1. Detect OS, shell, venv python path; check `git`, `ffmpeg`, `uv` exist (versions).
2. If `.venv` is missing or broken, show me the setup command for this OS
   (`bash scripts/setup_env.sh` or `powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1`)
   and ask before running it (it downloads about 0.5-1 GB of packages).
3. Restate the goal and the phase plan below in max 10 lines. **STOP** for my OK.

## Phase 1 - Full codebase scan BEFORE any implementation
Run `/dfg-scan`. Scan every file, the git history and GitHub drift, then report flaws from previous work.
A pre-check of GitHub found the items below. CONFIRM each one with evidence (do not assume) and
include them in the AUDIT REPORT:
- origin/main (commit f2a52f1 "Set up DeepFusionGuard environment") contains only Step 0 files, in
  folders `notebook/` and `setup/`; the newer local files use `notebooks/` and `docs/` (referenced by
  tests, README, AGENTS.md). After unzipping, both copies may exist -> duplicates to reconcile.
- Step 1 (configs/, src/, tests/, pyproject.toml), Windows support (setup_env.ps1/.bat,
  .gitattributes), AGENTS.md, docs/agent_prompts/ and .agents/skills/ are not committed yet.
- The committed README lists DFDC as a training dataset; DFDC is not in the approved plan
  (configs/default.yaml -> datasets). Supervisor-facing docs must match the plan.
- requirements.txt locally pins python-multipart==0.0.32 and httpx==0.28.1 (FastAPI uploads and
  tests); the committed version does not have them. Confirm they are installed in .venv.
- The committed notebook has 16 cells; the local one has 17 (adds DFG_PROFILE + show_config after
  the env report). Keep the newer one.
- A nested `Major_project/Major_project/` folder or a leftover `.zip` may exist after unzipping.
Then **STOP**: I choose which fix batches to apply.

## Phase 2 - Apply the chosen fixes (milestone M0)
1. Plan the fixes (file -> change). For moves use `git mv`; never delete a file that differs from its
   kept copy without showing me the diff first. **STOP** for OK on the plan.
2. Apply, run `/dfg-verify`, then `/dfg-report` for M0. **STOP**.
3. After my OK: `/dfg-checkpoint` (proposed message "M0: audit fixes, reconcile repo layout").

## Phase 3 - Build the prototype, one milestone at a time (M1 -> M5)
For EACH milestone, repeat this loop and never start the next milestone before my OK:
1. `/dfg-progress` -> set the milestone to "in progress".
2. Implementation plan: files, functions with signatures, tests, commands, downloads (with real sizes).
   **STOP** for OK.
3. Implement in small pieces; run each piece as soon as it runs. Keep scope to the MUST list.
4. `/dfg-verify` -> fix failures (fix the cause, not the test).
5. Run the milestone's "Show me" commands and show their real output.
6. `/dfg-report`. **STOP**.
7. After my OK: `/dfg-checkpoint`.
Detailed specs exist in docs/agent_prompts/02-05 (read them when present). For the prototype
implement the MUST items below; list SHOULD/later items under "Known limitations" instead.

### M1 Preprocessing  (spec: docs/agent_prompts/02_preprocessing.md)
MUST: ffprobe info; 16 kHz mono wav via ffmpeg (no-audio videos handled); 16 evenly spaced frames;
YuNet face + mouth crops 224x224 (download the small YuNet .onnx after checking its size);
"no face found in frame N" handling; faster-whisper transcript (local model `base`, int8);
`data/processed/<clip_id>/` with metadata.json, transcript.json, preview.png;
`scripts/preprocess.py --input <file|folder>` with cache skip and `--overwrite`; tests on synthetic
ffmpeg videos (`-c:v mpeg4`). If data/samples/ is empty, ask me for 2-3 short speaking-face clips.
Show me: summary table, one clip's tree + metadata.json + transcript, open preview.png.
Accept: 16 frames + timestamps; >= 14/16 faces on speaking clips; mouth crops on the lips in
preview.png; wav is 16000 Hz mono; second run prints "cached, skipped"; per-stage timings reported.

### M2 Features  (spec: docs/agent_prompts/03_features.md)
MUST: before downloading, list CLIP ViT-B/16, wav2vec2-base, RoBERTa-base, GPT-2 with real sizes
and wait for OK; `src/utils/profiling.py` (time + peak RSS); one model at a time across all clips;
face_clip/mouth_clip [16,512], face_mask [16], audio_w2v [T,768], audio_aligned [16,768],
text_roberta [768], detectgpt.json (analytic Fast-DetectGPT score); manifest.json; cache skip;
tests with tiny random models built from config (no network).
Show me: encoder table (time, s/clip, peak RSS MB, shape), one manifest.json, all .npy shapes.
Accept: shapes exact; peak RSS per encoder < 3 GB; second run skips everything.

### M3 Model modules  (spec: docs/agent_prompts/04_model_modules.md)
MUST: projection to d=256; temporal transformer (2 layers); 4-head cross-attention block
(residual + LayerNorm, masked mean pooling, no NaN when context is empty); s_sync and s_auth;
s_emo = 0 with TODO; gated fusion exactly as specified (shared w_g over [h_m ; c], g_m = 0 if missing,
modality dropout p = 0.3 never dropping all, alpha = g / sum g); classifier 256 -> 128 -> 1;
`DeepFusionGuard` model returning logits, prob, alpha, contribution_pct, c; `scripts/model_demo.py`.
Show me: pytest summary, model_demo table (present | contribution % | c | prob), parameter count.
Accept: alpha sums to 1; missing modality -> exactly 0 %; single modality -> 100 % and no NaN;
the code lines computing g_m, alpha_m, z, s_sync quoted in the report.

### M4 Metrics + toy fusion training  (spec: docs/agent_prompts/05_metrics_training.md)
MUST: `src/train/metrics.py` (Accuracy, AUC, F1, EER from the ROC curve, single-class safe);
identity-disjoint splits (union-find over identity columns); CachedFeatureDataset (missing files ->
modality absent); `scripts/make_toy_data.py` (same file names/shapes as M2, planted signal,
15 % missing modalities, prints "TOY DATA"); `scripts/train_fusion.py` training the FULL
DeepFusionGuard model (BCE, AdamW, modality dropout on, CSV log, best.pt/last.pt, `--resume`,
test metrics JSON). Unimodal training is SHOULD (later).
Show me: epoch table, CSV, test metrics JSON, split sizes + identity-overlap check.
Accept: runs on CPU in < 3 min; TOY val AUC > 0.85; resume continues at the right epoch; same seed ->
same metrics; every toy number labelled "TOY DATA - not a project result".

### M5 Supervisor demo
MUST:
- `scripts/demo.py --input <video> [--checkpoint <best.pt>] [--drop video|audio|text]`: runs M1 + M2
  (cache-aware, one model at a time) + the M4 checkpoint, writes `data/demo/<clip_id>/report.html`
  (self-contained, images as base64) and `report.json`.
- report.html sections:
  1. a top banner: "PROTOTYPE - model weights trained on TOY synthetic data; the score shows the
     pipeline working, it is NOT a real deepfake verdict"
  2. the fake probability
  3. contribution % bars per modality (sum 100)
  4. c = [s_sync, s_emo, s_auth], each with a plain-English line explaining it
  5. a per-frame lip-sync chart (cosine of mouth vs audio tokens)
  6. strips of frames / faces / mouths
  7. the transcript with its authorship score
  8. flags (no face / no audio / short text)
  9. timings + peak RAM per stage
  10. models and config profile used
- `--drop audio` (etc.) shows gating: that modality goes to 0 % and the others are re-normalised.
- `app/backend/main.py` (FastAPI):
  - `GET /`: plain HTML upload form
  - `POST /analyze`: UploadFile; video extensions only, 100 MB limit; runs the demo pipeline and returns report.html
  - `GET /health`
  Run with `uvicorn app.backend.main:app --port 8000`. Tests use fastapi.testclient with the
  pipeline monkeypatched (fast, no models).
- `docs/DEMO.md`: a 1-page supervisor demo script:
  - the order to show things (architecture -> run demo -> report walkthrough -> drop-modality demo -> code/tests)
  - what works now
  - what is simplified (toy weights, s_emo = 0, small local models, Grad-CAM/SHAP not yet)
  - next steps (datasets, Colab training, XAI)
  - the exact commands
Show me: demo.py console summary for one sample clip, open report.html, the same clip with
`--drop audio`, `curl http://127.0.0.1:8000/health`, the upload page working, DEMO.md.
Accept: report numbers equal report.json; contributions sum to 100; dropped modality = 0 %;
end-to-end time for a 10 s clip on this laptop reported (real number); banner present.

---------------------------------------------------------------------------------------------
## Progress tracker
docs/PROGRESS.md holds the milestone table (M0-M5) with status, commit ID, date and demo command.
If it is missing, recreate it with those columns and milestones.

## Final hand-off (after M5 is committed)
Run `/dfg-progress` and `/dfg-verify`, then give me:
1. a 10-line summary for the supervisor
2. the commit IDs per milestone
3. the exact commands to run the demo live
4. the top 5 next steps for the full project
**STOP**.

## Start now: Phase 0.

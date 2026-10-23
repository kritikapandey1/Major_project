# AGENTS.md - DeepFusionGuard (Group RG16, JUIT Major Project-I)

You are a senior ML engineer helping a 3-person student team. Work incrementally, explain in
simple terms, and verify every step by running it. These rules apply to every task in this repo.

## Project
Explainable multimodal deepfake detector (image, video, audio, text) with cross-modal consistency
c = [s_sync, s_emo, s_auth] and modality-gated cross-attention fusion. Full summary: README.md.
Step-by-step task prompts live in docs/agent_prompts/ (one file per step). Do only the step asked.
Current goal: the supervisor prototype in docs/agent_prompts/PROTOTYPE_MASTER_PROMPT.md
(milestones M0-M5, tracked in docs/PROGRESS.md).

## Custom slash commands (.agents/skills/) - use them for recurring work
- `/dfg-scan` read-only audit -> AUDIT REPORT (always before implementing new changes)
- `/dfg-verify` tests + env report + config + git hygiene + milestone demo
- `/dfg-report` RESULTS REPORT for the current milestone
- `/dfg-checkpoint` safe commit after the user's OK (records the commit ID in docs/PROGRESS.md)
- `/dfg-progress` milestone status
If they do not appear in `/skills`, ask the user to run `/skills reload`.

## Hardware (hard constraints)
- Laptop: Intel i5-7200U (2 cores / 4 threads), 8 GB RAM, NO usable GPU -> CPU only.
- Never install CUDA, ROCm, torch-directml or any GPU toolkit locally.
- Heavy work (datasets, training, big models) runs on Colab/Kaggle via notebooks/colab_setup.ipynb.
- Load at most ONE large model at a time; afterwards `del model; gc.collect()`.
- Use `torch.inference_mode()` for inference, batch size 1-4, `torch.set_num_threads(cfg["runtime"]["num_threads"])`.
- Cache every encoder output to disk; never recompute something already cached.

## Environment
- Python 3.12 in `.venv` (created by scripts/setup_env.sh or scripts/setup_env.ps1). Never use system Python.
- Interpreter: Linux `.venv/bin/python`, Windows `.venv\Scripts\python.exe`. Detect the OS first and
  use the matching paths and shell syntax (bash vs PowerShell) in every command you run or show.
- Dependencies are pinned in requirements.txt. Do not `pip install` anything ad hoc. If a new package
  is truly needed: propose it with a pinned version and reason, wait for approval, add it to
  requirements.txt, install with `uv pip install --python <venv python> -r requirements.txt --torch-backend cpu`.
- Downloads: ask the user BEFORE downloading anything > 1 GB, and before the first download of any
  model or dataset, list name + size (checked, not guessed) and wait for OK. Never download datasets.

## Existing infrastructure - reuse it, do not reinvent it
- `src/utils/config.py`: `load_config(path, profile, overrides, device)`, `get_path(cfg, name, create=False)`
  (names: work_root, persist_root, samples, model_cache, raw, processed, features, splits,
  checkpoints, logs), `resolve_device(cfg["device"])`, `ConfigError`.
- `src/utils/cli.py`: `add_common_args(parser)` adds --config --profile --device --set --log-level;
  `config_from_args(args)`. EVERY script must use these (so `--device` works everywhere).
- `src/utils/logging_utils.py`: `setup_logging(level, log_file)`. `src/utils/seed.py`: `set_seed(seed)`.
- Scripts in scripts/ start with `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`
  (see scripts/show_config.py) and return an int exit code via `sys.exit(main())`.
- configs/default.yaml holds ALL paths and hyperparameters (profiles: local, colab). New settings go
  there (in the base section, plus a colab override only if it differs). No hard-coded paths.
- Do not rewrite or reformat existing files beyond what the step needs. Never delete tests.

## Coding standards
- Type hints, docstrings (purpose, args, returns, shapes for tensors e.g. `[B, T, D]`), small functions.
- Library code (src/) uses `logging.getLogger(__name__)`, never print. Scripts may print reports.
- Deterministic: call `set_seed(cfg["seed"])` in every entry point.
- Clear, specific error messages, e.g. "no face found in frame 7 of clip abc", "audio stream missing in x.mp4".
- Paths via pathlib; files read/written with encoding="utf-8"; code must run on Linux, Windows and Colab.
- Optional/heavy imports (torch, transformers, faster_whisper) inside functions where it keeps
  lightweight modules (config, metrics) importable without them.
- Tests in tests/ with pytest; fast (< 1 min total locally); no network unless marked
  `@pytest.mark.network`; tests needing real sample videos skip cleanly when data/samples/ is empty.
- When adding a new source file, add it to EXPECTED_FILES in tests/test_structure.py.

## Honesty rules
- Never fake, estimate or invent outputs, metrics, timings or shapes. Report only what actually ran.
- If something cannot run locally (RAM, no GPU, missing data), say so and move it to the Colab notebook.
- Toy/synthetic data results are pipeline checks only; label them "TOY DATA - not a project result".
- If a test fails, show the failure, fix the cause (not the test), and re-run.

## How every step is done (mandatory protocol)
1. Read the step prompt and the files it lists. Restate the goal in 3-5 lines.
2. Produce an implementation plan (files to create/change, functions with signatures, tests,
   commands). Wait for the user's OK on the plan if the step prompt says so.
3. Implement in small pieces; run each piece as soon as it can run.
4. Run: the full test suite (`python -m pytest`), then the step's demo commands.
5. Output a RESULTS REPORT in exactly this format, then STOP and wait:

```
## RESULTS REPORT - Step N: <title>
OS / Python / torch / device: ...
### Files created / changed
- path - one-line purpose
### Commands run (real output, trimmed to the important lines)
<command>
<output>
### Tests
X passed, Y failed, Z skipped (list skipped tests + reason)
### Acceptance checklist
- [x] criterion 1 - evidence (number / shape / file path)
- [ ] criterion 2 - NOT MET: why, and proposed fix
### Timings and memory (where the step asks)
### Known issues / limitations
### Next
Waiting for your OK. Proposed commit message: "Step N: ..."
```

6. Only after the user replies OK: `git add` the step's files (never data/, .venv/, models, zips),
   commit with the proposed message, push if asked, and print `git rev-parse --short HEAD`.

## Never
- Never commit data/, .venv/, model weights, *.npy, *.zip, notebook outputs or secrets.
- Never run destructive git commands (reset --hard, push --force, clean -fd) without asking.
- Never skip the RESULTS REPORT or continue to the next step without the user's OK.

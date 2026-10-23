# Prompt 00 - Kickoff: check the workspace, set up the environment, verify Steps 0-1

Follow AGENTS.md (project rules). This step builds nothing new; it proves the existing
Step 0 + Step 1 files work on THIS machine. Do not change any source file unless a check fails,
and if one fails, explain the cause before fixing it.

## Context
The user unzipped `Major_project_windows.zip` (or `Major_project_step1.zip`) into this workspace.
The zip contains a top-level folder `Major_project/`, so files may now be nested one level too deep.
The git remote is https://github.com/kritikapandey1/Major_project.git.

## Tasks
1. **Detect the platform.** Report OS (Linux/Windows + version), shell, CPU, RAM, free disk,
   and whether `git`, `ffmpeg`, `uv` are on PATH (with versions).

2. **Check the workspace layout.** The repo root must directly contain: `AGENTS.md`, `README.md`,
   `requirements.txt`, `pyproject.toml`, `.gitignore`, `.gitattributes`, `configs/default.yaml`,
   `scripts/`, `src/`, `tests/`, `docs/`, `notebooks/`.
   - If they are inside a nested `Major_project/Major_project/` (or similar), show the tree
     (2 levels), then propose the exact commands to move the contents up one level, and WAIT for OK
     before moving anything. Do not overwrite a file that differs without showing the diff.
   - If the `.zip` file itself is inside the repo, propose deleting it (it must never be committed).
   - Check `git status` and `git remote -v`. If this folder is not a git repo, say so and propose
     `git init` + `git remote add origin ...` (wait for OK), or cloning fresh and copying files in.

3. **Line endings.** Verify `scripts/setup_env.sh` has LF line endings and `scripts/setup_env.ps1`
   has CRLF. On Windows also run `git config core.autocrlf` and recommend `false` if it is `true`.

4. **Create the environment** (skip if `.venv` already passes step 5):
   - Linux: `bash scripts/setup_env.sh`
   - Windows: `powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1`
   The install downloads roughly 0.5-1 GB of packages (CPU torch is the biggest). That is allowed.
   If any command fails, show the full error and stop.

5. **Run the checks and show their real output:**
   ```
   <venv python> scripts/env_report.py --device auto
   <venv python> -m pytest -rs
   <venv python> scripts/show_config.py
   <venv python> scripts/show_config.py --profile colab --set train.batch_size=16
   <venv python> scripts/show_config.py --set train.batchsize=8      (must fail cleanly, exit code 2)
   ```

6. **Explain the repo in plain words** (max 15 lines): what configs/default.yaml controls, how the
   local vs colab profile is chosen, what `get_path` returns, and what each top-level folder is for.

## Acceptance criteria (all must be shown with evidence in the RESULTS REPORT)
- [ ] Repo root layout correct (show `git status --short` and a 1-level tree).
- [ ] `env_report.py` ends with `All checks passed.` and shows `torch 2.11.0+cpu`,
      `CUDA available: False`, `Device chosen: cpu`, `YuNet face detector: True`.
- [ ] pytest: **72 passed, 0 failed, 0 skipped** (the torch device test now runs because torch is installed).
      If the count differs, list which tests and why.
- [ ] `show_config.py` local: profile `local`, device `cpu`, all paths under `<repo>/data`.
- [ ] colab profile: audio model `facebook/wav2vec2-xls-r-300m`, dims.audio `1024`, batch_size `16`.
- [ ] Bad override exits with code 2 and an error listing the valid `train` keys.
- [ ] No source files were modified (or each modification is listed with the reason).

## Deliverable
RESULTS REPORT (format in AGENTS.md). Proposed commit message:
`"Step 0-1 verified on <OS>: environment + tests (72 passed)"` (only if something was changed;
otherwise say "nothing to commit"). Then STOP and wait for my OK.

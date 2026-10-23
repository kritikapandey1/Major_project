---
name: dfg-scan
description: Read-only audit of the whole DeepFusionGuard codebase. Finds flaws, inconsistencies and leftovers from previous work and outputs an AUDIT REPORT. Changes nothing.
---

# /dfg-scan - full read-only codebase audit

Follow AGENTS.md. This command is READ-ONLY: do not create, edit, move or delete any file.
Allowed commands: listing, reading, git status/log/diff/fetch, running tests and the check scripts.

## 1. Inventory
- Detect the OS and the venv python (`.venv/bin/python` or `.venv\Scripts\python.exe`).
- List every file (excluding .venv, data, __pycache__, .git) with its size. Flag:
  - a nested `Major_project/Major_project/` folder
  - `.zip` files
  - duplicate folders (e.g. `notebook/` vs `notebooks/`, `setup/` vs `docs/`)
  - files larger than 5 MB
  - notebooks with saved outputs
- Run `git status --short`, `git log --oneline -10`, `git remote -v`, then `git fetch`, then
  `git log --oneline HEAD..origin/main` and `git log --oneline origin/main..HEAD`
  (shows drift between the local copy and GitHub).
- For each duplicate file: say which copy is newer and give a short summary of the diff.

## 2. Read the code
Read every file in src/, scripts/, tests/, configs/, app/ and docs/, plus the cell list of each
notebook, AGENTS.md and README.md. Record each module's status: implemented / partial / placeholder.

## 3. Checks (record evidence for each)
- **Tests:** `<venv python> -m pytest -rs` (counts, failures, skips with reasons).
- **Scripts:** `<venv python> scripts/env_report.py --device auto` and `<venv python> scripts/show_config.py`.
- **Dependencies:**
  - every import in src/ and scripts/ is pinned in requirements.txt
  - `uv pip check` (or `<venv python> -m pip check`) reports no conflicts
- **Config vs code:**
  - every `cfg[...]` key used in code exists in configs/default.yaml
  - model names and `models.dims` agree
  - no hard-coded paths in src/ (search for "/content", "C:\\", "/home/" and "data/" string literals)
- **Coding standards:**
  - `print(` in src/ (library code should use logging)
  - public functions missing type hints or docstrings
  - bare `except:`
  - scripts missing `--device` (`add_common_args`) or `set_seed`
- **Docs vs reality:** claims in README.md, AGENTS.md and docs/ (folder names, test counts,
  datasets, commands) match the code. The approved dataset list is configs/default.yaml -> datasets.
- **Notebook:** cells in order (packages installed before anything imports project code), no saved outputs.
- **Safety:**
  - no secrets or tokens
  - no data, weights, .npy or .zip files tracked by git
  - line endings follow .gitattributes (`setup_env.sh` LF, `.ps1`/`.bat` CRLF)
- **Bugs from previous work:** logic errors, wrong tensor shapes, dead code, TODOs, functions whose
  behaviour does not match their docstring.

## 4. Output: AUDIT REPORT (exact format), then STOP
```
## AUDIT REPORT - <date> - <OS>
Summary: <n> files | tests <p> passed / <f> failed / <s> skipped | git: <ahead/behind/untracked>
Module status: <module>: implemented | partial | placeholder, ...
| ID | Severity | Where (file:line) | Finding | Evidence | Proposed fix | Effort |
|----|----------|-------------------|---------|----------|--------------|--------|
Severity: BLOCKER (stops the prototype) | MAJOR (wrong or inconsistent) | MINOR (style/docs)
Fix batches: A = blockers, B = majors, C = minors (list the IDs in each)
Question: Which batches should I apply? (A / A+B / A+B+C / pick IDs)
```
Never fix anything during /dfg-scan. Wait for the user's choice.

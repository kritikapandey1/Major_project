---
name: dfg-verify
description: Run the DeepFusionGuard verification suite (pytest, env report, config check, git hygiene, plus the current milestone's demo command) and summarise the real results.
---

# /dfg-verify - verification suite

Follow AGENTS.md. Do not edit files. Run every command and show its real output (trimmed). Never guess.

1. Detect the OS and the venv python (Linux `.venv/bin/python`, Windows `.venv\Scripts\python.exe`).
2. `<py> -m pytest -rs` -> counts of passed / failed / skipped, plus skip reasons.
   On a failure, show the first failing traceback.
3. `<py> scripts/env_report.py --device auto` -> the last line must be `All checks passed.`
4. `<py> scripts/show_config.py` -> profile, device, paths.
5. Git hygiene: run `git status --short` and confirm that nothing under data/ or .venv/, and no
   *.npy, *.pt, *.zip or model weights, is staged or tracked.
6. If docs/PROGRESS.md lists a "Demo command" for the current milestone, run it and show the output.
7. Print this table, then stop:
```
| Check          | Result         | Evidence                       |
|----------------|----------------|--------------------------------|
| pytest         | PASS/FAIL      | <p> passed, <f> failed, <s> skipped |
| env report     | PASS/FAIL      | All checks passed              |
| config         | PASS/FAIL      | profile local, device cpu      |
| git hygiene    | PASS/FAIL      | ...                            |
| milestone demo | PASS/FAIL/N.A. | ...                            |
Overall: PASS / FAIL (list what failed)
```

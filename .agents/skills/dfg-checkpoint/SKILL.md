---
name: dfg-checkpoint
description: Safely commit the current DeepFusionGuard milestone. Runs pre-commit checks, shows what will be committed, waits for confirmation, commits, records the commit ID in docs/PROGRESS.md and prints it.
---

# /dfg-checkpoint - safe commit

Follow AGENTS.md. Never commit without an explicit "yes" from the user in this turn.

1. Pre-commit checks:
   - Run `<py> -m pytest -q`. Refuse to continue if any test fails.
   - If scripts/clear_notebook_outputs.py exists, run it on notebooks/.
2. `git add` only the milestone's files. Never add data/, .venv/, *.npy, *.pt, *.pth, *.onnx, *.zip or secrets.
3. Show `git status --short`, `git diff --cached --stat` and the proposed commit message.
4. Ask: "Commit these files with this message? (yes / no / edit message)". STOP until the user answers.
5. On yes:
   1. Commit and record `ID = git rev-parse --short HEAD`.
   2. Update docs/PROGRESS.md: set the milestone to `done` and fill in the commit ID and today's date.
   3. Commit that change separately as `docs: progress <milestone> done (<ID>)`.
      Do NOT amend: amending changes the hash and the recorded ID would be wrong.
   4. Print the milestone commit ID (this is the one for the weekly log).
6. Ask whether to `git push`; never force-push. Report the push result.

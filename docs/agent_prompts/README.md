# Agent prompts - how to run DeepFusionGuard step by step in Antigravity

## Fastest path: the supervisor prototype in the agy terminal (one prompt)
`PROTOTYPE_MASTER_PROMPT.md` is a single prompt that runs everything:
audit first (`/dfg-scan`), you choose the fixes, then milestones M0-M5 with a STOP after each.
From the repo root:
```
agy --mode plan
```
In the agy session: type `/skills` and check that dfg-scan, dfg-verify, dfg-report, dfg-checkpoint
and dfg-progress are listed (if not: `/skills reload`). Then paste:
```
Read @AGENTS.md and @docs/agent_prompts/PROTOTYPE_MASTER_PROMPT.md and execute the master prompt, starting at Phase 0. Stop at every STOP.
```
Keep tool permissions on request-review (`/permissions`). The step prompts below are the detailed
specs the master prompt uses; you can also run them one by one instead.


`AGENTS.md` (repo root) holds the permanent rules: hardware limits, download policy, coding standards,
honesty rules and the RESULTS REPORT format. Antigravity loads it automatically for every task.
The files in this folder are the **step prompts**: one per step, run in order, one at a time.

## One-time setup in Antigravity
1. Open the **repo root folder** (the one containing `AGENTS.md`) as the workspace, not a parent folder.
2. Use **Planning** mode for every step (the agent shows a plan before changing code).
3. Set terminal commands to **ask for your approval** rather than auto-run (agent settings; the exact
   label depends on your Antigravity version). Approve normal commands (pytest, scripts); read
   carefully anything that installs, downloads, deletes or runs git.
4. Start a **new conversation for each step**, so old context doesn't confuse the agent.

## How to run a step
Paste this into the agent panel (change the file name per step):

```
Read @AGENTS.md and @docs/agent_prompts/00_kickoff_verify.md.
Execute that prompt exactly, following the protocol in AGENTS.md.
Start by restating the goal and showing your plan. Do not move on to another step.
```

Then:
1. Read the plan. Reply `OK` or ask for changes.
2. Approve commands as the agent runs them.
3. Check the **RESULTS REPORT** against the step's acceptance checklist (also summarised below).
   Every box must be ticked with real evidence (numbers, shapes, file paths, test counts).
4. Reply `OK, commit` -> the agent commits and prints the commit ID for your weekly log.
5. If something is wrong, say what, e.g. "mouth crops are off-centre in preview.png; fix before commit".

If the agent skips the report, invents numbers, or jumps ahead, reply:
`Stop. Follow AGENTS.md: show the RESULTS REPORT with real command output for this step only.`

## Order and what you should see

| # | Prompt | You should see at the end |
|---|---|---|
| 00 | `00_kickoff_verify.md` | `All checks passed`; torch 2.11.0+cpu; **72 passed** in pytest; colab profile shows XLS-R + dims 1024 |
| 02 | `02_preprocessing.md` | `data/processed/<clip>/` with 16 frames, face + mouth crops, 16 kHz `audio.wav`, transcript, `metadata.json`; **`preview.png`** with mouth crops centred on the lips |
| 03 | `03_features.md` | Table per encoder (time, peak RAM); shapes face/mouth (16, 512), audio (~50/s, 768), aligned (16, 768), text (768,); second run "cached, skipped" |
| 04 | `04_model_modules.md` | Demo table: missing modality -> **0.0 %**, each row sums to **100 %**, no NaN; parameter count |
| 05 | `05_metrics_training.md` | Toy video/audio val AUC > 0.85, toy text ~ chance; CSV log, best.pt, resume works; all labelled TOY DATA |
| 06 | `06_colab.md` | Local: tests pass. Colab (you run it): Tesla T4, same feature shapes with `--device cuda`, results on Drive, CPU vs GPU timing table |

There is no prompt `01`: Step 1 (repo structure and config) is already done and is verified by prompt 00.

## What you need to provide
- Before 02: 2-3 short sample videos in `data/samples/` (5-15 s, one person speaking to the camera;
  phone recordings of a teammate are ideal). Optionally one clip with no face.
- Before 03: OK for the model downloads (the agent lists real sizes; each < 1 GB, ~2 GB in total).
- Before 06: a Colab session with a T4, and the sample clips uploaded to Drive as `samples.tar`.

## Tips
- Keep the 8 GB RAM in mind: close Chrome tabs before Step 3 (feature extraction).
- If the agent proposes a new package, it must give a pinned version and a reason; that is your call.
- The prompts are plain Markdown: if a requirement changes, edit the prompt file and commit it,
  so all three teammates run the same instructions.

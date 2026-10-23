---
name: dfg-progress
description: Show the DeepFusionGuard prototype progress tracker (milestones, status, commit IDs) from docs/PROGRESS.md, and the next action.
---

# /dfg-progress - progress tracker

Follow AGENTS.md. Read docs/PROGRESS.md, print its milestone table, then state the next action in one line
(e.g. "Next: M2 Features - waiting for OK on the plan").

If docs/PROGRESS.md does not exist, re-create it from the template in
docs/agent_prompts/PROTOTYPE_MASTER_PROMPT.md (section "Progress tracker") and say so.
That is the only file this command may write.

Allowed statuses: `todo` | `in progress` | `waiting for OK` | `done`. Then STOP.

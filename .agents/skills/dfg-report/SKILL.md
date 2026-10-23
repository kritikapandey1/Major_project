---
name: dfg-report
description: Produce the DeepFusionGuard RESULTS REPORT for the current milestone from real command output, checking every acceptance criterion with evidence.
---

# /dfg-report - milestone RESULTS REPORT

Follow AGENTS.md. Use only real output from this session; if a command's output is not in context,
re-run the command. Never invent numbers.

1. Read docs/PROGRESS.md to find the current milestone. Its acceptance criteria are in
   docs/agent_prompts/PROTOTYPE_MASTER_PROMPT.md.
2. If code changed since the last /dfg-verify, run /dfg-verify (or its steps) first.
3. Output the RESULTS REPORT exactly in the AGENTS.md format. Mark each acceptance criterion:
   - `[x]` with evidence: a number, tensor shape, file path or image path
   - `[ ]` with the reason it is not met and a proposed fix

   Label toy results "TOY DATA - not a project result".
4. End with "Waiting for your OK. Proposed commit message: ..." and STOP.

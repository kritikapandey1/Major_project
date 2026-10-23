# DeepFusionGuard prototype - progress

Maintained by the agent (`/dfg-progress`, `/dfg-checkpoint`). Commit IDs go in the weekly log.

| Milestone | Status | Commit | Date | Demo command | Acceptance (short) |
|-----------|--------|--------|------|--------------|--------------------|
| M0 Audit + fixes | done | ff92efc | 2026-10-23 | `/dfg-scan` | audit report shown, chosen fixes applied, tests green, repo layout reconciled |
| M1 Preprocessing | todo | | | `scripts/preprocess.py --input data/samples/` | 16 frames, face + mouth crops, 16 kHz wav, transcript, metadata.json, preview.png |
| M2 Features | todo | | | `scripts/extract_features.py --all` | cached .npy with correct shapes, one model at a time, second run skips |
| M3 Model modules | todo | | | `scripts/model_demo.py` | alpha sums to 1, missing modality = 0 %, no NaN, tests pass |
| M4 Metrics + toy training | todo | | | `scripts/train_fusion.py --run-name proto` | Acc/AUC/F1/EER, CSV log, best.pt, resume works (TOY DATA) |
| M5 Supervisor demo | todo | | | `scripts/demo.py --input <video>` | HTML report + local web upload page + docs/DEMO.md |

Statuses: `todo` | `in progress` | `waiting for OK` | `done`

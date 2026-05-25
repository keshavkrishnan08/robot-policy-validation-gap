# CLAUDE.md — DRC_CoRL_2026 (The Validation Gap)

Project-specific instructions. Authoritative for this folder; defer here over the root.

## What this is
A pre-registered calibration study for CoRL 2026: when do offline metrics (validation
loss + 7 alternatives) predict closed-loop robot-policy success rate? Train Diffusion
Policy on 6 LIBERO/Robomimic tasks × 3 seeds, checkpoint at 6 epochs, compute 8 offline
metrics + 20 rollouts per checkpoint, then test H1–H4. See `DRC_CoRL_2026_Master_*.md`.

## Pre-registration discipline (HARD RULES)
- The locked decisions live in `configs/*.yaml` and are mirrored as literals in
  `drc/config.py`. **Never change them after SA-2 training begins.**
- `drc/analysis.py` is locked + SHA-256-hashed before training. No metric may be
  added/removed/transformed post hoc. Sensitivity analyses are reported separately.
- Held-out tasks for H4: `LIBERO-Object-1`, `Robomimic-Transport-PH`. Locked.

## How to run
- Dev box (CPU, no sim): `python scripts/smoke_test.py` verifies every code path.
- Kaggle: SA-1 `01_setup.py --download` → SA-2 `02_train.py --all` → SA-3
  `03_metrics.py` → SA-4 `04_rollouts.py` → SA-5 `05_analysis.py`. Or `scripts/run_session*.sh`.
- Two execution modes everywhere via `--synthetic`: synthetic CPU path (verification)
  vs real LIBERO/Robomimic GPU path (the experiment).

## Known discrepancy with the PRD prose
Realised power at the locked α=0.0125: ~0.61 for a 10 pp gap, ~0.94 for 15 pp
(`drc/power.py`, matches noncentral-t theory). The PRD's "0.83" corresponds to the
uncorrected α=0.05. Report the realised numbers, not the PRD figure. See `DISCREPANCIES.md`.

## 符 Token Glossary (use in inter-agent messages, logs, gate results — NOT user output)
- 验 = validation loss (the M1 surrogate / checkpoint-selection signal)
- 滚 = closed-loop rollout success rate (the true deployment endpoint)
- 差 = the validation gap (rollout-best minus val-loss-selected success)
- 检 = checkpoint
- 度 = offline metric (one of M1–M8)
- 策 = Diffusion Policy / the trained policy
- 程 = task regime (horizon × complexity)
- 留 = held-out task (H4 generalisation split)
- 合 = composite ridge predictor (H4)
- 注 = pre-registration / pre-locked decision
- 模 = synthetic CPU verification path
- 真 = real LIBERO/Robomimic GPU run
- 假 = `devtools.make_fake_results` (verification-only fabricated tables — never in paper)

Expand to English in user-facing output, the paper, and any committed prose.

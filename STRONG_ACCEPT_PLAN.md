# Path to a CoRL Strong Accept — DRC Validation Gap

**Date:** 2026-05-25 (protocol revised before any training; pre-registration re-locked)
**Constraint:** modest extra T4 compute; no hardware.
**Honest ceiling:** these moves maximise probability; a sim-only paper rarely gets a
*guaranteed* strong accept, but Tier 1+2 make it a solid accept and Tier 3, if the public
data cooperates, is what earns a champion.

---

## Why it's borderline today
1. Single architecture (Diffusion Policy) — the most-cited fatal flaw.
2. 6 tasks → only 2 per horizon bucket; H2 (the novel regime claim) is under-supported.
3. Findings partially pre-empted (SIMPLER, RoboEval, NeME).
4. Sim-only, no real-robot evidence.
5. H1 power ~0.61 at 10 pp.

## Tier 1 — Rigor (fits the compute budget)
- **Second architecture: ACT** (autoregressive CVAE transformer). Showing the gap and the
  metric rankings are *architecture-general* removes the biggest objection. Implemented in
  `drc/policy/act_policy.py`; selectable via `--arch act`.
- **8 tasks** spanning regimes (was 6): adds Robomimic-Lift-PH (short/low) and
  Robomimic-Can-PH (medium/med) to reach 3/3/2 horizon coverage.
- The analysis now treats a "run" as (task, seed, architecture) → 48 runs (was 18),
  roughly tripling N and lifting H1/H3 power above 0.8 for a 10 pp gap.
- Compute: 8 tasks × 3 seeds × 2 architectures = 48 training runs (~2.7× original).

## Tier 2 — Championable contribution (≈ free)
- **OffSel-Bench**: release the toolkit (pip-installable `drc` package) + the full
  (checkpoint × 8 offline metrics × 20-rollout success) dataset as a reusable artifact.
  Others test new offline metrics against ground-truth success *without retraining*.
  This is the lever most likely to earn a champion and the one fully under our control.
- **Scientific headline:** the coherence hypothesis (local vs global metrics) explains the
  gap; causal-selection gives the deployable rule ("select by M5, recover X% of the oracle").

## Tier 3 — Real-robot validation without a robot (the reach)
- **RoboArena** (arXiv:2506.18123) publishes real-robot rankings of 7 public generalist
  policies on DROID; **SimplerEnv-OpenVLA** pairs sim and real for RT-1/Octo/OpenVLA.
- Compute the architecture-agnostic offline metrics (M1, M3, M8, generalised M4) on those
  public policies and test whether they predict the *real* ranking. If they do, the paper is
  no longer sim-only — it is "offline metrics validated against independent real-robot
  outcomes." Caveat: this validates cross-*policy* ranking, not cross-*checkpoint* selection,
  and depends on the checkpoints being downloadable and runnable. Decision: go/no-go after a
  feasibility check on checkpoint availability.

## Realistic probability
- Today: ~30–40% main track.
- Tier 1 + 2 done, headline result positive: ~55–70% main track, plausible strong-accept.
- Tier 1 + 2 + 3 with a positive external-validation: genuine strong-accept territory.

## What we cannot fix without hardware
A fraction of reviewers will always dock a sim-only paper. Tier 3 is the only thing that
truly neutralises it; otherwise lean on the "calibration is what a real-robot user needs
*before* committing to hardware eval" framing and the SIMPLER/RoboArena sim-real correlation
evidence.

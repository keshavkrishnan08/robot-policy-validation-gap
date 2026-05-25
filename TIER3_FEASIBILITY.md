# Tier 3 Feasibility — Real-Robot Validation Without a Robot

**Date:** 2026-05-25
**Question:** Can we validate the offline metrics against real-robot outcomes using public
data, so the paper escapes "sim-only"?
**Verdict: GO — via SIMPLER (not RoboArena).** Feasible, with bounded engineering risk.

---

## The decisive facts

1. **The baseline to beat is published.** SIMPLER (Li et al. 2024, Table I) reports the
   correlation between *validation MSE* and *real-world* success across policies:
   **Pearson r = 0.308** (MMRV 0.375). SIMPLER's own sim hits r = 0.924. Our claim becomes:
   *a no-rollout offline metric (M5/coherence) predicts real success better than validation
   MSE's r = 0.308* — recovering part of what SIMPLER needs a full simulator for.

2. **The comparison set is public.** SIMPLER ranks 6 checkpoints on Google Robot tasks
   (RT-1 Converged/15%/Begin, RT-1-X, RT-2-X, Octo-Base) plus Octo-Small on BridgeData V2.
   All checkpoints are public; OpenVLA (HF `openvla/openvla-7b`) adds a 7th, stronger policy.

3. **The inference is already wrapped.** `DelinQu/SimplerEnv-OpenVLA` runs RT-1, RT-1-X,
   Octo, OpenVLA, Pi0, CogACT in one repo. We add metric computation on top of existing
   inference rather than writing it from scratch.

4. **Real-robot ground truth exists.** SIMPLER's paired sim-real evaluations provide the
   per-policy real success used as the target (reported via their released data/figures).

---

## The experiment

- **Validation set:** held-out Open X-Embodiment demos in each policy's eval domain
  (Fractal/RT-1 for Google Robot; BridgeData V2 for WidowX).
- **Offline metrics, architecture-agnostic subset:** M1 (action L1 vs demo actions),
  M3 (sampling entropy, where the policy supports sampling), M8 (action confidence),
  M5 (open-loop replay distance — computable because SimplerEnv *is* the sim).
  M4 (latent Mahalanobis) is descoped: cross-architecture encoder access is messy.
- **Ground truth:** per-policy real-world success rate (from SIMPLER) and SIMPLER-sim success.
- **Test:** Spearman/Pearson between each offline metric and real success across the policies;
  compare to the published validation-MSE baseline (r = 0.308). Coherence metrics (M5) are
  predicted to win.

**What this proves:** the recommended offline metric predicts *real-robot* policy ranking on
an independent public benchmark. The abstract can then say "validated against real-robot
outcomes," neutralising the sim-only objection for most reviewers.

---

## Honest caveats (state them in the paper)

1. **Cross-policy, not cross-checkpoint.** This validates "offline metrics rank policies by
   real success" (SIMPLER's exact setup), not the within-run checkpoint-selection gap (H1).
   Frame as a corroborating external-validation appendix, not a replacement for the main study.
2. **Small n (6–7 policies).** Same n as SIMPLER's published comparison. The baseline (r=0.308)
   is fixed and beatable, so a clear improvement is still meaningful; report with the caveat.
3. **M4 descoped** for cross-architecture reasons; external validation covers M1/M3/M5/M8.

---

## Engineering plan (de-risked)

The only real risk is the dependency mess (TF 2.15 + JAX 0.4.20 + PyTorch 2.3 cannot coexist
cleanly). Mitigation — **run each policy family in its own environment, dump per-policy metric
JSONs, then combine offline:**

1. Env A (TF): RT-1 ×3, RT-1-X → `metrics_<policy>.json`
2. Env B (JAX): Octo-Base, Octo-Small → JSON
3. Env C (PyTorch): OpenVLA-7B, RT-2-X → JSON   (OpenVLA-7B fits on one T4 in bf16, ~14 GB)
4. Combine + correlate offline (CPU): `scripts/06_external_validation.py`

Compute: ~1 Kaggle session per env (3 sessions), light per-policy (a few hundred demo obs for
M1/M3/M8; a few dozen replays for M5). Within "some extra compute."

## Code scaffolding shipped now
- `drc/external_validation.py`: architecture-agnostic metric computation given a generic
  `PolicyAdapter` (just needs `predict(obs)` and optional `sample(obs)`), plus the
  correlate-with-real-success analysis (tested on synthetic numbers).
- `scripts/06_external_validation.py`: combines per-policy metric JSONs and runs the
  correlation analysis against the published real-success targets.
- Per-framework adapters are documented stubs (Kaggle-only) following the wrapper APIs.

## Recommendation
**Greenlight.** Start with the PyTorch env (OpenVLA + RT-2-X) since it is the cleanest, prove
the metric-vs-real-success correlation on that subset, then add the TF/JAX policies to grow n.
Even the PyTorch-only subset (OpenVLA + RT-2-X + any reproduced RT-1) gives a first real-robot
data point for the abstract.

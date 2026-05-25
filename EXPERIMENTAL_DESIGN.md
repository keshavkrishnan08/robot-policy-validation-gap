# Optimal Experimental Design — Kaggle-only, aimed at a strong accept

**Date:** 2026-05-25
**Constraint:** Kaggle dual-T4, 12h sessions, no real robot.
**Principle that changed everything:** the paper now has a theory (compounding bound +
identifiability limit). So the experiments must **test the theory's predictions**, not merely
report a gap. Theory + an experiment that confirms its predicted functional form is a far
stronger CoRL profile than a calibration survey. Every part below is chosen to confirm or falsify
a specific prediction.

## The two predictions the experiments must hit
- **P1 (functional form):** the validation gap grows with horizon $H$ and closed-loop gain $L$
  along $(L^H-1)/(L-1)$.
- **P2 (identifiability / Prop 1):** rollout-free metrics (M1–M4, M6, M8) lose the ability to rank
  checkpoints as $L\to{>}1$, while environment-querying metrics (M5, M7) keep it. There is a
  regime threshold — the "horizon wall."

The headline figures are: (i) gap vs the amplification factor $(L^H-1)/(L-1)$; (ii) metric-class
ranking power (Spearman with success) vs estimated $L$, showing rollout-free metrics collapsing
past $L\approx 1$ while M5 stays high.

---

## Part A — Controlled $L$-sweep (validates the theory cleanly). ~CPU, near-free.
On real tasks $L$ is unknown and confounded with everything else, so you cannot test P1 in
isolation. Use a controlled dynamical system whose gain $L$ you **set directly** (a linear/affine
reacher with a tunable feedback gain; the existing synthetic env, extended).
- Sweep $L \in \{0.7, 0.9, 1.0, 1.1, 1.3, 1.6\}$ at fixed horizon, and $H \in \{$short, med, long$\}$
  at fixed $L$.
- Train short policies, compute all 8 metrics + rollout success, fit gap vs $(L^H-1)/(L-1)$.
- **Buys:** a clean confirmation that the gap follows the predicted curve and that rollout-free
  metrics fail precisely at $L>1$ — the direct empirical validation of Prop 1. This is what makes
  reviewers believe the theory rather than nod at it.
- **Why it is not "just a toy":** you cannot vary $L$ alone on a real robot; controlled validation
  of a bound is standard and expected. The real-task study (Part B) carries external validity.

## Part B — Real-manipulation calibration (the core study). ~3–4 Kaggle sessions.
The pre-registered protocol, already built: 8 LIBERO/Robomimic tasks × {Diffusion Policy, ACT} ×
3 seeds × 6 checkpoints = 288 checkpoints; 8 offline metrics; 20 rollouts each.
- **New and essential — estimate $L$ per task empirically.** From a handful of perturbed-init
  rollouts, measure how fast trajectories diverge (a finite-time Lyapunov estimate:
  $\hat L = (\delta_H/\delta_0)^{1/H}$). This places each real task on the theory's $x$-axis, so
  Part B reproduces Part A's curve with real policies. Reuses the perturbation rollouts (below).
- Runs H1–H4, the coherence (local-vs-global) test, and the causal-selection analysis.
- **Buys:** the robotics-grade evidence that the theory's predictions hold for modern visuomotor
  policies across two architectures; the actionable selection rule; the released benchmark.

## Part C — Real-robot validation without a robot (the sim-only escape). ~2–3 sessions.
The single biggest lever, hardware-free (see `TIER3_FEASIBILITY.md`). Compute the
architecture-agnostic metrics on SIMPLER's public policies (RT-1/RT-1-X/Octo/OpenVLA) and show they
predict the **published real-world** success ranking better than validation MSE (SIMPLER's
$r=0.308$). Run each framework in its own session (TF/JAX/PyTorch), combine offline.
- **Buys:** the line "our offline metrics predict real-robot policy ranking on an independent
  public benchmark" — the one sentence that neutralises "sim-only" for most reviewers.
- **Risk:** heterogeneous frameworks; cross-policy not cross-checkpoint (state this honestly).

## Supporting experiments (cheap, high value)
- **Perturbation–gap link** (~40 rollouts): do val-loss-selected checkpoints degrade *more* under
  small init perturbations? Connects the gap to the LIBERO-PRO robustness story, and the same
  perturbation rollouts feed the $\hat L$ estimate. Two birds.
- **Causal selection** (free, post-hoc): expected deployed success under each metric-as-rule vs
  validation loss — the practitioner headline.
- **Sensitivity** (subset): rollout count (20 vs 40) and DDIM steps (4 vs 16) on a few checkpoints,
  to preempt "20 rollouts too few / sampling artifacts."

---

## Compute budget (Kaggle dual-T4)
| Part | Sessions | Note |
|------|----------|------|
| A controlled $L$-sweep | ~0 (CPU, hours) | run locally or on a Kaggle CPU kernel |
| B training (48 runs) | 3–4 | per-task train→metrics→rollouts→prune (storage-safe) |
| B perturbation + $\hat L$ | folded into B | reuse rollout harness |
| C external validation | 2–3 | one per framework; OpenVLA-7B fits one T4 (bf16) |
| **Total** | **~6–8 sessions** | feasible over ~2 weeks |

## Priority order (do in this sequence; each is shippable)
1. **Part B core** (H1–H4 + coherence + causal selection) — the paper exists at this point.
2. **$\hat L$ estimation + the two headline figures** (gap vs amplification; metric-class vs $\hat L$)
   — this is what converts "calibration study" into "theory confirmed."
3. **Part A controlled sweep** — clean validation of the bound and Prop 1.
4. **Part C external real-robot validation** — the strong-accept multiplier.
5. Perturbation + sensitivity — robustness, reviewer-proofing.

## What this package delivers for the review
- A **theory** (bound + identifiability limit) with a **predicted functional form**, **confirmed in
  a controlled sweep** and **reproduced on real manipulation** across **two architectures**.
- A **practical rollout-free selection rule** with a measured advantage over validation loss.
- **Real-robot external validation** of the metrics on an independent public benchmark.
- A **pre-registered, released benchmark** others reuse.
- Power > 0.95; honest limitations.

## Honest ceiling
This is the maximal Kaggle-only shot, and it is a genuine strong-accept *profile*. But two things
are still true: (1) "strong accept" is never guaranteed for a sim-only paper — Part C mitigates,
does not erase, the concern; (2) the verdict ultimately depends on the data — the design is built so
that a positive result is a clean strong-accept story and a negative result is a well-powered,
theory-grounded impossibility result (the "horizon wall"), which is the next-best outcome. The one
thing that would push past the ceiling — a few real-robot rollouts of your own — is the only piece
Kaggle cannot buy.

## New code needed (small)
- `drc/dynamics_sweep.py`: tunable-gain controlled system + $L$-sweep driver (Part A).
- `drc/lyapunov.py`: $\hat L$ estimator from perturbed-init rollouts (Part B).
- wire both into `05_analysis.py` figures (gap-vs-amplification; metric-class-vs-$\hat L$).
Everything else (training, metrics, rollouts, perturbation, external validation, analysis) is built.

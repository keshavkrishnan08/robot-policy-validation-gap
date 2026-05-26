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

## ✅ Part A status (redesigned 2026-05-25): DONE, clean results
The first realization (image policy on a saturating tracking task) was degenerate — for $L>1$ the
bounded-image state saturated, trivializing tracking. **Redesigned** as a faithful numerical
validation on a linear system that exactly meets the bound's assumptions (`drc/bound_validation.py`,
`scripts/run_partA_bound.py`). The theory was strengthened to a matched pair (Thm 1 impossibility lower bound + Thm 2 achievability
upper bound for offline selection) with two corollaries; all four validated. Real results
(`results/partA_bound.json`, CPU, seconds):
- **P1 — bound tightness:** measured deviation $=\varepsilon(L^H-1)/(L-1)$ to machine precision, $R^2=1.0$.
- **P2 — identifiability:** validation loss correlates $0.31$ with success; environment-querying
  replay $0.84$. Within a fixed gain band validation loss works ($0.68$), pooled it fails ($0.31$).
- **Thm 1 vs Thm 2 — selection regret:** rollout-free validation-loss selection $0.63$ (≈ random
  $0.60$) vs oracle $0.99$ → regret $0.37$; environment-querying replay selection → regret $0.00$.
- **Cor 1 — horizon wall:** predicted $H^\star=12.8$ vs empirical $13$ (matches within one step).
- Figures: `figures/partA/figA1`–`figA4`. In the paper §4 (Thm 1, Thm 2, Cor 1, Cor 2).
- **Scope:** controlled validation of the mechanism on a system meeting the assumptions; NOT a
  real-robot result. The manipulation evidence is Part B. The old image-sweep
  (`run_partA_sweep.py`) is superseded but kept.

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

### Full design: ~35–40h compute (~6–8 sessions)
| Part | Compute | Note |
|------|---------|------|
| A controlled $L$-sweep | ~3h CPU | local or Kaggle CPU kernel |
| B training (48 runs) | ~12h GPU | 8 tasks × 2 arch × 3 seeds |
| B metrics (288 ckpts) | ~4h CPU | M5 sim-replays dominate |
| B rollouts (5,760) | ~8h CPU | K=10 averaging is the cost |
| C external validation | ~8h | 3 frameworks (TF/JAX/PyTorch) |
| perturbation + sensitivity | ~4h | |

### LEAN design (recommended): ~18–20h compute (~3–4 sessions)
Four trims, each low-cost:
1. **Rollout K=10 → K=1** (deterministic, fixed seed): 8h → ~1h. A single fixed-seed DDIM
   sample is a valid deterministic rollout; the averaging was overkill. ~No accuracy cost.
2. **Seeds 3 → 2**: n=48 → 32 runs; H1 power 0.985 → ~0.95 (still well-powered). The two
   architectures already give four runs per task.
3. **Part C → PyTorch-only** (OpenVLA + RT-2-X): ~2 fewer sessions; ~3 real policies, still
   beats SIMPLER's r=0.308 baseline.
4. **Defer sensitivity ablations** to the rebuttal.

| Part | Lean compute |
|------|--------------|
| A controlled $L$-sweep | ~2h CPU |
| B training (32 runs) | ~8h GPU |
| B metrics (192 ckpts) | ~3h CPU |
| B rollouts (K=1) | ~1h CPU |
| B perturbation + $\hat L$ | folded in (~1h) |
| C external (PyTorch only) | ~4h, 1 session |
| **Total** | **~18–20h, 3–4 sessions** |

**Do not cut:** 8 tasks (the $L$-spread is the theory's x-axis), both architectures (single-arch
is the fatal flaw), Part A sweep (validates the bound, nearly free), $\hat L$ estimation. Cutting
any of these removes a load-bearing piece of the strong-accept profile.

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

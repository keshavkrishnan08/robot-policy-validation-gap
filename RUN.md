# How to run the experiments — split into three parts

Lean design (~18–20h; see `EXPERIMENTAL_DESIGN.md`). Each part is independently shippable;
do them in order, and the paper gets stronger after each.

## Pre-flight (every session, seconds, CPU)
```bash
python scripts/preflight.py        # 9-point readiness gate
```

---

## Part A — controlled validation of the bound (DONE). CPU, seconds, no GPU.
Numerical validation on a linear system meeting the bound's assumptions.
```bash
python scripts/run_partA_bound.py
# -> results/partA_bound.json ; figures/partA/figA1_bound_tightness.pdf, figA2_identifiability.pdf
```
Results: P1 bound tightness R^2=1.0; P2 identifiability val-loss 0.31 vs replay 0.84 correlation
with success. Scope: controlled validation of the mechanism, NOT a real-robot result (that is Part B).
(The earlier image-based `run_partA_sweep.py` is superseded — its bounded-image task saturated for L>1.)

---

## Part B — real manipulation calibration (the core study). 3–4 Kaggle dual-T4 sessions.
8 LIBERO/Robomimic tasks x {Diffusion Policy, ACT} x 3 seeds; 8 metrics; 20 rollouts (K=1).
Per-task interleave with checkpoint pruning keeps peak storage ~5GB.
```bash
# Session 1 (LIBERO):    setup + download + train/metrics/rollouts/prune per task
bash scripts/run_session1.sh
# Session 2 (Robomimic): same
bash scripts/run_session2.sh
# L-hat per task (reuses perturbation rollouts) + analysis + figures + paper macros
python scripts/05_analysis.py
python scripts/fill_paper.py
```
To trim to ~18h: set `seeds: [0, 1]` in `configs/train.yaml` (N=32 runs, power ~0.95).

The headline figures (gap vs amplification on real tasks; metric-class vs L_hat) come from
the per-task L_hat estimate (`drc/lyapunov.py`) joined with the gap/Spearman tables.

---

## Part C — real-robot external validation (the sim-only escape). 1 Kaggle session, PyTorch.
```bash
bash scripts/run_partC_external.sh   # prints the exact steps
# per-policy: write external/<policy>_metrics.json ; fill external/real_success.json from SIMPLER
python scripts/06_external_validation.py
```

---

## Finalize
```bash
python scripts/fill_paper.py
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
Then insert the one results sentence in the abstract (see `paper/ABSTRACT.md`) using the real
number, and — only after Part C — the real-robot validation line.

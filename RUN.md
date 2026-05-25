# How to run the experiments — split into three parts

Lean design (~18–20h; see `EXPERIMENTAL_DESIGN.md`). Each part is independently shippable;
do them in order, and the paper gets stronger after each.

## Pre-flight (every session, seconds, CPU)
```bash
python scripts/preflight.py        # 9-point readiness gate
```

---

## Part A — controlled L-sweep (validates the theory). CPU, ~2h, no GPU.
Confirms P1 (gap tracks the (L^H-1)/(L-1) amplification) and P2 (rollout-free metrics lose
ranking power past L=1; environment-querying metrics keep it). Produces figs 5 and 6.
```bash
python scripts/run_partA_sweep.py --L 0.7,0.85,0.95,1.0,1.1,1.3,1.6 --seeds 3 --epochs 5,15,40,80
# outputs: results/sweep_summary.csv, figures/partA/fig5_*, fig6_*
```

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

# How to run the experiments — split into three parts

Lean design (~18–20h; see `EXPERIMENTAL_DESIGN.md`). Each part is independently shippable;
do them in order, and the paper gets stronger after each.

**On Kaggle, use the notebooks** (`notebooks/`, one per stage, each pulls the code from GitHub):
`00_smoke_test` (CPU) → `01_session1_libero` → `02_session2_robomimic` → `03_partC_external`.
See `notebooks/README.md` for the GitHub-push and `REPO_URL` setup. The commands below are the
equivalent CLI for a local/own-GPU run.

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

## GPU note (read first): using both T4s
A single Python process only uses `cuda:0` — that is why "the second T4 won't activate." To use
both you must run **one process per GPU**, which the session scripts do (pinning via
`CUDA_VISIBLE_DEVICES`). On Kaggle: Settings → Accelerator → **GPU T4 x2**. The scripts print
`torch sees N GPU(s)` at the start — confirm it says **2** before committing the session; if it
says 1, the accelerator is set wrong (or you have a single T4) and the run falls back to serial.

**Timing (estimates, ±50%):**
| Hardware | seeds | runs | training | total | sessions |
|----------|-------|------|----------|-------|----------|
| dual T4 (both active) | 3 | 48 | ~12h | ~17h | ~2 |
| dual T4 (both active) | 2 | 32 | ~8h  | ~12h | ~1–2 |
| single T4 | 2 | 32 | ~16–22h | ~22–28h | ~3 |
| single T4 | 3 | 48 | ~24–32h | ~33–40h | ~3–4 (2 weeks of quota) |

Recommendation: dual T4 + 3 seeds (n=48, power 0.985). If only one T4 activates, set
`SEEDS="0 1"` (n=32, power 0.90) — and change the paper's seed/run/power numbers to match.

## Part B — real manipulation calibration (the core study).
8 LIBERO/Robomimic tasks x {Diffusion Policy, ACT} x 3 seeds; 8 metrics; 20 rollouts (K=1).
Per-task interleave with checkpoint pruning keeps peak storage ~5GB.
```bash
# Session 1 (LIBERO): auto-detects GPUs, one process per GPU, per-task train/metrics/rollouts/prune
bash scripts/run_session1.sh            # or:  SEEDS="0 1" bash scripts/run_session1.sh  (single T4)
# Session 2 (Robomimic): same
bash scripts/run_session2.sh
python scripts/05_analysis.py && python scripts/fill_paper.py
```
If a 12h session times out mid-run, just relaunch — completed tasks are saved (CSV append-merge),
and you can resume the rest with `03_metrics.py --tasks <name>` etc. or by re-running (trained
tasks are pruned, so re-run only the remaining ones).

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

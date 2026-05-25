# Discrepancies found while building the pipeline

Logged honestly during implementation. None block the experiment; all should be
reflected in the paper's methods/limitations.

## 1. H1 power — RESOLVED by the 48-run (two-architecture) design
- **Original concern (6 tasks × 3 seeds = 18 runs):** power ≈ 0.61 for a 10 pp gap
  at α=0.0125, vs the PRD's optimistic 0.83 (which used the uncorrected α=0.05).
- **Current design (8 tasks × 3 seeds × 2 architectures = 48 runs):**
  - delta=10 pp, sigma=15, α=0.0125 → power ≈ **0.985**
  - delta=15 pp → power ≈ **1.0**
  - H3 paired Wilcoxon, margin=0.08, α=0.00179 → power ≈ **0.99**
- **Why it changed:** treating a "run" as (task, seed, architecture) tripled N from
  18 to 48. Power scales accordingly. The single-architecture and low-power
  weaknesses are addressed by the same design change.
- **Action:** report the realised power curve (`drc/power.py`, now defaulting to
  N_RUNS=48). The protocol is now well-powered even for a 10 pp gap.

## 2. M2 (delta-MSE) definition is dimensionally loose
The PRD pseudocode subtracts the current proprio pose from the action chunk. Action
and proprio live in different spaces and may differ in dimensionality. The
implementation (`drc/metrics.compute_m2`) follows the locked spec but clips to the
shared leading dims `min(action_dim, proprio_dim)`. Documented as the operational
definition; reported as-is.

## 3. M5/M6 signing
M5 (open-loop replay distance) and M6 (inter-seed disagreement) are stored as raw
"lower-is-better" quantities. `drc/analysis.LOWER_IS_BETTER` includes them so the
signed Spearman is oriented consistently. `configs/stats.yaml` lists M3/M8 as the
only higher-is-better metrics; the code in `drc/config.py` is authoritative.

## 4. Synthetic vs real
Everything runs in two modes. The synthetic path verifies code correctness on CPU;
it is NOT the experiment. `drc/devtools.make_fake_results` fabricates result tables
purely to exercise the stats/figures code — these numbers must never enter the paper.

# Discrepancies found while building the pipeline

Logged honestly during implementation. None block the experiment; all should be
reflected in the paper's methods/limitations.

## 1. H1 power is lower than the PRD states
- **PRD claim (section 11.1):** "For delta = 10, the empirical power is approximately 0.83."
- **Realised (`drc/power.py`, 10k sims, n=18):**
  - delta=10 pp, sigma=15, α=0.0125 → power ≈ **0.61**
  - delta=15 pp, sigma=15, α=0.0125 → power ≈ **0.94**
  - H3 paired Wilcoxon, margin=0.08, sd=0.10, α=0.00179 → power ≈ **0.49**
- **Why:** 0.83 matches the *uncorrected* α=0.05; the locked test uses the
  Bonferroni α=0.0125. The simulation agrees with noncentral-t theory
  (Cohen's d=0.67, ncp≈2.83, t_crit≈2.46 → ~0.61–0.65; Wilcoxon slightly below t).
- **Action:** the protocol is well-powered for a 15 pp gap (the magnitude the
  field reports anecdotally) but only ~0.61 for exactly 10 pp. Report the realised
  power curve, not the PRD's single optimistic number. Consider that H3 is the
  weakest-powered test; treat a null H3 cautiously rather than as strong evidence.

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

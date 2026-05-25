# Idea Upgrades for CoRL Novelty and Significance

**Date:** 2026-05-25
**Input:** Full lit review (15 verified sources); searches confirming DRC's niche is open
**Purpose:** Actionable upgrade options ranked by impact vs experimental cost

---

## Executive summary

The paper's premise and niche are solid. Three upgrades — one conceptual, one analytical, one experimental — would each materially raise the CoRL acceptance probability. The conceptual upgrade (coherence hypothesis) is the highest-ceiling move. The analytical upgrade (causal selection analysis) is the easiest to implement and directly addresses what reviewers will care about most. The experimental upgrade (perturbation gap) is the highest-significance finding if positive.

---

## Upgrade 1 (Conceptual) — The coherence hypothesis  ★★★★★

**The idea:** Replace "which offline metric correlates with success?" with a falsifiable mechanistic claim: *offline metrics that measure global trajectory coherence predict deployment success better than metrics that measure local single-step action accuracy.*

**Why it elevates the paper:**
The current paper measures an effect. This upgrade explains it. The gap exists because validation loss (M1) measures local, single-step action prediction error. But deployment success requires multi-step coherence: the policy must follow a self-consistent trajectory from start to goal without compounding errors that push it off the training manifold. Metrics that capture global coherence — M4 (latent Mahalanobis, measuring whether the full observation sequence stays near the training distribution), M5 (open-loop replay fidelity, measuring multi-step trajectory correctness), and M7 (jerk, measuring trajectory smoothness as a proxy for self-consistency) — should systematically outperform local metrics.

**The testable prediction:** In H3, the coherence metrics (M4, M5, M7) beat M1/M2 at predicting success. In H2, long-horizon tasks show the largest gap because they provide the most opportunity for multi-step trajectory divergence from the training manifold.

**How to add it:**
- Add a "coherence vs accuracy" sub-analysis to H3: compute mean Spearman for the "local accuracy" group (M1, M2, M8) vs the "coherence" group (M4, M5, M7) and test whether the group difference is significant.
- Add one paragraph to the methodology section framing M4/M5/M7 as coherence metrics, contrasting them with M1/M2.
- Add one sentence to H2's hypothesis: "long-horizon tasks should show larger gaps because multi-step coherence failure compounds more severely."
- Cost: zero new experiments. Pure framing + one extra column in Table 1.

**Why it connects to the broader literature:**
FAIL-Detect (arXiv:2503.08558) uses the same latent OOD signal as M4 for runtime failure detection — it detects coherence failure at execution time. DRC detects it before execution, at checkpoint-selection time. The bridge to FAIL-Detect is "M4 predicts checkpoint quality for the same reason conformal OOD detection predicts runtime failure: both measure how far the policy has drifted from the training manifold." Cite FAIL-Detect in the M4 discussion.

---

## Upgrade 2 (Analytical) — Causal selection analysis  ★★★★☆

**The idea:** For each offline metric, simulate "what success rate would you achieve if you used this metric to select checkpoints?" and compare selection policies across all 6 metrics.

**Why current H3 is weaker than it looks:**
H3 measures which metric *correlates* with success (Spearman rank). But correlation ≠ selection utility. A metric could rank checkpoints correctly in 14 of 18 runs but fail in the 4 runs where the gap is largest — producing worse expected deployed performance than just using validation loss. Conversely, a metric with moderate Spearman might consistently catch the overfit checkpoints in the hard cases, giving better expected performance.

**The exact analysis:**
For each metric M and training run r, identify the checkpoint that optimises M (minimise for lower-is-better, maximise for higher-is-better). Record its success rate. Average across runs → expected success under M-based selection. Compare all 8 selection rules (including validation L1) on expected deployed success and worst-case deployed success (CVaR at α=0.10 across runs). This produces Table 2 in the paper: "Expected success rate under each selection rule."

**Cost:** zero new experiments. Entirely post-hoc analysis of the existing checkpoints × metrics × success-rate table. Add ~30 lines to analysis.py after H3.

**Why CoRL reviewers will love this:**
It answers the question practitioners actually ask: "Which metric should I use to pick my checkpoint?" Not "which metric correlates best in a statistical test" but "if I put this metric into my training loop as the selection rule, what deployed performance do I get?" The difference between these two answers is the difference between a measurement paper and a tool.

**Add to the paper as:**
- Section 5.5 "Selection policy comparison" (or part of the Discussion)
- Table 2 (selection policy × expected success) alongside Figure 1

---

## Upgrade 3 (Experimental) — The perturbation-gap connection  ★★★★☆

**The idea:** Test whether checkpoints with large validation gaps (where M1 selects badly) are also more fragile under simple perturbations.

**The hypothesis:** Validation-driven checkpoint selection picks overfit checkpoints. Overfit checkpoints should show two failure modes simultaneously: (1) lower success on canonical eval (the gap, measured by H1), and (2) lower success under slight distribution shift (novel start states, small obs noise). If these failure modes correlate, the validation gap is not just a calibration problem but a *generalization failure indicator* — which connects the DRC paper directly to the active LIBERO-PRO/Plus robustness literature.

**Implementation:**
At the end of SA-4, for 2 tasks × the val-loss-selected checkpoint × the rollout-best checkpoint, run 10 additional rollouts on **perturbed** initial conditions (start state ± 0.1 radians noise on joint angles, or object position ± 2cm). Compare success rates. 40 extra rollouts total.

Prediction: the rollout-best checkpoint degrades less under perturbation than the val-loss-selected checkpoint, because the rollout-best is less overfit. If true, the narrative becomes: "validation-driven selection does not just miss the best checkpoint — it actively selects the most brittle one."

**Cost:** 40 extra rollouts (~2 hours CPU on Kaggle). Code is already written (evaluate_checkpoint with different eval_conditions).

**Why this is CoRL-significant:**
It directly connects the DRC paper to the LIBERO-PRO/Plus robustness cluster, potentially doubling the citation surface. More importantly, it provides an *explanation* for why the field should care about the validation gap: it is not just suboptimal performance on the training-distribution test, it is selection of fragile policies.

---

## Upgrade 4 (Reference) — FAIL-Detect + runtime-failure-detection positioning  ★★★☆☆

**Add to related work:**
The runtime failure detection literature (FAIL-Detect, arXiv:2503.08558; "Failure Prediction at Runtime," arXiv:2510.09459; "Failure Identification via Statistical Filtering," arXiv:2604.13788) uses latent OOD signals and conformal prediction to detect failures while the policy is executing. DRC's M4 applies the same OOD principle at a different stage: before execution, at checkpoint selection. Cite and distinguish: "FAIL-Detect asks whether a running policy is about to fail; we ask whether a saved checkpoint is worth deploying at all."

This positions DRC in a larger lifecycle view — policy quality can be assessed offline (DRC), at the point of execution (FAIL-Detect), or post-hoc via rollout (the rollout-efficiency cluster) — which makes the contribution feel like part of a coherent programme rather than an isolated study.

**Cost:** two new bibtex entries and one paragraph addition to related work.

---

## Upgrade 5 (Framing) — Rollout-equivalent sample size  ★★★☆☆

**The idea:** Express the H4 composite predictor's performance in rollout-equivalent units.

Given the composite predictor has held-out RMSE R_composite, and a random-N-rollout estimate of success rate has expected MSE ~p(1-p)/N (where p≈0.5), compute the N such that sqrt(p(1-p)/N) = R_composite. That N is the "rollout equivalents" — the number of rollouts that would give the same prediction accuracy as the offline composite predictor. If N=3, the composite predictor is "worth 3 rollouts." If N=0.5, it's worse than a single rollout.

**Why:** CoRL reviewers and robotics practitioners think in rollout budgets. "This composite predictor has 3-rollout equivalent accuracy" is instantly interpretable. "It reduces RMSE by 20% vs baseline" is not.

---

## New related work entries required

| Paper | arXiv | Relation |
|---|---|---|
| FAIL-Detect | 2503.08558 | Same OOD signal as M4; runtime vs training-time application |
| Failure Prediction at Runtime | 2510.09459 | Extends FAIL-Detect to generative policies |
| Failure Identification via Statistical + Semantic Filtering | 2604.13788 | 2026, active area |
| "Understanding horizon in IL" | 2407.15007 | Horizon effects in BC; connects to H2 |

---

## Revised contribution statement (reflecting upgrades)

**Before:**
"We characterise the validation gap across task regimes, compare eight offline metrics, and test whether a composite predictor generalises."

**After:**
"We show that the validation gap is a symptom of trajectory-coherence failure: validation loss measures local action accuracy while deployment requires global coherence. We characterise when this gap occurs (regime-dependence), identify which metrics best capture coherence (M4/M5 vs M1/M2), and show that checkpoint selection by the best offline metric achieves X% higher expected deployed performance than selection by validation loss — and selects checkpoints that degrade Y% less under distributional shift. The full calibration framework is pre-registered."

This revised statement answers three questions: what causes the gap (coherence failure), what to do about it (use M4/M5), and how much it matters (X% performance + Y% robustness).

---

## Priority order for implementation

1. Upgrade 2 (causal selection analysis) — do first; zero cost; highest reviewer impact
2. Upgrade 1 (coherence hypothesis framing) — do with upgrade 2; zero cost; elevates entire paper
3. Upgrade 4 (FAIL-Detect related work) — one afternoon; adds depth without risk
4. Upgrade 3 (perturbation experiment) — run after Session 2 rollouts complete; 40 extra rollouts
5. Upgrade 5 (rollout-equivalent framing) — do in paper writing, not in code

The first two together take an afternoon and could shift the paper from a "nice empirical study" to one with a theoretical claim and a decision-tool deliverable — a much stronger CoRL submission.

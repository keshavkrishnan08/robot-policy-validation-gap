# Abstract — The Validation Gap (CoRL 2026)

## Title (candidates)
1. **The Validation Gap: Calibrating Offline Metrics for Robot Policy Selection**
2. Score the Journey, Not the Step: When Offline Metrics Predict Robot Policy Success
3. The Myopia of Validation Loss in Robot Policy Selection

---

## SUBMISSION ABSTRACT (current — in main.tex; concise, with theory + math, ~255 words)

Imitation-learned manipulation policies are deployed by choosing one checkpoint from the many
saved during training. Because measuring success requires costly rollouts, practitioners select by
validation loss on held-out actions, a criterion inherited from supervised learning that is
reliable there but fails on robots: the lowest-validation-loss checkpoint can be far worse than the
best achievable, in ways uncharacterized and with no systematic offline alternative. We name the
discrepancy the *validation gap* and explain it. Per-step action error ε compounds, so closed-loop
deviation from the expert obeys δ_H ≤ L_a·ε·(L^H−1)/(L−1) in the horizon H and closed-loop gain L:
validation loss estimates ε, but deployment depends on δ_H, and the two diverge as H and L grow.
We sharpen this into an identifiability limit. When L>1, no metric computed from the policy and
demonstrations alone can rank checkpoints by success, since policies that agree on the expert
distribution can differ arbitrarily off it; only a metric that queries the environment, such as an
open-loop replay, escapes the limit. This partitions offline metrics by environment access and
predicts which can work. We then supply the missing calibration: training Diffusion Policy and ACT
across eight LIBERO and Robomimic tasks, we score every checkpoint by eight metrics from both sides
of that line against rollout success, and test the predicted regime dependence, which metrics
recover success where validation loss fails, and whether a composite generalizes to held-out tasks.
To our knowledge this is the first identifiability characterization of offline policy-selection
metrics by environment access, and the first systematic calibration of such metrics against
deployment success on modern visuomotor policies. The protocol is pre-registered, needs no robot
hardware, and runs on consumer GPUs.

### Results sentence to insert after the run (NOT yet — no benchmark data exists)
Drop in before the "To our knowledge" sentence, only if the LIBERO/Robomimic run supports it:
> "We find that open-loop replay fidelity recovers the deployment ranking validation loss
> discards, most strongly on long-horizon tasks, beating validation-loss selection by [X] pp."

This single sentence is what flips a reviewer's prior from "unfinished" to "they did it and it
worked." Off the abstract alone, the version *without* it reads reject-leaning (an empirical paper
whose abstract states no empirical result); the theory raises the floor but does not substitute for
the calibration result.

---

## ALTERNATE: oral / story version (for the talk opening or a workshop; numberless)

Every imitation-learned robot policy reaches the same quiet fork: of the many checkpoints written
during training, which one do you deploy? Measuring success means running the robot, so the field
decides for free, by validation loss on held-out actions, a habit inherited from supervised
learning where it works. On a robot it does not, and the reason is structural. A classifier is
graded one example at a time; a policy is graded by a journey. The small per-step errors validation
loss forgives compound, action after action, until the robot drifts off the states it saw in
training and fails. We name this the *validation gap*, show it is compounding error amplified by a
horizon- and gain-dependent factor, and prove that in the contact-rich regime no rollout-free
metric can rank checkpoints, while a metric that touches the environment can. We then calibrate
eight offline metrics against true rollout success across two architectures and eight tasks. The
lesson: stop grading robot policies one step at a time. Score the journey, or run it.

---

## Honesty ledger — what each clause rests on (READ before submitting)

The abstract asserts only what is established in the literature, derived (theory), or guaranteed by
the pre-registered design. No experimental (benchmark) result is claimed.

| Clause | Status |
|---|---|
| "validation loss can be far worse than the best achievable" | ESTABLISHED (Robomimic 50–100%; SIMPLER). Safe. |
| "δ_H ≤ L_a·ε·(L^H−1)/(L−1)" compounding bound | DERIVED, but CLASSICAL — attribute (Ross & Bagnell; Spencer; Simchowitz; Tan 2026). Not claimed as novel. |
| "identifiability limit: no rollout-free metric can rank when L>1" | DERIVED (Prop 1). The environment-access partition is the genuinely fresh framing; honestly positioned as a selection-theoretic reading of known insufficiency results. |
| "we score every checkpoint... 2 architectures, 8 tasks, 8 metrics" | BY DESIGN (pre-registered protocol). Safe. |
| "test the predicted regime dependence / which metrics recover success" | QUESTIONS the study answers — no result asserted. Safe but reads reject-leaning (see below). |
| "first identifiability characterization / first systematic calibration" | NOVELTY claim. Empirical-calibration novelty is defensible (cite hpil2021 and differentiate). The identifiability framing is defensible; do not call the bound novel. |

**Status of results: NONE on the benchmark.** Only a toy synthetic control study has run (n=3,
2-D reacher) — directionally consistent with the prediction (coherence/replay metrics tracked
success; validation loss did not), but it is NOT a benchmark result and must not appear as one.

**To finish the abstract:**
1. Run Sessions 1–2 (Kaggle) → `scripts/05_analysis.py` → `scripts/fill_paper.py`.
2. Insert the results sentence above with the real number, only if supported.
3. After Tier 3, optionally add: "and these offline metrics predict the published real-robot
   ranking of open policies better than validation MSE." Do NOT add before the Tier-3 run.

Verified on CPU: every quantity the abstract references is computable end-to-end (gap, regime
model, local-vs-coherence comparison, selection-rule simulation, identifiability partition,
real-robot baseline wiring).

# Abstract — The Validation Gap (CoRL 2026)

## Title (candidates)
1. **Score the Journey, Not the Step: Calibrating Offline Metrics for Robot Policy Selection**
2. The Validation Gap: When Offline Metrics Fail to Predict Robot Policy Success — and When They Don't
3. The Myopia of Validation Loss in Robot Policy Selection

---

## Abstract (oral / story version, ~215 words, numberless)

Every imitation-learned robot policy reaches the same quiet fork: of the many checkpoints
written during training, which one do you deploy? Measuring success means running the robot,
so the field decides for free, by validation loss on held-out actions — a habit inherited
from supervised learning, where validation loss and test accuracy rise together. On a robot
they come apart, and the reason is structural. A classifier is graded one example at a time;
a policy is graded by a journey. The small per-step errors validation loss forgives compound,
action after action, until the robot drifts off the states it saw in training and fails.
Validation loss scores the step. Deployment scores the trajectory.

We name this the validation gap and study it as a calibration problem — the way a clinician
asks whether a cheap surrogate tracks the outcome that matters. Pre-registered and
reproducible on consumer GPUs, we score every checkpoint of policies trained across horizons,
contact regimes, and two architectures in two ways at once: offline signals that never touch
the robot, and closed-loop rollouts that measure the truth. One hypothesis organizes the
search. If the gap is born of compounding, then signals of local accuracy should fail just as
validation loss does, while signals of global coherence — how far a policy drifts from its
training manifold, whether its own rollout stays on course — should recover what validation
loss misses. The result is a calibrated map: the regimes where a free offline signal can
stand in for a rollout, and the boundary past which you must pay to run the robot. The lesson
is easy to act on. Stop grading robot policies one step at a time. Score the journey — or,
where you cannot do it offline, run it.

---

## The 3-sentence hook (talk opening / TL;DR)

Training a robot policy writes hundreds of checkpoints; the field picks which to deploy with
validation loss, a number borrowed from supervised learning. But a classifier is graded one
example at a time while a policy is graded by a journey — so the per-step errors validation
loss forgives compound into trajectories that drift off the training manifold and fail. We
name this the **validation gap**, explain it as a mismatch between *local accuracy* and
*global coherence*, and deliver a calibrated map of when a free offline signal can replace a
rollout and when it cannot.

---

## Honesty ledger — what each clause rests on (READ before submitting)

The abstract is deliberately numberless and asserts only what is either established in the
literature or guaranteed by the pre-registered design. No experimental result is claimed.

| Clause | Status |
|---|---|
| "validation loss and success come apart on robots" | ESTABLISHED (Robomimic 50–100%; SIMPLER). Safe to assert. |
| "the reason is structural — local step vs global journey" | Our FRAMING/thesis (the coherence hypothesis). Asserted as the paper's lens, not as a measured result. Safe. |
| "we score every checkpoint two ways across horizons/contact/2 architectures" | BY DESIGN (the pre-registered protocol). Safe. |
| "signals of local accuracy should fail... coherence should recover" | Stated as a HYPOTHESIS ("should"), not a finding. Safe. |
| "the result is a calibrated map / boundary" | The DELIVERABLE the design guarantees regardless of outcome. Safe. |

**Two things to harden once the run completes:**
1. If you want the abstract to land as a *definite-finding* oral (stronger), after the data is
   in, change "should recover what validation loss misses" → a stated finding (e.g., "and we
   find that coherence signals recover most of the deployment ranking that validation loss
   discards"). Only do this if H3/causal-selection actually support it.
2. To add the real-robot line (big credibility): after running Tier 3, append one sentence —
   "We confirm the surrogate's failure on physical robots, using published real-robot rankings
   of open policies." Do NOT add it before the Tier-3 run.

Everything the abstract references is computable end-to-end (verified on CPU: gap, regime
model, local-vs-coherence comparison, selection-rule simulation, outcome classification,
real-robot baseline wiring).

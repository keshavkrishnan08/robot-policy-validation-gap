# The Validation Gap

**When does offline validation loss predict a robot policy's real success rate, and when does it
lie? A pre-registered calibration study for imitation-learned manipulation policies, plus a theory
of where offline checkpoint selection has to fail.**

![In a controlled linear system that meets the theory's assumptions, validation loss correlates 0.31 with success and picks a working checkpoint 63% of the time against 59% for random, while one open-loop replay correlates 0.84 and picks a working checkpoint 99% of the time, the same as the oracle](docs/partA_selection.png)

Training an imitation policy writes many checkpoints. Only one gets deployed. Rollouts are
expensive, so the field picks the checkpoint with the lowest validation loss on held-out expert
actions, a habit inherited from supervised learning. On a robot that habit breaks: validation loss
grades one step at a time, while deployment grades a whole trajectory, and small per-step errors
compound as the policy drifts onto states the demonstrations never covered. This project treats
validation loss the way clinical trials treat a biomarker: a cheap surrogate endpoint that has to be
calibrated against the true endpoint, closed-loop success, before anyone trusts it.

```
LIBERO demonstrations (3 tasks: short, medium, long horizon)
    ->  train Diffusion Policy and ACT, 3 seeds each, checkpoints at epochs 10, 25, 50, 80, 120
    ->  8 offline metrics per checkpoint (no closed-loop rollouts)
    ->  20 closed-loop rollouts per checkpoint (the ground truth)
    ->  pre-registered tests H1 to H4, analysis code hashed before training
```

Score the journey, not the step.

## The core idea

1. **Per-step error compounds.** With per-step action error `eps`, horizon `H` and closed-loop gain
   `L`, deviation from the expert obeys `delta_H <= L_a * eps * (L^H - 1) / (L - 1)`. Validation
   loss estimates `eps`. Deployment depends on `delta_H`. The bound itself is classical and the
   paper attributes it; it is a lemma here, not the contribution.
2. **The gain lives in the environment.** `L` depends on how the world amplifies an action error at
   the states the policy itself reaches, many of them off the demonstration distribution. Any
   metric computed from the policy and the demonstrations alone (validation loss, action entropy,
   ensemble disagreement, latent distance) cannot see it.
3. **So the dividing line is environment access.** The theory predicts that when `L > 1` no
   rollout-free metric ranks checkpoints better than chance, while a single open-loop replay can
   reach oracle selection (under a feedback-margin assumption the benchmark is meant to test). The
   switch happens at a "horizon wall", `H* = log(1 + (tau / eps) * (L - 1)) / log(L)`, where `tau`
   is the success tolerance.
4. **Every outcome is pre-registered.** Tasks, metrics, rollout count, held-out split, thresholds
   and the analysis code are fixed and SHA-256 hashed before training. A six-row outcome matrix maps
   each combination of H1 to H4 results, including a clean null, to a stated conclusion.

## What it does (measured on a controlled linear system, not on robots)

Part A checks each prediction of the theory on a linear system `s_{t+1} = L * s_t + e_t` whose gain
`L` is set directly. Everything below comes from `results/partA_bound.json`, written by
`scripts/run_partA_bound.py` on a CPU in seconds.

| Prediction | Setup | Result |
|---|---|---|
| The compounding bound is tight | 30 (L, eps) pairs, H = 12 | R^2 = **1.0**, max relative error 3.9e-16 |
| Validation loss cannot rank varying-gain policies | 600 policies, eps and L drawn independently | correlation with success **0.31** (validation loss) vs **0.84** (open-loop replay) |
| Within a known gain band it partly can | the 233 policies with L >= 1 | 0.68 (validation loss) vs 0.76 (replay) |
| Rollout-free selection is near chance | 6,000 populations of 6 checkpoints, H = 18 | validation loss picks a working checkpoint 0.63 of the time, random 0.59, oracle 0.99: regret **0.37** |
| One trajectory reaches the oracle | same populations | replay regret **0.00** |
| Replay tolerates measurement noise | replay noise gamma from 0 to 1.0 | replay regret 0.00 to 0.045, validation loss 0.367 throughout |
| The horizon wall is where predicted | L = 1.15, eps = 0.03 | predicted H* = **12.8**, observed collapse at 13 |

These numbers confirm the mechanism on a system built to satisfy the theory. They are not evidence
that open-loop replay picks better checkpoints on LIBERO. That is the job of the benchmark study, and
its results are not in the repo yet.

## How it works

**Tasks.** The locked config (`drc/config.py`, `configs/tasks.yaml`) uses three LIBERO tasks that
differ mainly in episode length: LIBERO-Spatial-1 (200 steps), LIBERO-Goal-1 (500) and
LIBERO-Long-1 (800), 50 demonstrations each. LIBERO-Long-1 is held out for H4. A run is one
(task, seed, architecture) triple: 3 tasks x 3 seeds x 2 architectures = 18 runs and 90 checkpoints.

**Policies.** A self-contained Diffusion Policy (own DDPM/DDIM scheduler, 1D conditional U-net) and
ACT (a CVAE transformer), both with a ResNet-18 image encoder plus proprioception, prediction horizon
16, action chunk 8, observation horizon 2. AdamW at 1e-4, batch 64, EMA 0.9999, cosine schedule, 120
epochs. A small CNN backbone replaces ResNet-18 on the synthetic CPU path.

**Offline metrics.** M1 validation L1, M2 delta-action MSE, M3 action entropy, M4 latent Mahalanobis
distance (100 PCA components, Ledoit-Wolf shrinkage), M5 open-loop replay distance, M6 inter-seed
disagreement, M7 trajectory jerk, M8 action confidence. M5 and M7 feed the policy the expert's
observations and play its predicted actions open-loop in the simulator; the other six never touch
it. Signs are oriented so a positive Spearman always means "predictive".

**Rollouts.** Twenty closed-loop rollouts per checkpoint on initial conditions locked per task and
shared across checkpoints. Deterministic DDIM sampling with a fixed noise seed, so the only variance
is across the paired initial conditions. `drc/lyapunov.py` also estimates each task's gain `L_hat`
from perturbed-start rollouts, which places real tasks on the theory's axis.

**Analysis.** H1: one-sided Wilcoxon on the per-run gap (rollout-best minus validation-selected
success), supported only if the median gap clears 10 points at the Bonferroni-corrected
alpha = 0.0125, with a BCa bootstrap interval. H2: mixed-effects likelihood-ratio test for regime.
H3: paired Wilcoxon of each metric's per-run Spearman against M1, at alpha = 0.00179. H4: a ridge
composite fit on the two shorter tasks must cut held-out RMSE on LIBERO-Long-1 by at least 20%.

**Running on free GPUs.** The session scripts launch one training process per GPU so both Kaggle
T4s are used, download each task's data just before training it, and delete data and checkpoints
after evaluation, so peak disk stays at a few GB. MuJoCo renders offscreen through EGL. Every driver
takes `--synthetic`, which swaps in a 2D reaching task so the full pipeline runs on a CPU.

## Run it

Checks and the controlled study (CPU, no simulator):

```bash
pip install -r requirements.txt
python scripts/preflight.py           # 9-point readiness gate, seconds
python scripts/smoke_test.py          # full pipeline on synthetic data, both architectures
pytest -q
python scripts/run_partA_bound.py     # Part A -> results/partA_bound.json, figures/partA/
```

The benchmark study (Kaggle, GPU T4 x2):

```bash
bash scripts/kaggle_setup.sh          # LIBERO from source, robosuite 1.4.0
bash scripts/run_session1.sh          # per task: download, train 2 archs x 3 seeds, metrics, rollouts
python scripts/05_analysis.py         # H1 to H4, figures, results/results.json
python scripts/fill_paper.py          # write the results into paper/results_macros.tex
```

`notebooks/RUN_ALL.ipynb` runs the same sequence as one Kaggle notebook. `RUN.md` has timing
estimates and the single-T4 fallback.

## Layout

```
configs/               locked YAML: tasks, training, statistical plan
drc/
  config.py            pre-registered constants (authoritative over the YAML)
  model/               DDPM/DDIM scheduler, 1D conditional U-net, image + proprio encoder
  policy/              Diffusion Policy and ACT
  data/                sequence dataset, LIBERO and Robomimic adapters, synthetic reacher
  train.py             training and checkpointing
  metrics.py           offline metrics M1 to M8
  rollouts.py          closed-loop evaluation
  analysis.py          locked H1 to H4 tests, outcome matrix, coherence and selection analyses
  bound_validation.py  Part A controlled linear system
  lyapunov.py          finite-time gain estimate L_hat from perturbed rollouts
  external_*.py        Part C scaffolding: offline metrics on public policies vs published real success
  power.py             power analysis by simulation
  devtools.py          fake result tables for testing the statistics code only
scripts/               01_setup to 06_external_validation, preflight, smoke test, Kaggle session scripts
notebooks/             one Kaggle notebook per stage, plus RUN_ALL
paper/                 LaTeX draft, sections, bibliography, result macros (placeholders)
results/               partA_bound.json, the only committed result
checkpoints/           four 2-epoch training logs from a test run
tests/                 analysis, bound, config, external validation, metrics, model, sweep
EXPERIMENTAL_DESIGN.md, RUN.md, DISCREPANCIES.md, SESSION_LOG.md
```

## Honesty

* The benchmark study has not been run in this repo. There is no `results/metrics.csv`,
  `results/rollouts.csv` or `results/results.json`, and every result macro in
  `paper/results_macros.tex` still reads TBD. H1 to H4 are open questions.
* The only measured numbers are Part A. In that system validation loss is exactly the per-step
  error and the replay metric is the true deviation plus noise, so replay winning is close to true
  by construction. R^2 = 1.0 is likewise exact, because the simulated recurrence is the bound. Part A
  checks the mathematics; it says nothing yet about real policies.
* Scope changed after the paper was drafted. The locked config now uses three LIBERO tasks (18 runs,
  90 checkpoints). The paper text, `paper/results_macros.tex` (48 runs, 240 checkpoints) and
  `scripts/run_session2.sh` still describe the earlier eight-task design with Robomimic. They need
  to be brought in line before the paper is submitted anywhere.
* Power drops with the cut. The repo's own power simulation gave about 0.61 for a 10-point gap at 18
  runs (`SESSION_LOG.md`). The 0.985 in `DISCREPANCIES.md` and the "power above 0.95" in the paper
  belong to the 48-run design.
* Simulation only. Part C, which would test the offline metrics against SIMPLER's published
  real-robot rankings, is scaffolding with documented stubs and has not been run.
* M2 subtracts proprioception from actions over their shared leading dimensions, which is
  dimensionally loose. `DISCREPANCIES.md` logs this and the other known deviations.

MIT licensed.

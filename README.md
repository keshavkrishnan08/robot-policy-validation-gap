# The Validation Gap — DRC CoRL 2026

When does offline validation loss predict a robot policy's real success rate, and when
does it lie to you? This repo is the pre-registered, reproducible study that answers it.

We train two architectures (Diffusion Policy and ACT) on eight LIBERO/Robomimic manipulation
tasks, save six checkpoints per run across three seeds — 48 runs, 288 checkpoints — then
score every checkpoint two ways: eight offline metrics computed without touching the robot,
and twenty closed-loop rollouts that measure the truth. Four pre-locked hypotheses (H1–H4)
map which offline signals track deployment success, in which task regimes, and whether a
composite predictor generalises to held-out tasks. The framing is a surrogate-endpoint
calibration study, borrowed from clinical biomarker validation.

## Layout
```
configs/        pre-locked YAML: tasks, training, stats plan
drc/            the package
  config.py     single source of truth for every locked decision
  model/        scheduler (DDPM/DDIM), 1D conditional U-net, obs encoder
  policy/       DiffusionUnetImagePolicy
  data/         sequence dataset, synthetic generator+env, LIBERO/Robomimic adapters
  train.py      SA-2 training + checkpointing
  metrics.py    SA-3 metrics M1–M8
  rollouts.py   SA-4 closed-loop evaluation
  analysis.py   SA-5 LOCKED stats: H1–H4 + outcome-matrix classifier
  figures.py    the four paper figures
  power.py      power analysis by simulation
  devtools.py   verification-only fake-result generator (never used in the paper)
scripts/        01_setup … 05_analysis drivers, providers, smoke_test, run_session*.sh
tests/          pytest unit tests for the analysis + model code
paper/          CoRL LaTeX scaffold (results filled from results.json)
```

## Quickstart (dev box, CPU, no simulator)
```bash
pip install -r requirements.txt
python scripts/01_setup.py            # verify pre-lock consistency + environment
python scripts/smoke_test.py          # full pipeline on synthetic data + fake stats
pytest -q                             # unit tests
```

## Real run (Kaggle dual-T4)
```bash
python scripts/01_setup.py --download
python scripts/make_eval_conditions.py                       # lock 20 init conditions / task
python scripts/02_train.py --all --all_archs --device cuda   # 48 runs, 288 checkpoints
python scripts/03_metrics.py --all_archs                     # 8 offline metrics / checkpoint
python scripts/04_rollouts.py --all_archs                    # 20 rollouts / checkpoint
python scripts/05_analysis.py                                # H1–H4, figures, results.json
```
The `--synthetic` flag on every driver swaps in the CPU verification backend; `--all_archs`
sweeps both Diffusion Policy and ACT (omit it to run a single `--arch`).

## The eight metrics
M1 validation L1 · M2 delta-action MSE · M3 action entropy · M4 latent Mahalanobis ·
M5 open-loop replay distance · M6 inter-seed disagreement · M7 trajectory jerk ·
M8 action confidence.

## Pre-registration
`drc/config.py` + `configs/*.yaml` + `drc/analysis.py` are locked and hashed before any
training. See `CLAUDE.md` for the discipline and `DISCREPANCIES.md` for honest notes
(including the realised power, which differs from the PRD prose).

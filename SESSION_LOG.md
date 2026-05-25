# DRC_CoRL_2026 — Session Log

```
SESSION LOG ENTRY
=================
Date: 2026 May 25
Project: DRC_CoRL_2026
Session: Implementation session — full codebase build from the PRD
Wall clock time: ~1 session (autonomous build)

Activities:
- Built the complete experimental codebase implementing SA-1 through SA-6 of the PRD.
- Two execution modes throughout: a synthetic CPU verification path (runs on the dev box,
  no simulator/GPU) and a real LIBERO/Robomimic GPU path (for Kaggle).
- Implemented a self-contained Diffusion Policy (own DDPM/DDIM scheduler, 1D conditional
  U-net, image+proprio encoder) so the synthetic path needs no diffusers/torchvision/einops.
- Implemented all 8 offline metrics (M1-M8), closed-loop rollouts, and the PRE-LOCKED
  statistical analysis (H1-H4 + outcome-matrix classifier) exactly per PRD section 10.
- Implemented the simulation-based power analysis and the four paper figures.
- Wrote the CoRL LaTeX paper scaffold with a results-injection mechanism (fill_paper.py)
  so every reported number traces to results.json (SA-6 criterion).

Verification:
- scripts/smoke_test.py PASSES end to end: trains a tiny policy, computes all 8 metrics
  (finite), runs rollouts (synthetic env reaches the goal), and runs H1-H4 + figures + power.
- pytest: 21/21 tests pass (config, model shapes incl. 14-dim dual-arm actions, scheduler
  determinism, metrics finiteness, M6 zero-for-identical-policies, analysis structure,
  outcome-matrix rows, BCa edge cases).
- All modules compile (py_compile clean).

Decisions made (mirror the locked PRD):
- Tasks/seeds/epochs/metrics/held-out split all locked in configs/*.yaml + drc/config.py.
- analysis.py is the locked artifact; hash it before SA-2 (scripts/01_setup verifies prelock).
- M6 uses a fixed shared noise seed so disagreement isolates weights, not sampling noise.

Issues encountered / discrepancies (see DISCREPANCIES.md):
- Realised H1 power at the locked alpha=0.0125 is ~0.61 for a 10pp gap (~0.94 for 15pp),
  NOT the PRD's stated 0.83. The 0.83 corresponds to uncorrected alpha=0.05. Code is correct;
  PRD prose is optimistic. Paper reports the realised power curve and treats H3 (weakest,
  ~0.49) with caution.
- Fixed a GroupNorm divisibility bug for 14-dim (dual-arm Transport) actions.
- Fixed inference non-determinism: inference methods now force eval mode so the train-time
  random crop does not leak randomness into metrics/rollouts.

Files produced:
- drc/ package (config, seeds, utils, model/, policy/, data/, train, metrics, rollouts,
  analysis, power, figures, devtools, envs)
- scripts/ (01_setup .. 05_analysis, providers, make_eval_conditions, smoke_test,
  fill_paper, run_session1.sh, run_session2.sh)
- tests/ (test_config, test_model, test_metrics, test_analysis)
- paper/ (main.tex, sections/*, references.bib, results_macros.tex, fill_paper)
- configs/ (tasks.yaml, train.yaml, stats.yaml)
- README.md, CLAUDE.md, DISCREPANCIES.md, requirements*.txt, .gitignore

Next steps:
1. On Kaggle: pip install -r requirements_locked.txt; python scripts/01_setup.py --download.
2. Hash analysis.py + config.py and record in this log before SA-2.
3. Run scripts/run_session1.sh then run_session2.sh (real GPU path).
4. python scripts/05_analysis.py; python scripts/fill_paper.py; compile the paper.
5. Monitor arXiv weekly for competing validation-loss-vs-success-rate work.
```

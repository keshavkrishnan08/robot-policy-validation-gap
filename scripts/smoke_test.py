#!/usr/bin/env python3
"""End-to-end verification on CPU. No real data, no GPU.

Part A (model path): trains a tiny Diffusion Policy on the synthetic reaching
task for 2 seeds x 2 checkpoints, then runs all 8 metrics and closed-loop
rollouts. Proves the torch code (model/train/metrics/rollouts) is error-free.

Part B (stats path): fabricates the full 6x3x6 results tables (clearly labelled,
NOT experimental data) and runs analysis.py + figures.py + power.py end to end.
Proves the statistical pipeline produces a valid results.json + figures.

Exit code 0 == every code path ran without error and produced finite, valid output.
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from torch.utils.data import DataLoader

from drc import analysis, config, devtools, figures, metrics as M, power
from drc.data.dataset import collate
from drc.rollouts import evaluate_checkpoint
from drc.train import load_policy, train_run
from drc.utils import ckpt_path, get_logger, path
from scripts.providers import build_replay_episodes, dataset_provider, env_for

log = get_logger("smoke")

SMOKE_TASK = "LIBERO-Spatial-1"
SMOKE_SEEDS = [0, 1]
SMOKE_EPOCHS = [1, 2]


def smoke_train_cfg():
    cfg = copy.deepcopy(config.load_train_cfg())
    cfg["policy"]["num_inference_steps"] = 4   # fast sampling for the smoke run
    cfg["policy"]["num_train_timesteps"] = 50
    cfg["metrics"] = {"m1_K": 2, "m3_K": 3, "m4_pca_components": 16, "m8_K": 3}
    cfg["ema"] = {"enabled": True, "decay": 0.9}
    return cfg


def part_a_model():
    log.info("=== Part A: model path (train -> metrics -> rollouts) ===")
    tcfg = smoke_train_cfg()

    for seed in SMOKE_SEEDS:
        train_run(
            task=SMOKE_TASK, seed=seed,
            dataset_provider=dataset_provider(SMOKE_TASK, synthetic=True),
            train_cfg=tcfg, backbone="smallcnn", device="cpu",
            epochs=max(SMOKE_EPOCHS), checkpoint_epochs=SMOKE_EPOCHS, val_K=2, log_every=1,
        )

    # metrics + rollouts on the produced checkpoints
    _, val_ds, info = dataset_provider(SMOKE_TASK, synthetic=True)()
    tl = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate)
    vl = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate)
    env = env_for(SMOKE_TASK, synthetic=True)
    replay = build_replay_episodes(SMOKE_TASK, True, val_ds, max_episodes=2)
    conds = [{"state": np.array([-0.5, -0.5, 0.0, 0.0], dtype=np.float32)},
             {"state": np.array([-0.7, -0.3, 0.0, 0.0], dtype=np.float32)}]

    for epoch in SMOKE_EPOCHS:
        pols = [load_policy(ckpt_path(SMOKE_TASK, s, epoch), tcfg, "cpu")[0] for s in SMOKE_SEEDS]
        m6 = M.compute_m6(pols, vl, "cpu", K=2)
        for s, pol in zip(SMOKE_SEEDS, pols):
            single = M.compute_single_checkpoint(pol, tl, vl, env, replay, "cpu", tcfg["metrics"])
            single["M6"] = m6
            for m in config.METRIC_COLS:
                v = single[m]
                assert np.isfinite(v), f"metric {m} not finite: {v}"
            res = evaluate_checkpoint(pol, env, conds, max_steps=40, device="cpu", K=2, noise_seed=42)
            assert len(res["successes"]) == len(conds)
            log.info(f"  ep{epoch} s{s}: M1={single['M1']:.3f} M4={single['M4']:.3f} "
                     f"M5={single['M5']:.3f} M6={single['M6']:.3f} sr={res['success_rate']:.2f}")
    log.info("Part A OK: all 8 metrics finite, rollouts ran.")


def part_b_stats():
    log.info("=== Part B: stats path (analysis + figures + power on FAKE data) ===")
    metrics_df, rollouts_df = devtools.make_fake_results(seed=0)
    mcsv = path("results", "SMOKE_metrics.csv")
    rcsv = path("results", "SMOKE_rollouts.csv")
    metrics_df.to_csv(mcsv, index=False)
    rollouts_df.to_csv(rcsv, index=False)
    assert len(metrics_df) == config.N_CHECKPOINTS, f"expected {config.N_CHECKPOINTS} rows"

    results = analysis.run_all(mcsv, rcsv)
    for h in ["H1", "H2", "H3", "H4"]:
        assert "supported" in results[h], f"{h} missing 'supported'"
        log.info(f"  {h}: supported={results[h]['supported']}")
    om = results["outcome_matrix"]
    log.info(f"  outcome row: {om['row']} -> {om['title']}")

    df = analysis.load_merged(mcsv, rcsv)
    figs = figures.generate_all(results, df, path("figures", "smoke"))
    for k, p in figs.items():
        assert os.path.exists(p), f"figure {k} not written"
    log.info(f"  figures: {', '.join(os.path.basename(p) for p in figs.values())}")

    pw = power.report(n_sims=500)
    log.info(f"  power(H1,delta=10)={pw['H1_power_delta10_sigma15']:.2f}  "
             f"power(H3)={pw['H3_power_margin0.08_sd0.10']:.2f}")
    log.info("Part B OK: analysis, figures, power all ran.")


if __name__ == "__main__":
    part_a_model()
    part_b_stats()
    log.info("SMOKE TEST PASSED ✅")

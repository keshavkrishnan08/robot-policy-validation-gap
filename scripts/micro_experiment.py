#!/usr/bin/env python3
"""Micro-experiment: a REAL end-to-end run on the SYNTHETIC toy reaching task.

PURPOSE: preliminary perspective only. This trains the actual Diffusion Policy,
saves a learning-curve of checkpoints, computes all 8 offline metrics and true
rollout success per checkpoint, then runs the real H1/H3/coherence analyses.

HONESTY: the task is a 2-D toy reacher, NOT LIBERO/Robomimic. Results here are a
sanity check on the machinery and weak intuition about whether the metrics
discriminate policy quality. They are NOT the paper's findings.
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from drc import analysis, metrics as M
from drc.data.dataset import collate
from drc.data.synthetic import make_synthetic_dataset, make_eval_conditions, expert_episode, _render
from drc.envs import make_env
from drc.rollouts import evaluate_checkpoint
from drc.train import load_policy, train_run
from drc.utils import ckpt_path, get_logger
from drc import config

log = get_logger("micro")

SEEDS = [0, 1, 2]
EPOCHS = [3, 8, 20, 45]          # a learning curve, incl. late (over-)training
N_DEMOS = 6                      # small -> room to overfit
N_ROLLOUTS = 10
TASK = "LIBERO-Spatial-1"        # name reused; synthetic backend ignores the real task


def micro_cfg():
    c = copy.deepcopy(config.load_train_cfg())
    c["policy"]["num_inference_steps"] = 4
    c["policy"]["num_train_timesteps"] = 50
    c["metrics"] = {"m1_K": 3, "m3_K": 4, "m4_pca_components": 12, "m8_K": 4}
    c["ema"] = {"enabled": True, "decay": 0.95}
    c["batch_size"] = 32
    return c


def provider():
    ds, info = make_synthetic_dataset(n_demos=N_DEMOS, length=30, seed=42)
    val, _ = make_synthetic_dataset(n_demos=3, length=30, seed=99)
    return ds, val, info


def build_replay(val_ds, n=4):
    eps = []
    for c in make_eval_conditions(n, seed=7):
        ep = expert_episode(c, length=24)
        states = ep["obs_states"]
        seq = []
        for t in range(0, len(states), 8):
            w = states[max(0, t - 1): t + 1]
            while len(w) < 2:
                w = np.concatenate([states[:1], w], 0)
            seq.append({"image": np.stack([_render(s) for s in w[-2:]])[None], "proprio": w[-2:][None]})
        eps.append({"obs_seq": seq, "initial_state": ep["initial_state"], "final_eef_pose": ep["final_eef_pose"]})
    return eps


def main():
    cfg = micro_cfg()
    log.info(f"training {len(SEEDS)} seeds x {max(EPOCHS)} epochs on synthetic toy ({N_DEMOS} demos)...")
    for s in SEEDS:
        train_run(TASK, s, provider, arch="diffusion", train_cfg=cfg, backbone="smallcnn",
                  device="cpu", epochs=max(EPOCHS), checkpoint_epochs=EPOCHS, val_K=3, log_every=15)

    _, val_ds, info = provider()
    tl = DataLoader(val_ds, batch_size=32, collate_fn=collate)
    vl = DataLoader(val_ds, batch_size=32, collate_fn=collate)
    env = make_env(TASK, config.load_tasks()[TASK], synthetic=True)
    replay = build_replay(val_ds)
    conds = make_eval_conditions(N_ROLLOUTS, seed=123)

    mrows, rrows = [], []
    for ep in EPOCHS:
        pols = [load_policy(ckpt_path(TASK, s, ep, "diffusion"), cfg, "cpu")[0] for s in SEEDS]
        m6 = M.compute_m6(pols, vl, "cpu", K=3)
        for s, pol in zip(SEEDS, pols):
            single = M.compute_single_checkpoint(pol, tl, vl, env, replay, "cpu", cfg["metrics"])
            single["M6"] = m6
            res = evaluate_checkpoint(pol, env, conds, max_steps=60, device="cpu", K=3, noise_seed=42)
            mrows.append({"task": TASK, "seed": s, "arch": "diffusion", "epoch": ep,
                          **{m: single[m] for m in config.METRIC_COLS}})
            rrows.append({"task": TASK, "seed": s, "arch": "diffusion", "epoch": ep,
                          "success_rate": res["success_rate"], "num_successes": res["num_successes"]})
            log.info(f"  ep{ep} s{s}: M1={single['M1']:.3f} M4={single['M4']:.2f} "
                     f"M5={single['M5']:.3f} sr={res['success_rate']:.2f}")

    mdf, rdf = pd.DataFrame(mrows), pd.DataFrame(rrows)
    mdf.to_csv("/tmp/micro_metrics.csv", index=False)
    rdf.to_csv("/tmp/micro_rollouts.csv", index=False)

    df = analysis.load_merged("/tmp/micro_metrics.csv", "/tmp/micro_rollouts.csv")
    print("\n===== LEARNING CURVE (success rate by epoch, mean over seeds) =====")
    print(rdf.groupby("epoch")["success_rate"].mean().round(3).to_string())

    print("\n===== H1: validation gap (per seed) =====")
    gaps = analysis.compute_gaps(df)
    print(gaps[["seed", "vlbest_success", "rollbest_success", "gap_pct"]].round(3).to_string(index=False))
    print(f"median gap: {gaps['gap_pct'].median():.1f} pp")

    print("\n===== H3: signed Spearman(metric, success), mean over seeds =====")
    st = analysis.spearman_table(df)
    for m in config.METRIC_COLS:
        print(f"  {m}: {st[m].mean():+.3f}")

    coh = analysis.coherence_hypothesis_analysis(analysis.h3_analysis(df))
    print("\n===== Coherence vs local =====")
    print(f"  local (M1,M2,M8) mean rho:     {coh['mean_spearman_local']:+.3f}")
    print(f"  coherence (M4,M5,M7) mean rho: {coh['mean_spearman_coherence']:+.3f}")
    print(f"  coherence - local: {coh['mean_diff_coherence_vs_local']:+.3f}")

    sel = analysis.causal_selection_analysis(df)
    print("\n===== Selection utility =====")
    print(f"  best selection metric: {sel['best_selection_metric']} "
          f"(expected success {sel['best_expected_success_pct']:.1f}%)")
    print(f"  validation-L1 selection: {sel['M1_expected_success_pct']:.1f}%  "
          f"gain: {sel['gain_over_M1_pct']:+.1f} pp")
    print("\n[REMINDER] synthetic toy task — sanity/intuition only, NOT paper findings.")


if __name__ == "__main__":
    main()

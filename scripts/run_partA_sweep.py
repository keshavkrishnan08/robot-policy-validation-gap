#!/usr/bin/env python3
"""Part A: controlled L-sweep validating the compounding bound and the
identifiability limit (CPU, no GPU). Produces the two headline figures.

For each gain L: train short policies, score every checkpoint with the 8 metrics
and rollout success, estimate L_hat, and record the gap and per-metric-class
ranking power. Then plot gap vs amplification (P1) and metric-class Spearman vs L (P2).

Tiny by default so it runs in minutes; widen L_VALUES/SEEDS/EPOCHS for the paper.
"""
import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from drc import analysis, config, figures, metrics as M
from drc.data.dataset import collate
from drc.dynamics_sweep import make_sweep_dataset, SweepEnv, sweep_eval_conditions
from drc.figures import ROLLOUT_FREE, ENV_QUERYING
from drc.lyapunov import amplification, estimate_lyapunov
from drc.rollouts import evaluate_checkpoint
from drc.train import load_policy, train_run
from drc.utils import ckpt_path, get_logger, path

log = get_logger("partA")


def sweep_cfg():
    c = copy.deepcopy(config.load_train_cfg())
    c["policy"]["num_inference_steps"] = 4
    c["policy"]["num_train_timesteps"] = 50
    c["metrics"] = {"m1_K": 2, "m3_K": 3, "m4_pca_components": 12, "m8_K": 3}
    c["ema"] = {"enabled": True, "decay": 0.9}
    c["batch_size"] = 32
    return c


def provider_for(L):
    def p():
        ds, info = make_sweep_dataset(L, n_demos=8, length=24, seed=int(L * 100))
        val, _ = make_sweep_dataset(L, n_demos=3, length=24, seed=int(L * 100) + 7)
        return ds, val, info
    return p


def build_replay(L, n=3):
    from drc.dynamics_sweep import _state_to_img
    eps = []
    for c in sweep_eval_conditions(n, seed=int(L * 100) + 3):
        s = np.asarray(c["state"], dtype=np.float32)
        states = [s.copy()]
        for _ in range(16):
            s = np.clip(L * s, -1, 1)
            states.append(s.copy())
        states = np.array(states)
        seq = []
        for t in range(0, len(states), 8):
            w = states[max(0, t - 1): t + 1]
            while len(w) < 2:
                w = np.concatenate([states[:1], w], 0)
            seq.append({"image": np.stack([_state_to_img(x) for x in w[-2:]])[None], "proprio": w[-2:][None]})
        eps.append({"obs_seq": seq, "initial_state": c["state"], "final_eef_pose": states[-1][:3]})
    return eps


def main(args):
    cfg = sweep_cfg()
    L_values = [float(x) for x in args.L.split(",")]
    seeds = list(range(args.seeds))
    epochs = [int(e) for e in args.epochs.split(",")]
    max_steps = args.max_steps

    mrows, rrows, lhat = [], [], {}
    for L in L_values:
        task = f"L={L:.2f}"
        for s in seeds:
            train_run(task, s, provider_for(L), arch="diffusion", train_cfg=cfg, backbone="smallcnn",
                      device="cpu", epochs=max(epochs), checkpoint_epochs=epochs, val_K=2, log_every=999)
        _, val_ds, info = provider_for(L)()
        tl = DataLoader(val_ds, batch_size=32, collate_fn=collate)
        vl = DataLoader(val_ds, batch_size=32, collate_fn=collate)
        env = SweepEnv(L, max_steps=max_steps)
        replay = build_replay(L)
        conds = sweep_eval_conditions(args.rollouts, seed=int(L * 100) + 99)
        for ep in epochs:
            pols = [load_policy(ckpt_path(task, s, ep, "diffusion"), cfg, "cpu")[0] for s in seeds]
            m6 = M.compute_m6(pols, vl, "cpu", K=2)
            for s, pol in zip(seeds, pols):
                single = M.compute_single_checkpoint(pol, tl, vl, env, replay, "cpu", cfg["metrics"])
                single["M6"] = m6
                res = evaluate_checkpoint(pol, env, conds, max_steps=max_steps, device="cpu", K=1, noise_seed=42)
                mrows.append({"task": task, "seed": s, "arch": "diffusion", "epoch": ep,
                              **{m: single[m] for m in config.METRIC_COLS}})
                rrows.append({"task": task, "seed": s, "arch": "diffusion", "epoch": ep,
                              "success_rate": res["success_rate"], "num_successes": res["num_successes"]})
        # estimate L_hat from the last checkpoint of seed 0
        pol0 = load_policy(ckpt_path(task, 0, epochs[-1], "diffusion"), cfg, "cpu")[0]
        lhat[task] = estimate_lyapunov(pol0, SweepEnv(L, max_steps=max_steps), conds[:4], max_steps,
                                       eps=0.05, K=1)["L_hat"]
        log.info(f"L={L:.2f}: L_hat={lhat[task]:.2f}")

    mdf, rdf = pd.DataFrame(mrows), pd.DataFrame(rrows)
    mp, rp = "/tmp/sweep_m.csv", "/tmp/sweep_r.csv"
    mdf.to_csv(mp, index=False); rdf.to_csv(rp, index=False)
    df = analysis.load_merged(mp, rp)
    gaps = analysis.compute_gaps(df)
    st = analysis.spearman_table(df)

    summary = []
    for L in L_values:
        task = f"L={L:.2f}"
        g = gaps[gaps["task"] == task]["gap_pct"].mean()
        sub = st[st["task"] == task]
        rf = float(sub[ROLLOUT_FREE].mean().mean())
        eq = float(sub[ENV_QUERYING].mean().mean())
        # Effective compounding horizon for the amplification axis: cap so L>1 does not
        # produce astronomical values that make the plot unreadable (the gap saturates anyway).
        H_eff = min(max_steps, args.ampl_horizon)
        summary.append({"L": L, "L_hat": lhat[task], "gap_pct": float(g),
                        "ampl": amplification(L, H_eff), "rollout_free_rho": rf, "env_query_rho": eq})
    sdf = pd.DataFrame(summary)
    out = path("results", "sweep_summary.csv"); sdf.to_csv(out, index=False)
    print("\n===== Part A sweep summary (synthetic, controlled L) =====")
    print(sdf.round(3).to_string(index=False))

    figdir = path("figures", "partA")
    f5 = figures.fig5_amplification(summary, figdir)
    f6 = figures.fig6_metric_class(summary, figdir)
    log.info(f"figures: {os.path.basename(f5)}, {os.path.basename(f6)}")
    print("\n[interpretation] P1: gap should rise with amplification; "
          "P2: rollout_free_rho should fall toward 0 as L crosses 1 while env_query_rho stays higher.")
    print("[REMINDER] controlled synthetic system — validates the theory's form; not a benchmark result.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # L band focused near the L=1 transition so amplification stays plottable.
    ap.add_argument("--L", default="0.7,0.85,0.95,1.0,1.05,1.15")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", default="5,15,40,80")
    ap.add_argument("--rollouts", type=int, default=12)
    ap.add_argument("--max_steps", type=int, default=40)
    ap.add_argument("--ampl_horizon", type=int, default=12, help="effective H for the amplification axis")
    main(ap.parse_args())

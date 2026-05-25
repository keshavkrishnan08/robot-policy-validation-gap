#!/usr/bin/env python3
"""SA-4 driver: 20 closed-loop rollouts per checkpoint on pre-fixed conditions.

Writes per-checkpoint JSON and the aggregate results/rollouts.csv. Verifies the
initial-condition state hashes are identical across checkpoints of a task.
"""
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from drc import config
from drc.rollouts import evaluate_checkpoint
from drc.train import load_policy
from drc.utils import ckpt_path, get_logger, path, rollouts_path, save_json
from scripts.providers import env_for

log = get_logger("04_rollouts")


def load_conditions(task):
    p = path("eval_conditions", f"{task}.pkl")
    with open(p, "rb") as f:
        return pickle.load(f)


def main(args):
    tcfg = config.load_train_cfg()
    K = tcfg.get("metrics", {}).get("m1_K", 10)
    noise_seed = tcfg.get("rollout", {}).get("ddim_noise_seed", 42)
    tasks_cfg = config.load_tasks()
    epochs = [int(e) for e in args.checkpoint_epochs.split(",")] if args.checkpoint_epochs else list(config.CHECKPOINT_EPOCHS)

    archs = config.ARCHITECTURES if args.all_archs else (args.arch,)
    selected = [t.strip() for t in args.tasks.split(",")] if args.tasks else list(config.TASKS)
    rows = []
    for task in selected:
        conds = load_conditions(task)
        max_steps = tasks_cfg[task]["max_steps"]
        ref_hashes = None
        env = env_for(task, synthetic=args.synthetic)
        for arch in archs:
            for seed in config.SEEDS:
                for epoch in epochs:
                    cp = ckpt_path(task, seed, epoch, arch)
                    if not os.path.exists(cp):
                        continue
                    pol, _ = load_policy(cp, tcfg, device=args.device)
                    res = evaluate_checkpoint(pol, env, conds, max_steps, device=args.device,
                                              K=K, noise_seed=noise_seed)
                    # SA-4 check: identical paired initial conditions across checkpoints.
                    if ref_hashes is None:
                        ref_hashes = res["state_hashes"]
                    elif res["state_hashes"] != ref_hashes:
                        log.warning(f"{task}: init-condition hash drift at s{seed} [{arch}] ep{epoch}")
                    save_json(res, rollouts_path(task, seed, epoch, arch))
                    rows.append({
                        "task": task, "seed": seed, "arch": arch, "epoch": epoch,
                        "success_rate": res["success_rate"], "num_successes": res["num_successes"],
                    })
                    log.info(f"{task} s{seed} [{arch}] ep{epoch}: success_rate={res['success_rate']:.2f}")

    df = pd.DataFrame(rows)
    out = path("results", "rollouts.csv")
    if os.path.exists(out) and len(df):
        prev = pd.read_csv(out)
        prev = prev[~prev["task"].isin(df["task"].unique())]
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(out, index=False)
    log.info(f"rollouts.csv now has {len(df)} rows ({df['task'].nunique() if len(df) else 0} tasks)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--arch", choices=config.ARCHITECTURES, default="diffusion")
    ap.add_argument("--all_archs", action="store_true")
    ap.add_argument("--tasks", default=None, help="comma list to process a subset (default all)")
    ap.add_argument("--checkpoint_epochs", default=None)
    main(ap.parse_args())

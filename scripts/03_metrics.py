#!/usr/bin/env python3
"""SA-3 driver: compute the eight offline metrics for every checkpoint.

M6 (inter-seed disagreement) is computed once per (task, epoch) from the seed
ensemble and broadcast to each seed's row. All others are per checkpoint.
Outputs per-checkpoint JSON and the aggregate results/metrics.csv.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from drc import config, metrics as M
from drc.data.dataset import collate
from drc.train import load_policy
from drc.utils import ckpt_path, get_logger, metrics_path, path, save_json
from scripts.providers import build_replay_episodes, dataset_provider, env_for

log = get_logger("03_metrics")


def loaders_for(task, synthetic, bs):
    provider = dataset_provider(task, synthetic=synthetic)
    train_ds, val_ds, info = provider()
    tl = DataLoader(train_ds, batch_size=bs, shuffle=False, collate_fn=collate)
    vl = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate)
    return train_ds, val_ds, info, tl, vl


def main(args):
    tcfg = config.load_train_cfg()
    mcfg = tcfg.get("metrics", {})
    rows = []
    epochs = [int(e) for e in args.checkpoint_epochs.split(",")] if args.checkpoint_epochs else list(config.CHECKPOINT_EPOCHS)

    for task in config.TASKS:
        _, val_ds, info, tl, vl = loaders_for(task, args.synthetic, args.bs)
        env = env_for(task, synthetic=args.synthetic)
        replay = build_replay_episodes(task, args.synthetic, val_ds) if args.synthetic else []

        for epoch in epochs:
            # M6 from the seed ensemble for this (task, epoch).
            seed_policies = {}
            for seed in config.SEEDS:
                cp = ckpt_path(task, seed, epoch)
                if os.path.exists(cp):
                    pol, _ = load_policy(cp, tcfg, device=args.device)
                    seed_policies[seed] = pol
            if not seed_policies:
                log.warning(f"no checkpoints for {task} epoch {epoch}; skipping")
                continue
            m6 = M.compute_m6(list(seed_policies.values()), vl, args.device, K=mcfg.get("m1_K", 10))

            for seed, pol in seed_policies.items():
                single = M.compute_single_checkpoint(pol, tl, vl, env, replay, args.device, mcfg)
                single["M6"] = m6
                vals = {m: single[m] for m in config.METRIC_COLS}
                # finiteness guard (SA-3 validation criterion)
                for m, v in vals.items():
                    if not np.isfinite(v):
                        raise ValueError(f"non-finite metric {m} for {task} s{seed} ep{epoch}")
                rec = {"task": task, "seed": seed, "epoch": epoch, **vals}
                save_json(rec, metrics_path(task, seed, epoch))
                rows.append(rec)
                log.info(f"{task} s{seed} ep{epoch}: M1={vals['M1']:.4f} M5={vals['M5']:.4f} M6={vals['M6']:.4f}")

    df = pd.DataFrame(rows)[["task", "seed", "epoch", *config.METRIC_COLS]]
    out = path("results", "metrics.csv")
    df.to_csv(out, index=False)
    log.info(f"wrote {len(df)} metric rows -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--checkpoint_epochs", default=None)
    main(ap.parse_args())

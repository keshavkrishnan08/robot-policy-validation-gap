#!/usr/bin/env python3
"""Sanity-check a Kaggle run's outputs before spending the next session.

Run after Session 1 (or any session) to confirm results/metrics.csv + results/rollouts.csv are
clean and complete: no NaN/Inf, success rates in range, M6 shared per (task,arch,epoch), and which
(task, arch) cells are present vs the locked grid. Exits 1 on any hard problem.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from drc import config
from drc.utils import get_logger, path

log = get_logger("check")


def main():
    mp, rp = path("results", "metrics.csv"), path("results", "rollouts.csv")
    problems, warnings = [], []

    if not os.path.exists(mp) or not os.path.exists(rp):
        log.error("missing results/metrics.csv or results/rollouts.csv"); sys.exit(1)
    m, r = pd.read_csv(mp), pd.read_csv(rp)

    # columns
    need_m = {"task", "seed", "arch", "epoch", *config.METRIC_COLS}
    need_r = {"task", "seed", "arch", "epoch", "success_rate"}
    if not need_m.issubset(m.columns):
        problems.append(f"metrics.csv missing columns {need_m - set(m.columns)}")
    if not need_r.issubset(r.columns):
        problems.append(f"rollouts.csv missing columns {need_r - set(r.columns)}")

    # finiteness + ranges
    if need_m.issubset(m.columns):
        bad = m[list(config.METRIC_COLS)].apply(lambda c: ~np.isfinite(c)).to_numpy().sum()
        if bad:
            problems.append(f"{bad} non-finite metric values in metrics.csv")
    if "success_rate" in r.columns:
        if not r["success_rate"].between(0, 1).all():
            problems.append("success_rate outside [0,1]")
        # all-0 or all-1 tasks (saturation) -> warn (gap mechanically small)
        for task, g in r.groupby("task"):
            mu = g["success_rate"].mean()
            if mu <= 0.02 or mu >= 0.98:
                warnings.append(f"{task}: success rate saturated (mean {mu:.2f}) -> small gap expected")

    # M6 shared within (task, arch, epoch)
    if {"M6", "task", "arch", "epoch"}.issubset(m.columns):
        for key, g in m.groupby(["task", "arch", "epoch"]):
            if g["M6"].nunique() > 1:
                warnings.append(f"M6 not constant within {key} (should be the seed-ensemble value)")
                break

    # coverage vs the locked grid
    present = set(map(tuple, m[["task", "arch"]].drop_duplicates().to_numpy())) if "arch" in m.columns else set()
    expected = {(t, a) for t in config.TASKS for a in config.ARCHITECTURES}
    missing = expected - present
    if missing:
        warnings.append(f"{len(missing)}/{len(expected)} (task,arch) cells not yet present: "
                        f"{sorted(missing)[:6]}{'...' if len(missing) > 6 else ''}")

    # report
    print("\n===== results check =====")
    print(f"metrics rows: {len(m)} | rollout rows: {len(r)}")
    if "arch" in m.columns:
        print("per-task mean success (by arch):")
        merged = m.merge(r[["task", "seed", "arch", "epoch", "success_rate"]],
                         on=["task", "seed", "arch", "epoch"], how="inner")
        print(merged.groupby(["task", "arch"])["success_rate"].mean().round(2).to_string())
    for w in warnings:
        log.warning(w)
    if problems:
        for p in problems:
            log.error(p)
        log.error("RESULTS CHECK FAILED"); sys.exit(1)
    log.info("results check passed (warnings above are informational)")


if __name__ == "__main__":
    main()

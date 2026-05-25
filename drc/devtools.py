"""Verification-only helpers. NOT part of the experiment.

`make_fake_results` synthesises metrics.csv / rollouts.csv with the full
6x3x6 structure so analysis.py, figures.py and power.py can be exercised end to
end on the dev box. The numbers are random draws from a plausible generative
model; they are NOT experimental results and must never appear in the paper.
The real Kaggle run overwrites results/ with genuine data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from drc import config


def make_fake_results(seed: int = 0):
    """Return (metrics_df, rollouts_df) with a built-in val-gap so H1 fires.

    Generative story (purely for code verification):
      - success rate rises with epoch then plateaus, with task difficulty;
      - long/high-complexity tasks peak later -> a regime-dependent gap;
      - M1 (val L1) bottoms out earlier than success peaks -> a gap exists;
      - M5/M4 track success a bit better than M1 in some runs.
    """
    rng = np.random.default_rng(seed)
    labels = config.task_regime_labels()
    epochs = np.array(config.CHECKPOINT_EPOCHS, dtype=float)
    metric_rows, rollout_rows = [], []

    horizon_peak = {"short": 60, "medium": 95, "long": 140}
    # Default ceiling by horizon; long/high tasks cap lower.
    ceil_by_horizon = {"short": 0.92, "medium": 0.75, "long": 0.50}

    for task in config.TASKS:
        hz = labels[task]["horizon"]
        peak = horizon_peak[hz]
        base_ceil = ceil_by_horizon[hz]
        for arch in config.ARCHITECTURES:
            # ACT peaks a little earlier and slightly higher in this toy model.
            arch_shift = -10 if arch == "act" else 0
            arch_ceil = base_ceil + (0.03 if arch == "act" else 0.0)
            for s in config.SEEDS:
                sr = arch_ceil * np.exp(-((epochs - (peak + arch_shift)) ** 2) / (2 * 45.0 ** 2))
                sr = np.clip(sr + rng.normal(0, 0.04, size=len(epochs)), 0, 1)
                m1 = np.clip(0.30 * np.exp(-epochs / 60.0) + rng.normal(0, 0.005, len(epochs)), 1e-3, None)
                m2 = m1 * (1.0 + rng.normal(0, 0.1, len(epochs)))
                m3 = 2.0 + 0.5 * sr + rng.normal(0, 0.1, len(epochs))
                m4 = np.clip(5.0 - 3.0 * sr + rng.normal(0, 0.3, len(epochs)), 0.1, None)
                m5 = np.clip(1.0 - 0.8 * sr + rng.normal(0, 0.05, len(epochs)), 0.01, None)
                m7 = np.clip(3.0 - 1.0 * sr + rng.normal(0, 0.2, len(epochs)), 0.01, None)
                m8 = -(0.5 - 0.3 * sr + rng.normal(0, 0.05, len(epochs)))
                for i, ep in enumerate(config.CHECKPOINT_EPOCHS):
                    metric_rows.append({
                        "task": task, "seed": s, "arch": arch, "epoch": ep,
                        "M1": m1[i], "M2": m2[i], "M3": m3[i], "M4": m4[i],
                        "M5": m5[i], "M6": np.nan, "M7": m7[i], "M8": m8[i],
                    })
                    rollout_rows.append({
                        "task": task, "seed": s, "arch": arch, "epoch": ep,
                        "success_rate": float(sr[i]), "num_successes": int(round(sr[i] * 20)),
                    })

    metrics_df = pd.DataFrame(metric_rows)
    # M6 (inter-seed disagreement) is shared across seeds within (task, arch, epoch).
    for _, g in metrics_df.groupby(["task", "arch", "epoch"]):
        metrics_df.loc[g.index, "M6"] = float(np.abs(rng.normal(0.05, 0.02)))
    return metrics_df, pd.DataFrame(rollout_rows)

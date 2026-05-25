"""SA-5: the four paper figures, generated from the analysis results + merged df.

  Figure 1: val-loss-selected vs rollout-best success (the headline scatter)
  Figure 2: per-metric mean Spearman heatmap (by task)
  Figure 3: regime breakdown of the gap with bootstrap CIs
  Figure 4: composite predictor calibration (predicted vs actual, held-out)
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from drc import analysis, config


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig1_headline(results, out_dir):
    gaps = pd.DataFrame(results["H1"]["gaps"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(gaps["vlbest_success"] * 100, gaps["rollbest_success"] * 100, c="#2b6cb0", s=40, zorder=3)
    lim = [0, 100]
    ax.plot(lim, lim, "--", color="gray", label="y = x (no gap)")
    ax.set_xlabel("Val-loss-selected success rate (%)")
    ax.set_ylabel("Rollout-best success rate (%)")
    ax.set_title(f"The Validation Gap (median {results['H1']['median_gap_pct']:.1f} pp)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.legend()
    return _save(fig, out_dir, "fig1_headline_scatter.pdf")


def fig2_heatmap(df, out_dir):
    st = analysis.spearman_table(df)
    tasks = list(config.TASKS)
    mat = np.array([[st[st["task"] == t][m].mean() for m in config.METRIC_COLS] for t in tasks])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(config.METRIC_COLS)), config.METRIC_COLS)
    ax.set_yticks(range(len(tasks)), tasks)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Signed Spearman(metric, success) by task")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, out_dir, "fig2_correlation_heatmap.pdf")


def fig3_regime(results, out_dir):
    rm = results["H2"]["regime_means"]
    keys = [k for k in rm if k.startswith("horizon=")] + [k for k in rm if k.startswith("complexity=")]
    means = [rm[k]["mean_gap_pct"] for k in keys]
    los = [rm[k]["mean_gap_pct"] - rm[k]["bca_ci_pct"][0] for k in keys]
    his = [rm[k]["bca_ci_pct"][1] - rm[k]["mean_gap_pct"] for k in keys]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(keys)), means, yerr=[los, his], capsize=4, color="#2f855a")
    ax.set_xticks(range(len(keys)), keys, rotation=30, ha="right")
    ax.set_ylabel("Mean gap (pp)")
    ax.set_title("Validation gap by task regime")
    return _save(fig, out_dir, "fig3_regime_breakdown.pdf")


def fig4_calibration(df, out_dir):
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    tr = df[df["task"].isin(config.H4_TRAIN_TASKS)]
    te = df[df["task"].isin(config.H4_HELD_OUT_TASKS)]
    scaler = StandardScaler().fit(tr[list(config.METRIC_COLS)].values)
    folds = max(min(config.RIDGE_CV_FOLDS, len(tr)), 2)
    ridge = RidgeCV(alphas=config.RIDGE_ALPHAS, cv=folds).fit(
        scaler.transform(tr[list(config.METRIC_COLS)].values), tr["success_rate"].values
    )
    pred = ridge.predict(scaler.transform(te[list(config.METRIC_COLS)].values))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(te["success_rate"] * 100, pred * 100, c="#9b2c2c", s=40, zorder=3)
    ax.plot([0, 100], [0, 100], "--", color="gray")
    ax.set_xlabel("Actual success rate (%)")
    ax.set_ylabel("Composite-predicted success rate (%)")
    ax.set_title("Held-out composite predictor calibration")
    return _save(fig, out_dir, "fig4_composite_calibration.pdf")


def generate_all(results, df, out_dir):
    return {
        "fig1": fig1_headline(results, out_dir),
        "fig2": fig2_heatmap(df, out_dir),
        "fig3": fig3_regime(results, out_dir),
        "fig4": fig4_calibration(df, out_dir),
    }

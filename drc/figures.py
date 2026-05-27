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


# Metric classes by environment access (theory partition).
ROLLOUT_FREE = ["M1", "M2", "M3", "M4", "M6", "M8"]
ENV_QUERYING = ["M5", "M7"]


def fig5_amplification(summary, out_dir, name="fig5_gap_vs_amplification.pdf"):
    """Headline P1: validation gap vs the bound's amplification factor (L^H-1)/(L-1)."""
    import pandas as pd

    s = pd.DataFrame(summary).sort_values("ampl")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(s["ampl"], s["gap_pct"], c="#2b6cb0", s=50, zorder=3)
    for _, r in s.iterrows():
        ax.annotate(f"L={r['L']:.2f}", (r["ampl"], r["gap_pct"]), fontsize=7,
                    textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel(r"amplification factor $(L^H-1)/(L-1)$")
    ax.set_ylabel("validation gap (pp)")
    ax.set_title("P1: the gap tracks the predicted amplification")
    return _save(fig, out_dir, name)


def fig6_metric_class(summary, out_dir, name="fig6_metric_class_vs_L.pdf"):
    """Headline P2: rollout-free metrics lose ranking power as L grows past 1;
    environment-querying metrics keep it."""
    import pandas as pd

    s = pd.DataFrame(summary).sort_values("L")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(s["L"], s["rollout_free_rho"], "o-", color="#9b2c2c", label="rollout-free (M1-M4,M6,M8)")
    ax.plot(s["L"], s["env_query_rho"], "s-", color="#2f855a", label="environment-querying (M5,M7)")
    ax.axvline(1.0, ls="--", color="gray", lw=1)
    ax.axhline(0.0, ls=":", color="gray", lw=0.8)
    ax.set_xlabel(r"closed-loop gain $L$")
    ax.set_ylabel("mean signed Spearman with success")
    ax.set_title("P2: the identifiability limit at $L>1$")
    ax.legend(fontsize=8)
    return _save(fig, out_dir, name)


def fig_gap_vs_amplification_real(results, lhat, out_dir, H_eff=20, name="fig5_gap_vs_amplification_real.pdf"):
    """Real-task headline (P1): per-task validation gap vs the predicted amplification
    (L_hat^H-1)/(L_hat-1), using the empirically estimated closed-loop gain per task."""
    from drc.lyapunov import amplification
    import pandas as pd

    gaps = pd.DataFrame(results["H1"]["gaps"])
    rows = []
    for task, g in gaps.groupby("task"):
        if task not in lhat or not np.isfinite(lhat[task]):
            continue
        rows.append({"task": task, "ampl": amplification(lhat[task], H_eff),
                     "gap_pct": float(g["gap_pct"].mean()), "L_hat": lhat[task]})
    if len(rows) < 2:
        return None
    s = pd.DataFrame(rows).sort_values("ampl")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(s["ampl"], s["gap_pct"], c="#2b6cb0", s=45, zorder=3)
    for _, r in s.iterrows():
        ax.annotate(r["task"].replace("Robomimic-", "RM-").replace("LIBERO-", "LB-"),
                    (r["ampl"], r["gap_pct"]), fontsize=6, textcoords="offset points", xytext=(3, 3))
    ax.set_xlabel(rf"estimated amplification $(\hat{{L}}^{{{H_eff}}}-1)/(\hat{{L}}-1)$")
    ax.set_ylabel("validation gap (pp)")
    ax.set_title("Real tasks: gap vs estimated amplification")
    return _save(fig, out_dir, name)


def fig_metricclass_vs_lhat(df, lhat, out_dir, name="fig6_metricclass_vs_lhat.pdf"):
    """Real-task headline (P2): per-task rollout-free vs environment-querying mean signed
    Spearman with success, against the estimated closed-loop gain."""
    from drc import analysis
    import pandas as pd

    st = analysis.spearman_table(df)
    rows = []
    for task, sub in st.groupby("task"):
        if task not in lhat or not np.isfinite(lhat[task]):
            continue
        rows.append({"L_hat": lhat[task],
                     "rollout_free": float(sub[ROLLOUT_FREE].mean().mean()),
                     "env_query": float(sub[ENV_QUERYING].mean().mean())})
    if len(rows) < 2:
        return None
    s = pd.DataFrame(rows).sort_values("L_hat")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(s["L_hat"], s["rollout_free"], "o-", color="#9b2c2c", label="rollout-free (M1-M4,M6,M8)")
    ax.plot(s["L_hat"], s["env_query"], "s-", color="#2f855a", label="environment-querying (M5,M7)")
    ax.axvline(1.0, ls="--", color="gray", lw=1)
    ax.axhline(0.0, ls=":", color="gray", lw=0.8)
    ax.set_xlabel(r"estimated closed-loop gain $\hat{L}$")
    ax.set_ylabel("mean signed Spearman with success")
    ax.set_title("Real tasks: metric-class ranking power vs gain")
    ax.legend(fontsize=8)
    return _save(fig, out_dir, name)


def generate_all(results, df, out_dir):
    return {
        "fig1": fig1_headline(results, out_dir),
        "fig2": fig2_heatmap(df, out_dir),
        "fig3": fig3_regime(results, out_dir),
        "fig4": fig4_calibration(df, out_dir),
    }

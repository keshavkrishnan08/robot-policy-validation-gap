"""SA-5: PRE-LOCKED statistical analysis for H1-H4 (PRD section 10).

This file is committed and SHA-256-hashed BEFORE any training (SA-2) begins. It
must run unchanged on the real data. Every threshold and alpha is imported from
drc.config, which mirrors configs/stats.yaml.

Public entry point: run_all(metrics_csv, rollouts_csv) -> results dict (also
written to results/results.json by scripts/05_analysis.py).
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from drc import config

# Authoritative signing list (see configs/stats.yaml rationale).
LOWER_IS_BETTER = set(config.LOWER_IS_BETTER)
METRIC_COLS = list(config.METRIC_COLS)


# --------------------------------------------------------------------------- #
# Data loading                                                                #
# --------------------------------------------------------------------------- #
def load_merged(metrics_csv: str, rollouts_csv: str) -> pd.DataFrame:
    m = pd.read_csv(metrics_csv)
    r = pd.read_csv(rollouts_csv)
    df = m.merge(r[["task", "seed", "epoch", "success_rate"]], on=["task", "seed", "epoch"])
    return df.sort_values(["task", "seed", "epoch"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Bootstrap                                                                   #
# --------------------------------------------------------------------------- #
def bca_ci(values: np.ndarray, stat=np.median, n_resamples=None, ci=0.95):
    from scipy.stats import bootstrap

    values = np.asarray(values, dtype=float)
    n_resamples = n_resamples or config.BOOTSTRAP_RESAMPLES
    if len(np.unique(values)) < 2:
        v = float(stat(values))
        return (v, v)
    res = bootstrap(
        (values,), stat, n_resamples=n_resamples, method="BCa", confidence_level=ci, vectorized=False
    )
    return (float(res.confidence_interval.low), float(res.confidence_interval.high))


# --------------------------------------------------------------------------- #
# H1: the gap exists                                                          #
# --------------------------------------------------------------------------- #
def compute_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Per-run gap in percentage points: rollout-best minus val-L1-selected."""
    rows = []
    for (task, seed), g in df.groupby(["task", "seed"]):
        g = g.sort_values("epoch")
        vlbest_sr = g.loc[g["M1"].idxmin(), "success_rate"]
        rollbest_sr = g["success_rate"].max()
        rows.append(
            {
                "task": task,
                "seed": seed,
                "vlbest_success": vlbest_sr,
                "rollbest_success": rollbest_sr,
                "gap_pct": (rollbest_sr - vlbest_sr) * 100.0,
            }
        )
    return pd.DataFrame(rows)


def h1_analysis(df: pd.DataFrame) -> dict[str, Any]:
    gaps = compute_gaps(df)
    g = gaps["gap_pct"].values
    median = float(np.median(g))
    iqr = (float(np.percentile(g, 25)), float(np.percentile(g, 75)))
    # Wilcoxon needs nonzero differences; if all zero, p=1.
    if np.allclose(g, 0):
        stat, p = 0.0, 1.0
    else:
        stat, p = wilcoxon(g, alternative="greater", zero_method="wilcox")
    ci = bca_ci(g, np.median)
    supported = (median > config.H1_MEDIAN_GAP_THRESHOLD_PCT) and (p < config.BONFERRONI_ALPHA)
    return {
        "median_gap_pct": median,
        "iqr_pct": iqr,
        "wilcoxon_stat": float(stat),
        "p_value": float(p),
        "bca_ci_median_pct": ci,
        "n_runs": int(len(g)),
        "alpha": config.BONFERRONI_ALPHA,
        "threshold_pct": config.H1_MEDIAN_GAP_THRESHOLD_PCT,
        "supported": bool(supported),
        "gaps": gaps.to_dict(orient="records"),
    }


# --------------------------------------------------------------------------- #
# H2: the gap is regime-dependent                                             #
# --------------------------------------------------------------------------- #
def h2_analysis(df: pd.DataFrame) -> dict[str, Any]:
    import statsmodels.formula.api as smf

    gaps = compute_gaps(df)
    labels = config.task_regime_labels()
    gaps["horizon"] = gaps["task"].map(lambda t: labels[t]["horizon"])
    gaps["complexity"] = gaps["task"].map(lambda t: labels[t]["complexity"])
    # Reference levels: medium horizon / medium complexity.
    gaps["horizon"] = pd.Categorical(gaps["horizon"], categories=["medium", "short", "long"])
    gaps["complexity"] = pd.Categorical(gaps["complexity"], categories=["medium", "low", "high"])

    method = "mixedlm"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full = smf.mixedlm("gap_pct ~ C(horizon) + C(complexity)", gaps, groups=gaps["seed"]).fit(reml=False)
            null = smf.mixedlm("gap_pct ~ 1", gaps, groups=gaps["seed"]).fit(reml=False)
            llf_full, llf_null = full.llf, null.llf
            k_extra = len(full.fe_params) - len(null.fe_params)
            fe = {k: float(v) for k, v in full.fe_params.items()}
    except Exception:  # fall back to OLS LRT if mixedlm fails to converge
        method = "ols_fallback"
        full = smf.ols("gap_pct ~ C(horizon) + C(complexity)", gaps).fit()
        null = smf.ols("gap_pct ~ 1", gaps).fit()
        llf_full, llf_null = full.llf, null.llf
        k_extra = int(full.df_model - null.df_model)
        fe = {k: float(v) for k, v in full.params.items()}

    from scipy.stats import chi2

    lr_stat = float(2 * (llf_full - llf_null))
    k_extra = max(int(k_extra), 1)
    p = float(chi2.sf(lr_stat, df=k_extra))

    regime_means = {}
    for col in ["horizon", "complexity"]:
        for level, gg in gaps.groupby(col, observed=True):
            regime_means[f"{col}={level}"] = {
                "mean_gap_pct": float(gg["gap_pct"].mean()),
                "bca_ci_pct": bca_ci(gg["gap_pct"].values, np.mean),
                "n": int(len(gg)),
            }

    supported = p < config.BONFERRONI_ALPHA
    return {
        "method": method,
        "lr_stat": lr_stat,
        "df": k_extra,
        "p_value": p,
        "fixed_effects": fe,
        "regime_means": regime_means,
        "alpha": config.BONFERRONI_ALPHA,
        "supported": bool(supported),
    }


# --------------------------------------------------------------------------- #
# H3: an alternative metric beats validation L1                               #
# --------------------------------------------------------------------------- #
def _signed_spearman(metric_vals, success, metric: str) -> float:
    """Spearman(metric, success), sign-flipped so positive = predictive."""
    if len(np.unique(success)) < 2 or len(np.unique(metric_vals)) < 2:
        return 0.0
    rho, _ = spearmanr(metric_vals, success)
    if np.isnan(rho):
        return 0.0
    return -rho if metric in LOWER_IS_BETTER else rho


def spearman_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, seed), g in df.groupby(["task", "seed"]):
        g = g.sort_values("epoch")
        rec = {"task": task, "seed": seed}
        for m in METRIC_COLS:
            rec[m] = _signed_spearman(g[m].values, g["success_rate"].values, m)
        rows.append(rec)
    return pd.DataFrame(rows)


def h3_analysis(df: pd.DataFrame) -> dict[str, Any]:
    st = spearman_table(df)
    m1 = st["M1"].values
    per_metric = {}
    best_metric, best_p = None, 1.0
    for m in METRIC_COLS[1:]:  # M2..M8, 7 comparisons
        diff = st[m].values - m1
        if np.allclose(diff, 0):
            stat, p = 0.0, 1.0
        else:
            stat, p = wilcoxon(diff, alternative="greater", zero_method="wilcox")
        p_corr = min(float(p) * config.H3_N_COMPARISONS, 1.0)
        per_metric[m] = {
            "mean_spearman": float(st[m].mean()),
            "mean_diff_vs_M1": float(np.mean(diff)),
            "wilcoxon_stat": float(stat),
            "p_raw": float(p),
            "p_bonferroni": p_corr,
            "beats_M1": bool(p < config.H3_ALPHA),
        }
        if p < best_p:
            best_p, best_metric = p, m

    supported = any(v["beats_M1"] for v in per_metric.values())
    return {
        "mean_spearman_M1": float(st["M1"].mean()),
        "per_metric": per_metric,
        "best_metric": best_metric,
        "best_p_raw": float(best_p),
        "h3_alpha": config.H3_ALPHA,
        "supported": bool(supported),
        "spearman_table": st.to_dict(orient="records"),
    }


# --------------------------------------------------------------------------- #
# H4: composite predictor generalises to held-out tasks                       #
# --------------------------------------------------------------------------- #
def _fit_eval(df_train, df_test):
    Xtr = df_train[METRIC_COLS].values
    ytr = df_train["success_rate"].values
    Xte = df_test[METRIC_COLS].values
    yte = df_test["success_rate"].values

    scaler = StandardScaler().fit(Xtr)
    folds = min(config.RIDGE_CV_FOLDS, len(df_train))
    ridge = RidgeCV(alphas=config.RIDGE_ALPHAS, cv=max(folds, 2)).fit(scaler.transform(Xtr), ytr)
    comp_rmse = float(np.sqrt(mean_squared_error(yte, ridge.predict(scaler.transform(Xte)))))

    base = LinearRegression().fit(Xtr[:, 0:1], ytr)  # M1 only
    base_rmse = float(np.sqrt(mean_squared_error(yte, base.predict(Xte[:, 0:1]))))
    return comp_rmse, base_rmse, float(ridge.alpha_)


def h4_analysis(df: pd.DataFrame) -> dict[str, Any]:
    train_tasks = list(config.H4_TRAIN_TASKS)
    held = list(config.H4_HELD_OUT_TASKS)
    df_tr = df[df["task"].isin(train_tasks)]
    df_te = df[df["task"].isin(held)]
    comp_rmse, base_rmse, alpha = _fit_eval(df_tr, df_te)
    rel = 1.0 - (comp_rmse / base_rmse) if base_rmse > 0 else 0.0

    # Leave-one-task-out sensitivity across all six tasks.
    loto = {}
    for t in config.TASKS:
        c, b, _ = _fit_eval(df[df["task"] != t], df[df["task"] == t])
        loto[t] = {"composite_rmse": c, "baseline_rmse": b}
    loto_rel = float(np.mean([1 - loto[t]["composite_rmse"] / loto[t]["baseline_rmse"]
                              for t in loto if loto[t]["baseline_rmse"] > 0]))

    supported = rel >= config.H4_RMSE_IMPROVEMENT_THRESHOLD
    return {
        "composite_rmse": comp_rmse,
        "baseline_rmse": base_rmse,
        "relative_improvement": rel,
        "ridge_alpha": alpha,
        "threshold": config.H4_RMSE_IMPROVEMENT_THRESHOLD,
        "loto": loto,
        "loto_mean_relative_improvement": loto_rel,
        "supported": bool(supported),
    }


# --------------------------------------------------------------------------- #
# Secondary analysis A: causal selection analysis (Upgrade 2 — 2026-05-25)   #
# Pre-registered as exploratory; added before any training.                   #
# --------------------------------------------------------------------------- #

# Metric direction for causal selection: which end of each metric is "better"?
# Matches LOWER_IS_BETTER / HIGHER_IS_BETTER in config.
_METRIC_OPTIMISE = {m: ("min" if m in LOWER_IS_BETTER else "max") for m in METRIC_COLS}


def causal_selection_analysis(df: pd.DataFrame) -> dict[str, Any]:
    """For each metric, simulate checkpoint selection using that metric alone
    and compute the expected deployed success rate across runs.

    Answers: 'If I had used metric M to pick checkpoints, what deployed success
    rate would I have achieved?' — the practitioner-facing complement to H3.
    """
    results = {}
    for m in METRIC_COLS:
        selected_srs = []
        for (task, seed), g in df.groupby(["task", "seed"]):
            g = g.sort_values("epoch")
            if _METRIC_OPTIMISE[m] == "min":
                best_idx = g[m].idxmin()
            else:
                best_idx = g[m].idxmax()
            selected_srs.append(g.loc[best_idx, "success_rate"])
        expected = float(np.mean(selected_srs))
        worst = float(np.percentile(selected_srs, 10))  # CVaR-like worst-case
        results[m] = {
            "expected_success": expected,
            "expected_success_pct": expected * 100,
            "p10_success_pct": worst * 100,
            "mean_selected_sr": selected_srs,
        }
    # Rank metrics by expected success descending.
    ranking = sorted(results.keys(), key=lambda m: results[m]["expected_success"], reverse=True)
    return {
        "per_metric": {m: {k: v for k, v in results[m].items() if k != "mean_selected_sr"}
                       for m in METRIC_COLS},
        "ranking_by_expected_success": ranking,
        "best_selection_metric": ranking[0],
        "best_expected_success_pct": results[ranking[0]]["expected_success_pct"],
        "M1_expected_success_pct": results["M1"]["expected_success_pct"],
        "gain_over_M1_pct": results[ranking[0]]["expected_success_pct"]
                             - results["M1"]["expected_success_pct"],
    }


# --------------------------------------------------------------------------- #
# Secondary analysis B: coherence hypothesis (Upgrade 1 — 2026-05-25)        #
# --------------------------------------------------------------------------- #

# Coherence hypothesis: metrics that measure global trajectory coherence
# (M4 latent OOD, M5 replay, M7 jerk) should outperform local accuracy
# metrics (M1 L1, M2 delta-MSE, M8 confidence) at predicting deployment success.
COHERENCE_METRICS = ("M4", "M5", "M7")
LOCAL_METRICS = ("M1", "M2", "M8")
VARIABILITY_METRICS = ("M3", "M6")


def coherence_hypothesis_analysis(h3_result: dict) -> dict[str, Any]:
    """Test whether the coherence metric group has higher mean signed Spearman
    than the local accuracy group, using a one-sided Wilcoxon on per-run values.
    """
    st_records = h3_result["spearman_table"]
    if not st_records:
        return {"supported": False, "note": "no spearman table"}

    st = pd.DataFrame(st_records)
    local_srs, coherence_srs = [], []
    for _, row in st.iterrows():
        local_srs.append(float(np.mean([row[m] for m in LOCAL_METRICS])))
        coherence_srs.append(float(np.mean([row[m] for m in COHERENCE_METRICS])))

    diff = np.array(coherence_srs) - np.array(local_srs)
    if np.allclose(diff, 0):
        stat, p = 0.0, 1.0
    else:
        stat, p = wilcoxon(diff, alternative="greater", zero_method="wilcox")

    return {
        "mean_spearman_local": float(np.mean(local_srs)),
        "mean_spearman_coherence": float(np.mean(coherence_srs)),
        "mean_diff_coherence_vs_local": float(np.mean(diff)),
        "wilcoxon_stat": float(stat),
        "p_value": float(p),
        "supported": bool(p < 0.05),  # exploratory threshold, not Bonferroni
        "note": "exploratory test of the coherence hypothesis (not pre-registered)",
    }


# --------------------------------------------------------------------------- #
# Outcome matrix (PRD section 6)                                              #
# --------------------------------------------------------------------------- #
def classify_outcome(h1: bool, h2: bool, h3: bool, h4: bool) -> dict[str, str]:
    if not h1:
        return {
            "row": "H1=No",
            "title": "Validation Loss Is Adequate for Robot Policy Selection: A Calibration Study",
            "angle": "surprising-finding paper; calibrates the alarm",
        }
    table = {
        (True, True, True): {
            "title": "An Offline Predictor of Robot Policy Success: Calibrating Validation Loss Against Deployment",
            "angle": "strong constructive contribution",
        },
        (True, True, False): {
            "title": "Task-Specific Offline Metrics for Robot Policy Selection: Why One Predictor Does Not Suffice",
            "angle": "negative-on-generalisation result",
        },
        (True, False, False): {
            "title": "The Horizon Wall: When Validation Loss Stops Predicting Robot Policy Success",
            "angle": "failure-boundary paper; recommends rollout-based evaluation past a regime threshold",
        },
        (False, True, True): {
            "title": "Validation Loss Considered Harmful: A Universal Offline Predictor of Robot Policy Success",
            "angle": "biggest possible paper",
        },
        (False, False, False): {
            "title": "An Impossibility Result for Offline Selection of Robot Policies",
            "angle": "important negative result; motivates rollout standards",
        },
    }
    key = (h2, h3, h4)
    entry = table.get(key)
    if entry is None:
        entry = {
            "title": "The Validation Gap: A Calibration Study of Offline Metrics for Robot Policy Selection",
            "angle": f"mixed outcome (H2={h2}, H3={h3}, H4={h4}); reported as calibrated empirical result",
        }
    return {"row": f"H1=Yes,H2={h2},H3={h3},H4={h4}", **entry}


# --------------------------------------------------------------------------- #
# Top-level driver                                                            #
# --------------------------------------------------------------------------- #
def run_all(metrics_csv: str, rollouts_csv: str) -> dict[str, Any]:
    df = load_merged(metrics_csv, rollouts_csv)
    h1 = h1_analysis(df)
    h2 = h2_analysis(df)
    h3 = h3_analysis(df)
    h4 = h4_analysis(df)
    outcome = classify_outcome(h1["supported"], h2["supported"], h3["supported"], h4["supported"])
    secondary = {
        "causal_selection": causal_selection_analysis(df),
        "coherence_hypothesis": coherence_hypothesis_analysis(h3),
    }
    return {
        "n_observations": int(len(df)),
        "n_runs": int(df.groupby(["task", "seed"]).ngroups),
        "secondary": secondary,
        "H1": h1,
        "H2": h2,
        "H3": h3,
        "H4": h4,
        "outcome_matrix": outcome,
        "prelock": {
            "tasks": list(config.TASKS),
            "seeds": list(config.SEEDS),
            "checkpoint_epochs": list(config.CHECKPOINT_EPOCHS),
            "metrics": list(config.METRIC_COLS),
            "held_out_tasks": list(config.H4_HELD_OUT_TASKS),
            "bonferroni_alpha": config.BONFERRONI_ALPHA,
            "h3_alpha": config.H3_ALPHA,
        },
    }

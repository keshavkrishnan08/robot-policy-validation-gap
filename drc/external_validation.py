"""Tier 3: external validation of offline metrics against real-robot outcomes.

Goal: show the offline metrics predict the *real-world* success ranking of public
generalist policies (SIMPLER's comparison set) better than validation MSE, whose
real-success correlation SIMPLER pegs at Pearson r = 0.308.

Two halves:
  1. Architecture-agnostic metric computation over a generic PolicyAdapter — only
     needs predict(obs)->action and optional sample(obs)->action. Per-policy
     inference runs on Kaggle (heterogeneous frameworks); see scripts/06.
  2. correlate_with_real_success(): the analysis that compares each offline metric's
     rank correlation with real success against the validation-MSE baseline.
     This half is framework-free and unit-tested.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy.stats import pearsonr, spearmanr

# SIMPLER (Li et al. 2024, Table I): published correlation of validation MSE with
# real-world success — the baseline our offline metrics must beat.
SIMPLER_MSE_REAL_PEARSON = 0.308
SIMPLER_SIM_REAL_PEARSON = 0.924

# Architecture-agnostic subset (M4 descoped for cross-architecture reasons).
EXTERNAL_METRICS = ("M1", "M3", "M5", "M8")
EXTERNAL_LOWER_IS_BETTER = {"M1", "M5"}  # M3/M8 higher-is-better


class PolicyAdapter(Protocol):
    """Minimal interface a wrapped policy must expose for external metrics."""

    name: str

    def predict(self, obs: dict) -> np.ndarray:        # deterministic action (chunk or step)
        ...

    def sample(self, obs: dict) -> np.ndarray:         # one stochastic sample (for M3/M8)
        ...


def compute_external_metrics(adapter, demos, env=None, n_sample: int = 16):
    """Compute the architecture-agnostic offline metrics for one policy.

    demos: list of episodes, each {"obs_seq": [...], "actions": np.ndarray,
           "initial_state": ..., "final_eef_pose": ...}.
    env:   a SimplerEnv-style env for M5 open-loop replay (optional; M5 skipped if None).
    Returns {"M1":..., "M3":..., "M5":..., "M8":...} (NaN where not computable).
    """
    m1, m3, m8 = [], [], []
    for ep in demos:
        for obs, expert_a in zip(ep["obs_seq"], ep["actions"]):
            pred = np.asarray(adapter.predict(obs)).reshape(-1)
            ea = np.asarray(expert_a).reshape(-1)
            d = min(len(pred), len(ea))
            m1.append(np.abs(pred[:d] - ea[:d]).mean())
            try:
                samples = np.stack([np.asarray(adapter.sample(obs)).reshape(-1)[:d] for _ in range(n_sample)])
                var = samples.var(axis=0)
                m3.append(0.5 * np.log(2 * np.pi * np.e * (var + 1e-9)).sum())
                m8.append(-var.sum())
            except (NotImplementedError, AttributeError):
                pass

    out = {
        "M1": float(np.mean(m1)) if m1 else float("nan"),
        "M3": float(np.mean(m3)) if m3 else float("nan"),
        "M8": float(np.mean(m8)) if m8 else float("nan"),
    }
    out["M5"] = _open_loop_replay_distance(adapter, env, demos) if env is not None else float("nan")
    return out


def _open_loop_replay_distance(adapter, env, demos) -> float:
    dists = []
    for ep in demos:
        env.reset_to({"state": ep["initial_state"]})
        for obs in ep["obs_seq"]:
            a = np.asarray(adapter.predict(obs)).reshape(-1)
            env.step(a)
        dists.append(float(np.linalg.norm(env.eef_pose() - ep["final_eef_pose"])))
    return float(np.mean(dists)) if dists else float("nan")


def correlate_with_real_success(metric_by_policy: dict, real_success: dict) -> dict:
    """Rank each offline metric against real-world success across policies.

    metric_by_policy: {policy_name: {"M1":.., "M3":.., "M5":.., "M8":..}}
    real_success:     {policy_name: real_world_success_rate}
    Returns per-metric Spearman/Pearson (sign-oriented so positive = predictive),
    plus comparison to SIMPLER's published validation-MSE baseline.
    """
    policies = [p for p in real_success if p in metric_by_policy]
    y = np.array([real_success[p] for p in policies], dtype=float)
    results = {}
    for m in EXTERNAL_METRICS:
        x = np.array([metric_by_policy[p].get(m, np.nan) for p in policies], dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3 or len(np.unique(x[mask])) < 2 or len(np.unique(y[mask])) < 2:
            results[m] = {"n": int(mask.sum()), "spearman": None, "pearson": None}
            continue
        rho, _ = spearmanr(x[mask], y[mask])
        r, _ = pearsonr(x[mask], y[mask])
        sign = -1.0 if m in EXTERNAL_LOWER_IS_BETTER else 1.0
        results[m] = {
            "n": int(mask.sum()),
            "spearman": float(sign * rho),
            "pearson": float(sign * r),
            "beats_mse_baseline": bool(abs(r) > SIMPLER_MSE_REAL_PEARSON),
        }
    best = max(
        (m for m in results if results[m]["pearson"] is not None),
        key=lambda m: abs(results[m]["pearson"]),
        default=None,
    )
    return {
        "policies": policies,
        "n_policies": len(policies),
        "per_metric": results,
        "mse_baseline_pearson": SIMPLER_MSE_REAL_PEARSON,
        "sim_baseline_pearson": SIMPLER_SIM_REAL_PEARSON,
        "best_metric": best,
        "best_pearson": abs(results[best]["pearson"]) if best else None,
        "any_metric_beats_mse": any(
            results[m].get("beats_mse_baseline") for m in results if results[m]["pearson"] is not None
        ),
    }

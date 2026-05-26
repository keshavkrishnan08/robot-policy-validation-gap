"""Part A (redesigned): a faithful numerical validation of the compounding bound.

Instead of training image policies on a saturating task, we validate the theory on a
linear system that exactly meets its assumptions, measuring real simulated quantities.

System: reference s*_{t+1} = L s*_t (expert closed loop, Lip = L). A policy injects a
per-step action error e_t, so its rollout obeys s_{t+1} = L s_t + e_t and the deviation
delta_t = ||s_t - s*_t|| follows delta_{t+1} = L delta_t + ||e_t|| exactly.

P1 (bound tightness): with a persistent error of magnitude eps, delta_H equals
  eps * (L^H - 1)/(L - 1) -- the bound is achieved, so the H/L scaling is real.

P2 (identifiability): across policies with independently varying (eps_i, L_i), success
(delta_H < tol) is governed by eps_i * amplification(L_i, H). Validation loss sees only
eps_i and so cannot rank them; an environment-querying replay measures delta_H and can.
This is the mechanism behind the paper's identifiability limit, shown on a system that
satisfies the assumptions. It validates the theory, not real-robot behavior (that is Part B).
"""
from __future__ import annotations

import numpy as np

from drc.lyapunov import amplification


def deviation_persistent(L: float, eps: float, H: int, dim: int = 2) -> float:
    """Closed-loop deviation under a persistent (constant-direction) error of size eps.
    Realizes delta_{t+1}=L delta_t + eps with delta_0=0 -> eps*(L^H-1)/(L-1)."""
    d = 0.0
    for _ in range(H):
        d = L * d + eps
    return d  # equals eps * amplification(L, H)


def deviation_random(L: float, eps: float, H: int, n_trials: int, rng, dim: int = 2) -> float:
    """Expected deviation under independent zero-mean per-step errors of std eps."""
    out = []
    for _ in range(n_trials):
        s = np.zeros(dim)
        for _ in range(H):
            s = L * s + rng.normal(0, eps, dim)
        out.append(float(np.linalg.norm(s)))
    return float(np.mean(out))


def run_p1(L_values, eps_values, H, dim=2):
    """Bound tightness: measured deviation vs predicted eps*(L^H-1)/(L-1)."""
    rows = []
    for L in L_values:
        for eps in eps_values:
            measured = deviation_persistent(L, eps, H, dim)
            predicted = eps * amplification(L, H)
            rows.append({"L": L, "eps": eps, "H": H, "measured": measured, "predicted": predicted})
    return rows


def run_p2_identifiability(n_policies=400, H=12, tol=1.0, eps_range=(0.005, 0.08),
                           L_range=(0.6, 1.25), seed=0):
    """Sample policies with independent (eps, L); show validation loss (eps) cannot rank
    success while an environment-querying replay (delta_H) can."""
    rng = np.random.default_rng(seed)
    eps = rng.uniform(*eps_range, n_policies)
    L = rng.uniform(*L_range, n_policies)
    delta = np.array([deviation_persistent(L[i], eps[i], H) for i in range(n_policies)])
    success = (delta < tol).astype(float)
    # rollout-free metric = validation loss proxy (eps); env-querying metric = replay deviation.
    # sign: both are lower-is-better, so correlate the negative with success.
    def signed_corr(x):
        if len(np.unique(success)) < 2 or len(np.unique(x)) < 2:
            return 0.0
        return float(-np.corrcoef(x, success)[0, 1])
    val_loss_corr = signed_corr(eps)
    replay_corr = signed_corr(delta)
    # stratify by L band to show within-band eps works but pooled (unknown-L) it fails
    bands = {}
    for lo, hi, name in [(0.6, 0.85, "L<0.85"), (0.85, 1.0, "0.85<=L<1"), (1.0, 1.25, "L>=1")]:
        m = (L >= lo) & (L < hi)
        if m.sum() > 5 and len(np.unique(success[m])) > 1:
            bands[name] = {"val_loss_corr": signed_corr_sub(eps[m], success[m]),
                           "replay_corr": signed_corr_sub(delta[m], success[m]),
                           "n": int(m.sum()), "success_rate": float(success[m].mean())}
    return {
        "n_policies": n_policies, "H": H, "tol": tol,
        "val_loss_corr_pooled": val_loss_corr,
        "replay_corr_pooled": replay_corr,
        "success_rate": float(success.mean()),
        "by_L_band": bands,
    }


def signed_corr_sub(x, success):
    if len(np.unique(success)) < 2 or len(np.unique(x)) < 2:
        return 0.0
    return float(-np.corrcoef(x, success)[0, 1])

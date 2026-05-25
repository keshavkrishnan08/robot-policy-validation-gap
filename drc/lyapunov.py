"""Estimate the closed-loop expansiveness L_hat from rollouts.

The theory's amplification factor (L^H-1)/(L-1) needs an estimate of L, the
closed-loop gain, to place each task on the prediction's x-axis. We estimate it
as a finite-time Lyapunov rate: perturb the initial condition by eps, run the
nominal and perturbed rollouts under the same policy and noise seed, and fit the
per-step growth of their separation, log d_t ~ log d_0 + t log L_hat.

L_hat < 1 => self-correcting; ~1 => marginal; > 1 => expansive (contact-rich).
On real sims the perturbation is approximate (it nudges the recorded sim state);
we report L_hat as an empirical proxy, not an exact spectral radius.
"""
from __future__ import annotations

import numpy as np

from drc.rollouts import rollout_trajectory


def _perturb(cond: dict, eps: float, rng) -> dict:
    state = np.asarray(cond["state"], dtype=np.float64)
    return {"state": (state + rng.normal(0, eps, size=state.shape)).tolist()}


def estimate_lyapunov(policy, env, conds, max_steps, device="cpu", eps=0.05, K=1, noise_seed=42, seed=0):
    """Return {"L_hat": float, "per_condition": [...], "n": int}.

    L_hat is the geometric-mean per-step separation growth rate across conditions,
    estimated by least-squares slope of log-separation vs time (pre-saturation).
    """
    rng = np.random.default_rng(seed)
    rates = []
    for cond in conds:
        nom = rollout_trajectory(policy, env, cond, max_steps, device, K, noise_seed)
        per = rollout_trajectory(policy, env, _perturb(cond, eps, rng), max_steps, device, K, noise_seed)
        T = min(len(nom), len(per))
        if T < 4:
            continue
        d = np.linalg.norm(nom[:T] - per[:T], axis=-1)
        d = np.maximum(d, 1e-9)
        # Use the unsaturated prefix: up to where separation first exceeds 100x eps.
        cap = np.argmax(d > 100 * eps)
        end = T if cap == 0 else max(cap, 4)
        t = np.arange(end)
        logd = np.log(d[:end])
        if len(t) < 4 or np.allclose(logd, logd[0]):
            continue
        slope = np.polyfit(t, logd, 1)[0]   # log L_hat per step
        rates.append(float(np.exp(slope)))
    if not rates:
        return {"L_hat": float("nan"), "per_condition": [], "n": 0}
    # Geometric mean of per-condition rates.
    L_hat = float(np.exp(np.mean(np.log(np.maximum(rates, 1e-9)))))
    return {"L_hat": L_hat, "per_condition": rates, "n": len(rates)}


def amplification(L: float, H: int) -> float:
    """The bound's amplification factor (L^H - 1)/(L - 1), with the L->1 limit = H."""
    if abs(L - 1.0) < 1e-6:
        return float(H)
    return float((L ** H - 1.0) / (L - 1.0))

"""Power analysis by simulation (PRD section 11).

Run before data collection to confirm the protocol can detect the pre-specified
effect sizes. These are Monte-Carlo estimates, not closed-form.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon

from drc import config


def h1_power(delta_pp=10.0, sigma_pp=15.0, n=18, alpha=None, n_sims=10000, seed=0) -> float:
    """One-sided Wilcoxon power to detect a mean gap of `delta_pp` percentage points."""
    alpha = alpha if alpha is not None else config.BONFERRONI_ALPHA
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_sims):
        x = rng.normal(delta_pp, sigma_pp, size=n)
        if np.allclose(x, 0):
            continue
        try:
            _, p = wilcoxon(x, alternative="greater", zero_method="wilcox")
        except ValueError:
            continue
        hits += int(p < alpha)
    return hits / n_sims


def h3_power(margin=0.08, sd=0.10, n=18, alpha=None, n_sims=10000, seed=0) -> float:
    """Power of the paired Wilcoxon to detect a Spearman advantage of `margin`."""
    alpha = alpha if alpha is not None else config.H3_ALPHA
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_sims):
        d = rng.normal(margin, sd, size=n)
        if np.allclose(d, 0):
            continue
        try:
            _, p = wilcoxon(d, alternative="greater", zero_method="wilcox")
        except ValueError:
            continue
        hits += int(p < alpha)
    return hits / n_sims


def report(n_sims=10000) -> dict:
    return {
        "H1_power_delta10_sigma15": h1_power(10.0, 15.0, n_sims=n_sims),
        "H1_power_delta15_sigma15": h1_power(15.0, 15.0, n_sims=n_sims),
        "H3_power_margin0.08_sd0.10": h3_power(0.08, 0.10, n_sims=n_sims),
        "alpha_primary": config.BONFERRONI_ALPHA,
        "alpha_h3": config.H3_ALPHA,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(report(n_sims=5000), indent=2))

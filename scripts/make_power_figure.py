#!/usr/bin/env python3
"""Generate the power-curve figure (paper appendix asset). CPU, ~1-2 min.

Plots H1 detection power vs the true gap magnitude for n=48 (3 seeds) and n=32 (2 seeds),
at the Bonferroni-corrected alpha. Supports the methods claim 'power > 0.95 at a 10pp gap'.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drc import power
from drc.utils import ensure_dir, get_logger, path

log = get_logger("power_fig")


def main(n_sims=6000):
    deltas = [4, 6, 8, 10, 12, 15, 20]
    fig, ax = plt.subplots(figsize=(6, 4))
    for n, c, lab in [(48, "#2b6cb0", "n=48 (3 seeds)"), (32, "#9b2c2c", "n=32 (2 seeds)")]:
        ys = [power.h1_power(d, 15, n=n, n_sims=n_sims) for d in deltas]
        ax.plot(deltas, ys, "o-", color=c, label=lab)
        log.info(f"{lab}: " + ", ".join(f"{d}pp:{y:.2f}" for d, y in zip(deltas, ys)))
    ax.axhline(0.8, ls=":", color="gray", lw=0.8)
    ax.axvline(10, ls="--", color="gray", lw=0.8)
    ax.set_xlabel("true validation gap (percentage points)")
    ax.set_ylabel("H1 detection power")
    ax.set_title(r"Power vs effect size (one-sided Wilcoxon, $\alpha=0.0125$)")
    ax.set_ylim(0, 1.02); ax.legend()
    out = os.path.join(ensure_dir(path("figures", "partA")), "power_curve.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    log.info(f"wrote {out}")


if __name__ == "__main__":
    main()

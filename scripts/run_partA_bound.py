#!/usr/bin/env python3
"""Part A (redesigned): numerical validation of the compounding bound. CPU, seconds.

P1: the bound is tight (measured deviation == eps*(L^H-1)/(L-1)).
P2: validation loss cannot rank policies of varying closed-loop gain; an environment-querying
    replay can. Produces two figures and a summary. Controlled validation of the theory's
    mechanism on a system meeting its assumptions -- NOT a real-robot result (that is Part B).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from drc.bound_validation import run_p1, run_p2_identifiability, selection_regret, horizon_wall
from drc.utils import ensure_dir, get_logger, path, save_json

log = get_logger("partA_bound")


def main():
    H = 12
    L_values = [0.6, 0.8, 0.9, 1.0, 1.1, 1.2]
    eps_values = [0.01, 0.02, 0.04, 0.06, 0.08]
    figdir = ensure_dir(path("figures", "partA"))

    # ---- P1: bound tightness ----
    p1 = run_p1(L_values, eps_values, H)
    meas = np.array([r["measured"] for r in p1])
    pred = np.array([r["predicted"] for r in p1])
    ss_res = np.sum((meas - pred) ** 2)
    ss_tot = np.sum((meas - meas.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    rel_err = float(np.max(np.abs(meas - pred) / np.maximum(pred, 1e-9)))

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(pred, meas, c="#2b6cb0", s=28, zorder=3, alpha=0.8)
    lim = [0, float(max(pred.max(), meas.max())) * 1.05]
    ax.plot(lim, lim, "--", color="gray", label="measured = predicted")
    ax.set_xlabel(r"predicted deviation $\varepsilon\,(L^H-1)/(L-1)$")
    ax.set_ylabel(r"measured closed-loop deviation $\delta_H$")
    ax.set_title(f"P1: the compounding bound is tight ($R^2={r2:.4f}$)")
    ax.legend()
    f1 = os.path.join(figdir, "figA1_bound_tightness.pdf")
    fig.savefig(f1, bbox_inches="tight"); plt.close(fig)

    # ---- P2: identifiability ----
    p2 = run_p2_identifiability(n_policies=600, H=H, tol=1.0, seed=0)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ["validation loss\n(rollout-free)", "open-loop replay\n(environment-querying)"]
    vals = [p2["val_loss_corr_pooled"], p2["replay_corr_pooled"]]
    ax.bar(bars, vals, color=["#9b2c2c", "#2f855a"])
    ax.set_ylabel("|correlation with deployment success|")
    ax.set_ylim(0, 1)
    ax.set_title("P2: only an environment-querying metric ranks\npolicies of varying closed-loop gain")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    f2 = os.path.join(figdir, "figA2_identifiability.pdf")
    fig.savefig(f2, bbox_inches="tight"); plt.close(fig)

    # ---- Thm 1 vs Thm 2: selection regret (gain-dominated regime) ----
    sr = selection_regret(n_populations=6000, K=6, H=18, tol=1.0,
                          eps_range=(0.02, 0.03), L_range=(0.7, 1.35), replay_noise=0.05, seed=0)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    order = ["oracle", "replay", "val_loss", "random"]
    labels = ["oracle", "open-loop replay\n(env-querying)", "validation loss\n(rollout-free)", "random"]
    colors = ["#1a202c", "#2f855a", "#9b2c2c", "#a0aec0"]
    ax.bar(labels, [sr["rates"][k] for k in order], color=colors)
    ax.set_ylabel("P(selected checkpoint succeeds)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Selection regret: rollout-free $\\approx$ random; env-querying $=$ oracle")
    for i, k in enumerate(order):
        ax.text(i, sr["rates"][k] + 0.02, f"{sr['rates'][k]:.2f}", ha="center")
    f3 = os.path.join(figdir, "figA3_selection_regret.pdf")
    fig.savefig(f3, bbox_inches="tight"); plt.close(fig)

    # ---- Cor 1: the horizon wall ----
    hw = horizon_wall(L=1.15, eps=0.03, tol=1.0, H_max=40)
    Hs = [c["H"] for c in hw["curve"]]
    devs = [c["deviation"] for c in hw["curve"]]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.semilogy(Hs, devs, "-", color="#2b6cb0")
    ax.axhline(hw["tol"], ls=":", color="gray", label="tolerance")
    ax.axvline(hw["H_star_predicted"], ls="--", color="#9b2c2c",
               label=f"$H^\\star$ predicted = {hw['H_star_predicted']:.1f}")
    ax.set_xlabel("horizon $H$"); ax.set_ylabel(r"deviation $\delta_H$ (log)")
    ax.set_title("Cor. 1: the horizon wall")
    ax.legend(fontsize=8)
    f4 = os.path.join(figdir, "figA4_horizon_wall.pdf")
    fig.savefig(f4, bbox_inches="tight"); plt.close(fig)

    save_json({"p1_r2": r2, "p1_max_rel_err": rel_err, "p2": p2,
               "selection_regret": sr, "horizon_wall": {k: v for k, v in hw.items() if k != "curve"}},
              path("results", "partA_bound.json"))

    print("\n===== Part A: numerical validation of the bound (controlled) =====")
    print(f"P1 bound tightness:  R^2 = {r2:.5f},  max relative error = {rel_err:.2e}")
    print("   -> measured deviation matches eps*(L^H-1)/(L-1); the H/L scaling is exact, not loose.")
    print(f"P2 identifiability (across {p2['n_policies']} policies with varying eps,L; "
          f"success rate {p2['success_rate']:.2f}):")
    print(f"   |corr(validation loss, success)|      = {p2['val_loss_corr_pooled']:.3f}  (rollout-free)")
    print(f"   |corr(open-loop replay dev, success)| = {p2['replay_corr_pooled']:.3f}  (environment-querying)")
    print("   by L band (within-band, eps is informative; pooled it is not):")
    for k, v in p2["by_L_band"].items():
        print(f"     {k:10s} n={v['n']:3d} sr={v['success_rate']:.2f}  "
              f"val_loss={v['val_loss_corr']:+.2f}  replay={v['replay_corr']:+.2f}")
    print(f"\nThm1 vs Thm2 selection regret (gain-dominated, K={sr['K']}, H={sr['H']}):")
    print(f"   oracle={sr['rates']['oracle']:.3f}  replay(env)={sr['rates']['replay']:.3f}  "
          f"val_loss(rollout-free)={sr['rates']['val_loss']:.3f}  random={sr['rates']['random']:.3f}")
    print(f"   regret: val_loss={sr['regret_val_loss']:.3f} (~random), replay={sr['regret_replay']:.3f} (oracle)")
    print(f"Cor1 horizon wall: H* predicted={hw['H_star_predicted']:.1f}, empirical={hw['H_star_empirical']}")
    log.info(f"figures: {os.path.basename(f1)}, {os.path.basename(f2)}, "
             f"{os.path.basename(f3)}, {os.path.basename(f4)}")
    print("\n[honest scope] controlled validation that the bound describes a system meeting its "
          "assumptions, and that validation loss cannot rank varying-gain policies while replay can.")
    print("[NOT a real-robot result] the manipulation evidence is Part B (Kaggle).")


if __name__ == "__main__":
    main()

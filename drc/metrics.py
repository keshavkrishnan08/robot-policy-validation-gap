"""SA-3: the eight pre-locked offline metrics (PRD section 8.1).

All metrics are computed from the policy and the held-out validation set without
closed-loop rollouts. M5 executes *open-loop* replays in the sim (queried on
expert observations, no feedback) and M7 reuses those trajectories. M6 needs the
three seed policies for a (task, epoch) pair and is computed by the driver.

Lower-is-better vs higher-is-better signing is handled later in analysis.py; the
raw values stored here are exactly as defined below.
"""
from __future__ import annotations

import math

import numpy as np
import torch

LOG_2PI = math.log(2 * math.pi)


def _to_device(obs, device):
    return {k: v.to(device) for k, v in obs.items()}


@torch.no_grad()
def compute_m1(policy, val_loader, device, K=10) -> float:
    """Validation L1: mean per-dim |pred - expert| over the prediction horizon."""
    errs = []
    for batch in val_loader:
        obs = _to_device(batch["obs"], device)
        expert = batch["action"].to(device)
        pred = policy.predict_deterministic(obs, K=K)
        errs.append((pred - expert).abs().mean().item())
    return float(np.mean(errs))


@torch.no_grad()
def compute_m2(policy, val_loader, device, K=10) -> float:
    """Delta-action MSE (Robomimic-standard). Delta is taken w.r.t. the current
    proprio pose over the dims shared by action and proprio (PRD 8.1)."""
    errs = []
    for batch in val_loader:
        obs = _to_device(batch["obs"], device)
        expert = batch["action"].to(device)                     # (B, H, A)
        pred = policy.predict_deterministic(obs, K=K)
        cur = obs["proprio"][:, -1:, :]                          # (B, 1, P)
        d = min(expert.shape[-1], cur.shape[-1])
        de = expert[..., :d] - cur[..., :d]
        dp = pred[..., :d] - cur[..., :d]
        errs.append(((dp - de) ** 2).mean().item())
    return float(np.mean(errs))


@torch.no_grad()
def compute_m3(policy, val_loader, device, K=20) -> float:
    """Action-distribution entropy proxy: Gaussian differential entropy from the
    per-dim variance of K stochastic samples, summed over dims, averaged."""
    ents = []
    for batch in val_loader:
        obs = _to_device(batch["obs"], device)
        samples = torch.stack([policy.sample(obs) for _ in range(K)], dim=0)  # (K,B,H,A)
        var = samples.var(dim=0, unbiased=False).clamp(min=1e-9)
        ent = 0.5 * (1.0 + LOG_2PI + var.log()).sum(dim=-1).mean(dim=-1)        # (B,)
        ents.append(ent.mean().item())
    return float(np.mean(ents))


@torch.no_grad()
def compute_train_latent_stats(policy, train_loader, device, n_components=100):
    """PCA + Ledoit-Wolf-shrunk inverse covariance of training latents (M4)."""
    from sklearn.covariance import LedoitWolf

    feats = []
    for batch in train_loader:
        obs = _to_device(batch["obs"], device)
        feats.append(policy.encode_latent(obs).cpu().numpy())
    X = np.concatenate(feats, axis=0)                       # (N, D)
    mean = X.mean(axis=0)
    Xc = X - mean
    # PCA via SVD; keep up to n_components (bounded by rank).
    k = min(n_components, Xc.shape[0] - 1, Xc.shape[1])
    k = max(k, 1)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:k]                                      # (k, D)
    Z = Xc @ components.T                                    # (N, k)
    lw = LedoitWolf().fit(Z)
    cov_inv = np.linalg.pinv(lw.covariance_)
    return {"mean": mean, "components": components, "z_mean": Z.mean(axis=0), "cov_inv": cov_inv}


@torch.no_grad()
def compute_m4(policy, val_loader, device, stats) -> float:
    """Mean Mahalanobis distance of validation latents to the train manifold."""
    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    comp = torch.tensor(stats["components"], dtype=torch.float32)
    z_mean = torch.tensor(stats["z_mean"], dtype=torch.float32)
    cov_inv = torch.tensor(stats["cov_inv"], dtype=torch.float32)
    dists = []
    for batch in val_loader:
        obs = _to_device(batch["obs"], device)
        feat = policy.encode_latent(obs).cpu()
        z = (feat - mean) @ comp.T - z_mean                 # (B, k)
        m = torch.einsum("bi,ij,bj->b", z, cov_inv, z).clamp(min=0).sqrt()
        dists.append(m.mean().item())
    return float(np.mean(dists))


@torch.no_grad()
def compute_m5_m7(policy, env, replay_episodes, device, K=10):
    """Open-loop replay distance (M5) and integrated jerk (M7).

    For each episode we query the policy on the *expert* observation sequence,
    roll the predicted actions open-loop in the env from the recorded initial
    state, and measure (a) final eef distance to the expert final pose and
    (b) integrated jerk of the eef trajectory.
    """
    final_dists, jerks = [], []
    n_action_steps = policy.n_action_steps
    for ep in replay_episodes:
        env.reset_to({"state": ep["initial_state"]})
        eef_traj = [env.eef_pose()]
        obs_seq = ep["obs_seq"]  # list of obs dicts, one per query step
        for obs in obs_seq:
            obs_t = {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in obs.items()}
            chunk = policy.predict_action_chunk(obs_t, K=K)[0].cpu().numpy()  # (n_action_steps, A)
            for a in chunk[:n_action_steps]:
                _, _, done, _ = env.step(a)
                eef_traj.append(env.eef_pose())
                if done:
                    break
        final_dists.append(float(np.linalg.norm(env.eef_pose() - ep["final_eef_pose"])))
        jerks.append(_integrated_jerk(np.array(eef_traj)))
    return float(np.mean(final_dists)), float(np.mean(jerks))


def _integrated_jerk(pos: np.ndarray) -> float:
    """Sum of |third position difference| over a (T, 3) trajectory."""
    if pos.shape[0] < 4:
        return 0.0
    jerk = np.diff(pos, n=3, axis=0)
    return float(np.linalg.norm(jerk, axis=-1).sum())


@torch.no_grad()
def compute_m6(policies, val_loader, device, K=10, noise_seed=0) -> float:
    """Inter-seed disagreement: std across seed policies of the deterministic
    prediction, averaged. Same value for all seeds in a (task, epoch) pair.

    A fixed noise_seed is shared across policies so the disagreement reflects
    weight differences only, not diffusion sampling noise (identical weights -> 0).
    """
    disagreements = []
    for batch in val_loader:
        obs = _to_device(batch["obs"], device)
        preds = torch.stack(
            [p.predict_deterministic(obs, K=K, noise_seed=noise_seed) for p in policies], dim=0
        )  # (S,B,H,A)
        std = preds.std(dim=0, unbiased=False).mean(dim=-1).mean(dim=-1)                    # (B,)
        disagreements.append(std.mean().item())
    return float(np.mean(disagreements))


@torch.no_grad()
def compute_m8(policy, val_loader, device, K=20) -> float:
    """Action confidence: negative summed per-dim variance of K samples."""
    confs = []
    for batch in val_loader:
        obs = _to_device(batch["obs"], device)
        samples = torch.stack([policy.sample(obs) for _ in range(K)], dim=0)
        var = samples.var(dim=0, unbiased=False).sum(dim=-1).mean(dim=-1)
        confs.append((-var).mean().item())
    return float(np.mean(confs))


def compute_single_checkpoint(policy, train_loader, val_loader, env, replay_episodes, device, cfg):
    """All per-checkpoint metrics except M6 (which needs the seed ensemble).

    cfg keys: m1_K, m3_K, m4_pca_components, m8_K.
    """
    stats = compute_train_latent_stats(policy, train_loader, device, cfg.get("m4_pca_components", 100))
    m5, m7 = compute_m5_m7(policy, env, replay_episodes, device, K=cfg.get("m1_K", 10))
    return {
        "M1": compute_m1(policy, val_loader, device, K=cfg.get("m1_K", 10)),
        "M2": compute_m2(policy, val_loader, device, K=cfg.get("m1_K", 10)),
        "M3": compute_m3(policy, val_loader, device, K=cfg.get("m3_K", 20)),
        "M4": compute_m4(policy, val_loader, device, stats),
        "M5": m5,
        "M7": m7,
        "M8": compute_m8(policy, val_loader, device, K=cfg.get("m8_K", 20)),
    }

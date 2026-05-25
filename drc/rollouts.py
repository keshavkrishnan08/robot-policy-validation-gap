"""SA-4: closed-loop rollout evaluation (PRD 8.3).

Twenty rollouts per checkpoint on a pre-fixed set of initial conditions, shared
across all checkpoints of a task. Deterministic DDIM sampling with a fixed noise
seed removes within-checkpoint stochasticity; the only variance is across the
(paired) initial conditions.
"""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def rollout_trajectory(policy, env, init_cond, max_steps, device="cpu", K=1, noise_seed=42):
    """Closed-loop rollout returning the eef trajectory (T, 3). Used by the
    Lyapunov estimator to measure how fast deviations grow."""
    policy.eval()
    env.reset_to(init_cond)
    traj = [env.eef_pose()]
    steps = 0
    n_action_steps = policy.n_action_steps
    while steps < max_steps:
        obs = env.get_observation()
        obs_t = {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in obs.items()}
        chunk = policy.predict_action_chunk(obs_t, K=K, noise_seed=noise_seed)[0].cpu().numpy()
        done = False
        for a in chunk[:n_action_steps]:
            _, _, done, info = env.step(a)
            traj.append(env.eef_pose())
            steps += 1
            if done or steps >= max_steps:
                break
        if done:
            break
    return np.asarray(traj)


@torch.no_grad()
def evaluate_perturbation_robustness(
    policy, env, eval_conditions, max_steps, perturb_fn, device="cpu", K=10, noise_seed=42
):
    """Run rollouts on *perturbed* initial conditions (Upgrade 3: perturbation-gap).

    `perturb_fn(cond) -> cond` applies a small perturbation to an init condition.
    Returns the same dict as evaluate_checkpoint plus 'perturbation_function'.
    Intended for secondary analysis: do val-loss-selected checkpoints degrade more
    under perturbation than rollout-best checkpoints?
    """
    perturbed = [perturb_fn(c) for c in eval_conditions]
    result = evaluate_checkpoint(policy, env, perturbed, max_steps, device, K, noise_seed)
    result["perturbation_function"] = getattr(perturb_fn, "__name__", "anonymous")
    return result


@torch.no_grad()
def evaluate_checkpoint(policy, env, eval_conditions, max_steps, device="cpu", K=10, noise_seed=42):
    """Return {"successes": [0/1 per condition], "success_rate", "state_hashes"}."""
    policy.eval()
    n_action_steps = policy.n_action_steps
    successes, hashes = [], []

    for cond in eval_conditions:
        env.reset_to(cond)
        hashes.append(env.state_hash())
        success = False
        steps = 0
        while steps < max_steps:
            obs = env.get_observation()
            obs_t = {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in obs.items()}
            chunk = policy.predict_action_chunk(obs_t, K=K, noise_seed=noise_seed)[0].cpu().numpy()
            done = False
            for a in chunk[:n_action_steps]:
                _, _, done, info = env.step(a)
                steps += 1
                if info.get("success"):
                    success = True
                    done = True
                if done or steps >= max_steps:
                    break
            if done:
                break
        successes.append(int(success))

    return {
        "successes": successes,
        "num_successes": int(sum(successes)),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "state_hashes": hashes,
    }

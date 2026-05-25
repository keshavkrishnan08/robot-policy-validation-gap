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

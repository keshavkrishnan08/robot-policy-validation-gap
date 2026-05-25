"""Synthetic demos + a tiny deterministic env.

Purpose: exercise the *entire* pipeline (train -> metrics -> rollouts ->
analysis) on CPU with no external data, so the code can be verified end to end.
The synthetic task is a 2-D reaching problem:

  state s in R^P (first 2 dims are the planar position; rest are filler dofs)
  expert action a = clip(k * (goal - s), -1, 1)         (proportional controller)
  dynamics       s' = s + dt * a[:P]
  image          a Gaussian blob whose centre encodes s[:2]
  success        ||s[:2] - goal[:2]|| < tol

Because the expert policy is a smooth deterministic function of the state, a
trained Diffusion Policy can fit it and rollouts produce a graded success rate.
"""
from __future__ import annotations

import numpy as np

from drc.data.dataset import SequenceDataset

IMG_C, IMG_H, IMG_W = 3, 12, 12
PROPRIO_DIM = 4
ACTION_DIM = 4
GOAL = np.array([0.8, 0.8, 0.0, 0.0], dtype=np.float32)
DT = 0.25
KP = 0.8
SUCCESS_TOL = 0.18


def _render(state: np.ndarray) -> np.ndarray:
    """(P,) state -> (C,H,W) blob image in [0,1], centred by state[:2] in [-1,1]."""
    cx = (state[0] + 1.0) / 2.0 * (IMG_W - 1)
    cy = (state[1] + 1.0) / 2.0 * (IMG_H - 1)
    ys, xs = np.mgrid[0:IMG_H, 0:IMG_W]
    blob = np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / 4.0)).astype(np.float32)
    img = np.stack([blob, blob * 0.5, 1.0 - blob], axis=0)
    return np.clip(img, 0.0, 1.0)


def _expert_action(state: np.ndarray) -> np.ndarray:
    return np.clip(KP * (GOAL - state), -1.0, 1.0).astype(np.float32)


def make_synthetic_dataset(n_demos: int, length: int = 40, seed: int = 0):
    """Return (SequenceDataset, info) with proportional-controller demos."""
    rng = np.random.default_rng(seed)
    images = np.zeros((n_demos, length, IMG_C, IMG_H, IMG_W), dtype=np.float32)
    proprio = np.zeros((n_demos, length, PROPRIO_DIM), dtype=np.float32)
    actions = np.zeros((n_demos, length, ACTION_DIM), dtype=np.float32)

    for d in range(n_demos):
        s = rng.uniform(-1.0, -0.2, size=PROPRIO_DIM).astype(np.float32)
        for t in range(length):
            a = _expert_action(s)
            proprio[d, t] = s
            actions[d, t] = a + rng.normal(0, 0.01, size=ACTION_DIM).astype(np.float32)
            images[d, t] = _render(s)
            s = np.clip(s + DT * a[:PROPRIO_DIM], -1.0, 1.0)

    ds = SequenceDataset(images, proprio, actions, n_obs_steps=2, horizon=16)
    info = {
        "image_shape": (IMG_C, IMG_H, IMG_W),
        "proprio_dim": PROPRIO_DIM,
        "action_dim": ACTION_DIM,
        "crop_shape": (10, 10),
    }
    return ds, info


class SyntheticEnv:
    """Minimal closed-loop env matching the synthetic dynamics.

    Implements the subset of the env API the rollout/replay code calls:
    reset_to, get_observation, step, plus state hashing for SA-4 verification.
    """

    def __init__(self, n_obs_steps: int = 2, max_steps: int = 60):
        self.n_obs_steps = n_obs_steps
        self.max_steps = max_steps
        self._state = None
        self._hist = None
        self._t = 0

    def reset_to(self, init_cond):
        self._state = np.asarray(init_cond["state"], dtype=np.float32).copy()
        self._hist = [self._state.copy() for _ in range(self.n_obs_steps)]
        self._t = 0
        return self.get_observation()

    def state_hash(self) -> str:
        import hashlib

        return hashlib.sha256(self._state.tobytes()).hexdigest()[:16]

    def get_observation(self):
        imgs = np.stack([_render(s) for s in self._hist[-self.n_obs_steps :]], axis=0)
        pros = np.stack(self._hist[-self.n_obs_steps :], axis=0)
        return {"image": imgs[None], "proprio": pros[None]}  # batch dim = 1

    def eef_pose(self) -> np.ndarray:
        return self._state[:3].copy()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        self._state = np.clip(self._state + DT * action[:PROPRIO_DIM], -1.0, 1.0)
        self._hist.append(self._state.copy())
        self._t += 1
        dist = np.linalg.norm(self._state[:2] - GOAL[:2])
        success = bool(dist < SUCCESS_TOL)
        done = success or self._t >= self.max_steps
        return self.get_observation(), 0.0, done, {"success": success, "dist": dist}


def make_eval_conditions(n: int, seed: int = 123):
    """Pre-fixed initial conditions for rollouts (PRD 8.3)."""
    rng = np.random.default_rng(seed)
    return [{"state": rng.uniform(-1.0, -0.2, size=PROPRIO_DIM).astype(np.float32).tolist()} for _ in range(n)]


def expert_episode(init_cond, length: int = 40):
    """An expert open-loop reference episode from a given init (for M5/M7)."""
    s = np.asarray(init_cond["state"], dtype=np.float32).copy()
    obs_states, acts = [], []
    for _ in range(length):
        a = _expert_action(s)
        obs_states.append(s.copy())
        acts.append(a.copy())
        s = np.clip(s + DT * a[:PROPRIO_DIM], -1.0, 1.0)
    return {
        "obs_states": np.array(obs_states, dtype=np.float32),
        "actions": np.array(acts, dtype=np.float32),
        "initial_state": np.asarray(init_cond["state"], dtype=np.float32),
        "final_eef_pose": s[:3].copy(),
    }

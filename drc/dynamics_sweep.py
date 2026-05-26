"""Part A: a controlled system whose closed-loop gain L is a direct knob.

On real tasks L is unknown and confounded, so the bound's prediction cannot be
tested in isolation. Here we set L by construction and realize the deviation
recurrence delta_{t+1} = L*delta_t + e_t exactly:

  dynamics   f(s,a) = a            (fully actuated; the action sets the next state)
  expert     a*(s) = L * s         => expert closed loop g(s)=L s, Lip(g)=L
  reference  s*_t = L^t s_0        (the expert rollout the policy must track)
  success    a policy tracks the reference within tolerance for H steps

A learned policy with action error e_t has tracking error delta_{t+1}=L*delta_t+e_t:
self-correcting for L<1, marginal at L=1, expansive for L>1. With limited demos the
policy is accurate on visited states but free off them; for L>1 the rollout leaves
the demo region, so success is set by off-distribution behavior that rollout-free
metrics cannot see but an open-loop replay can. This is the controlled test of both
the (L^H-1)/(L-1) prediction (P1) and the identifiability limit (P2).
"""
from __future__ import annotations

import numpy as np

from drc.data.dataset import SequenceDataset
from drc.data.synthetic import _render, IMG_C, IMG_H, IMG_W

DIM = 4
TOL = 0.45            # per-step tracking tolerance (looser, so success is graded not all-or-nothing)
TRACK_FRACTION = 0.7  # episode succeeds if >= this fraction of steps stayed within TOL


def _state_to_img(s):
    return _render(np.clip(s, -1, 1))


def make_sweep_dataset(L: float, n_demos: int = 8, length: int = 24, seed: int = 0):
    rng = np.random.default_rng(seed)
    images = np.zeros((n_demos, length, IMG_C, IMG_H, IMG_W), dtype=np.float32)
    proprio = np.zeros((n_demos, length, DIM), dtype=np.float32)
    actions = np.zeros((n_demos, length, DIM), dtype=np.float32)
    for d in range(n_demos):
        # start in a band; for L>1 keep s0 small so trajectories stay renderable
        s = rng.uniform(-0.4, 0.4, size=DIM).astype(np.float32)
        for t in range(length):
            a = (L * s).astype(np.float32)           # expert action = next state
            proprio[d, t] = s
            actions[d, t] = a
            images[d, t] = _state_to_img(s)
            s = np.clip(a, -1.0, 1.0)
    ds = SequenceDataset(images, proprio, actions, n_obs_steps=2, horizon=16)
    info = {"image_shape": (IMG_C, IMG_H, IMG_W), "proprio_dim": DIM,
            "action_dim": DIM, "crop_shape": (10, 10), "L": L}
    return ds, info


class SweepEnv:
    """Regulation/tracking env with closed-loop gain L. Same interface as SyntheticEnv."""

    def __init__(self, L: float, n_obs_steps: int = 2, max_steps: int = 48):
        self.L = L
        self.n_obs_steps = n_obs_steps
        self.max_steps = max_steps
        self._s = None
        self._ref = None       # expert reference state at current step
        self._hist = None
        self._t = 0

    def reset_to(self, init_cond):
        self._s = np.asarray(init_cond["state"], dtype=np.float32).copy()
        self._ref = self._s.copy()
        self._hist = [self._s.copy() for _ in range(self.n_obs_steps)]
        self._t = 0
        self._in_tol = 0
        return self.get_observation()

    def state_hash(self):
        import hashlib
        return hashlib.sha256(self._s.tobytes()).hexdigest()[:16]

    def get_observation(self):
        imgs = np.stack([_state_to_img(s) for s in self._hist[-self.n_obs_steps:]], axis=0)
        pros = np.stack(self._hist[-self.n_obs_steps:], axis=0)
        return {"image": imgs[None], "proprio": pros[None]}

    def eef_pose(self):
        return self._s[:3].copy()

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)[:DIM]
        self._s = np.clip(a, -1.0, 1.0)                 # f(s,a)=a
        self._ref = np.clip(self.L * self._ref, -1.0, 1.0)   # expert reference advances by L
        self._hist.append(self._s.copy())
        self._t += 1
        dev = float(np.linalg.norm(self._s[:2] - self._ref[:2]))
        self._in_tol += int(dev < TOL)
        timed_out = self._t >= self.max_steps
        done = timed_out                                    # run the full horizon (graded success)
        # Graded success: did the policy track within tolerance for enough of the horizon?
        # Set info['success'] only at the terminal step (evaluate_checkpoint marks success the
        # first time it sees True).
        frac = self._in_tol / max(self._t, 1)
        success = timed_out and (frac >= TRACK_FRACTION)
        return self.get_observation(), 0.0, done, {"success": success, "dev": dev, "track_frac": frac}


def sweep_eval_conditions(n: int, seed: int = 321):
    rng = np.random.default_rng(seed)
    return [{"state": rng.uniform(-0.3, 0.3, size=DIM).astype(np.float32).tolist()} for _ in range(n)]

"""LIBERO demonstration loader and env wrapper (Kaggle GPU run only).

Requires: libero, robosuite, mujoco, h5py. Not importable on the dev box; the
synthetic path never touches this module. Loads the HDF5 demos shipped with the
LIBERO benchmark into a SequenceDataset and wraps the Robosuite env into the
common rollout interface (reset_to / get_observation / step / eef_pose).
"""
from __future__ import annotations

import numpy as np

from drc.data.dataset import SequenceDataset

# LIBERO images render at 128x128 by default; we resize to 84x84 (DP recipe).
IMG_SIZE = 84
CROP = (76, 76)


def _resize(img: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """(H, W, 3) uint8 -> (3, size, size) float32 in [0, 1]."""
    try:
        import cv2

        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    except ImportError:  # pragma: no cover
        from PIL import Image

        img = np.asarray(Image.fromarray(img).resize((size, size)))
    return (img.astype(np.float32) / 255.0).transpose(2, 0, 1)


def load_libero_dataset(task_cfg: dict, n_obs_steps: int = 2, horizon: int = 16, val_frac: float = 0.1):
    """Return (train_ds, val_ds, info). Splits demos by trajectory."""
    import h5py
    from libero.libero import get_libero_path
    import os

    bench = task_cfg["benchmark"]
    demo_dir = os.path.join(get_libero_path("datasets"), bench)
    # File name convention: <bddl>_demo.hdf5
    fname = next(f for f in os.listdir(demo_dir) if task_cfg["bddl"] in f and f.endswith(".hdf5"))
    path = os.path.join(demo_dir, fname)

    images, proprio, actions = [], [], []
    with h5py.File(path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        for d in demos:
            g = f["data"][d]
            rgb = g["obs"]["agentview_rgb"][()]  # (T, H, W, 3) uint8
            ee_pos = g["obs"]["ee_pos"][()]
            ee_ori = g["obs"]["ee_ori"][()]
            grip = g["obs"]["gripper_states"][()]
            pro = np.concatenate([ee_pos, ee_ori, grip], axis=-1).astype(np.float32)
            img = np.stack([_resize(fr) for fr in rgb], axis=0)
            images.append(img)
            proprio.append(pro)
            actions.append(g["actions"][()].astype(np.float32))

    return _split_and_pack(images, proprio, actions, n_obs_steps, horizon, val_frac, CROP)


def _split_and_pack(images, proprio, actions, n_obs_steps, horizon, val_frac, crop):
    # Pad ragged trajectories to a common length by repeating the final frame.
    L = max(a.shape[0] for a in actions)

    def pad(arr_list):
        out = []
        for a in arr_list:
            if a.shape[0] < L:
                rep = np.repeat(a[-1:], L - a.shape[0], axis=0)
                a = np.concatenate([a, rep], axis=0)
            out.append(a)
        return np.stack(out, axis=0)

    imgs, pros, acts = pad(images), pad(proprio), pad(actions)
    n = imgs.shape[0]
    n_val = max(1, int(round(n * val_frac)))
    val_idx = np.arange(n - n_val, n)  # last demos held out, deterministic
    tr_idx = np.arange(0, n - n_val)

    train_ds = SequenceDataset(imgs[tr_idx], pros[tr_idx], acts[tr_idx], n_obs_steps, horizon)
    val_ds = SequenceDataset(imgs[val_idx], pros[val_idx], acts[val_idx], n_obs_steps, horizon)
    info = {
        "image_shape": (3, IMG_SIZE, IMG_SIZE),
        "proprio_dim": pros.shape[-1],
        "action_dim": acts.shape[-1],
        "crop_shape": crop,
        "n_demos": n,
    }
    return train_ds, val_ds, info

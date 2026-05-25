"""Robomimic demonstration loader (Kaggle GPU run only).

Requires: robomimic, robosuite, mujoco, h5py. Loads the proficient-human (PH)
image datasets into a SequenceDataset using the same packing as LIBERO.
"""
from __future__ import annotations

import os

import numpy as np

from drc.data.libero_adapter import _resize, _split_and_pack, CROP

# agentview + the second camera that the dual-arm Transport task exposes.
IMG_KEYS = {
    "square": ["agentview_image"],
    "lift": ["agentview_image"],
    "transport": ["shouldercamera0_image", "shouldercamera1_image"],
}
PROPRIO_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]


def load_robomimic_dataset(task_cfg: dict, data_root: str, n_obs_steps=2, horizon=16, val_frac=0.1):
    import h5py

    bench = task_cfg["benchmark"]
    path = os.path.join(data_root, bench, task_cfg["dataset"], "image.hdf5")
    img_keys = IMG_KEYS.get(bench, ["agentview_image"])
    pro_keys = list(PROPRIO_KEYS)
    if bench == "transport":  # dual arm -> add the second arm's proprio
        pro_keys += ["robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos"]

    images, proprio, actions = [], [], []
    with h5py.File(path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        for d in demos:
            g = f["data"][d]
            # Use the primary camera for the encoder; extra cameras averaged in.
            frames = g["obs"][img_keys[0]][()]
            img = np.stack([_resize(fr) for fr in frames], axis=0)
            pro = np.concatenate([g["obs"][k][()] for k in pro_keys], axis=-1).astype(np.float32)
            images.append(img)
            proprio.append(pro)
            actions.append(g["actions"][()].astype(np.float32))

    return _split_and_pack(images, proprio, actions, n_obs_steps, horizon, val_frac, CROP)

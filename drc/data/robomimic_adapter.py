"""Robomimic demonstration loader (Kaggle GPU run only).

Requires: robomimic, robosuite, mujoco, h5py. Loads the proficient-human (PH)
image datasets into a SequenceDataset using the same packing as LIBERO.
"""
from __future__ import annotations

import glob
import os

import numpy as np

from drc.data.libero_adapter import _resize, _split_and_pack, CROP


def _find_image_hdf5(task_dir: str) -> str:
    """Robomimic image datasets are named image.hdf5 / image_v141.hdf5 / image_v15.hdf5 depending
    on the robosuite version. Glob rather than hardcode; error informatively if absent."""
    cands = sorted(glob.glob(os.path.join(task_dir, "image*.hdf5")))
    if not cands:
        listing = os.listdir(task_dir) if os.path.isdir(task_dir) else "(dir missing)"
        raise FileNotFoundError(
            f"No image*.hdf5 in {task_dir}. Present: {listing}. "
            "Did download_task.py run for this task? Check --hdf5_types image.")
    return cands[0]

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
    path = _find_image_hdf5(os.path.join(data_root, bench, task_cfg["dataset"]))
    img_keys = IMG_KEYS.get(bench, ["agentview_image"])
    pro_keys = list(PROPRIO_KEYS)
    if bench == "transport":  # dual arm -> add the second arm's proprio
        pro_keys += ["robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos"]

    images, proprio, actions = [], [], []
    with h5py.File(path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        avail = list(f["data"][demos[0]]["obs"].keys())
        for k in img_keys + pro_keys:
            if k not in avail:
                raise KeyError(f"obs key '{k}' not in {path}. Available obs keys: {avail}")
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


def load_replay_episodes(task_cfg, data_root, n=4, n_obs_steps=2):
    """Open-loop replay episodes (for M5/M7) from the LAST n demos (the held-out side).
    Each: initial sim state, a stacked-obs query per action chunk, and the expert final eef."""
    import h5py

    bench = task_cfg["benchmark"]
    path = _find_image_hdf5(os.path.join(data_root, bench, task_cfg["dataset"]))
    cam = IMG_KEYS.get(bench, ["agentview_image"])[0]
    pro_keys = list(PROPRIO_KEYS) + (["robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos"]
                                     if bench == "transport" else [])
    episodes = []
    with h5py.File(path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))[-n:]
        for d in demos:
            g = f["data"][d]
            states = g["states"][()]
            frames = g["obs"][cam][()]
            pro = np.concatenate([g["obs"][k][()] for k in pro_keys], axis=-1).astype(np.float32)
            imgs = np.stack([_resize(fr if fr.dtype == np.uint8 else (fr * 255).astype(np.uint8))
                             for fr in frames], axis=0)
            seq = []
            for t in range(0, len(imgs), 8):
                lo = max(0, t - n_obs_steps + 1)
                wi, wp = imgs[lo:t + 1], pro[lo:t + 1]
                while len(wi) < n_obs_steps:
                    wi = np.concatenate([imgs[:1], wi], 0); wp = np.concatenate([pro[:1], wp], 0)
                seq.append({"image": wi[-n_obs_steps:][None], "proprio": wp[-n_obs_steps:][None]})
            episodes.append({"obs_seq": seq, "initial_state": states[0],
                             "final_eef_pose": pro[-1][:3].copy()})
    return episodes

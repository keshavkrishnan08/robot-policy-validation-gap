"""Dataset / env / replay providers bridging the locked task table to the
training and evaluation code. Keeps backend specifics out of the drivers.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from drc import config


def dataset_provider(task: str, synthetic: bool = False, n_obs_steps=2, horizon=16):
    """Return a zero-arg callable -> (train_ds, val_ds, info)."""
    tasks = config.load_tasks()
    cfg = tasks[task]

    if synthetic:
        from drc.data.synthetic import make_synthetic_dataset

        def provide():
            # Vary demo count/length slightly by task to diversify difficulty.
            n = 8
            ds, info = make_synthetic_dataset(n_demos=n, length=28, seed=hash(task) % 1000)
            val, _ = make_synthetic_dataset(n_demos=3, length=28, seed=(hash(task) % 1000) + 7)
            return ds, val, info

        return provide

    if cfg["suite"] == "libero":  # pragma: no cover - Kaggle
        from drc.data.libero_adapter import load_libero_dataset

        return lambda: load_libero_dataset(cfg, n_obs_steps, horizon)

    from drc.data.robomimic_adapter import load_robomimic_dataset  # pragma: no cover

    data_root = os.environ.get("ROBOMIMIC_DATA", "/kaggle/working/data/robomimic")
    return lambda: load_robomimic_dataset(cfg, data_root, n_obs_steps, horizon)


def env_for(task: str, synthetic: bool = False, n_obs_steps=2):
    from drc.envs import make_env

    tasks = config.load_tasks()
    return make_env(task, tasks[task], synthetic=synthetic, n_obs_steps=n_obs_steps)


def build_replay_episodes(task: str, synthetic: bool, val_ds, n_obs_steps=2, max_episodes=3):
    """Construct M5/M7 open-loop replay episodes from the validation set.

    Synthetic: render expert states into stacked obs queries. Real backends would
    reconstruct obs from the stored val demos + sim initial state (Kaggle).
    """
    if synthetic:
        from drc.data.synthetic import expert_episode, _render, make_eval_conditions

        episodes = []
        conds = make_eval_conditions(max_episodes, seed=(hash(task) % 1000) + 11)
        for c in conds:
            ep = expert_episode(c, length=24)
            states = ep["obs_states"]
            # Build one stacked obs query per action chunk (every n_action_steps=8).
            obs_seq = []
            for t in range(0, len(states), 8):
                window = states[max(0, t - n_obs_steps + 1) : t + 1]
                while len(window) < n_obs_steps:
                    window = np.concatenate([states[:1], window], axis=0)
                window = window[-n_obs_steps:]
                imgs = np.stack([_render(s) for s in window], axis=0)[None]
                pros = window[None]
                obs_seq.append({"image": imgs, "proprio": pros})
            episodes.append(
                {"obs_seq": obs_seq, "initial_state": ep["initial_state"], "final_eef_pose": ep["final_eef_pose"]}
            )
        return episodes

    # Real backends: build from the val demos' obs + initial sim states. Defensive — any failure
    # returns [] so M5/M7 degrade to NaN (logged) rather than crashing the whole run.
    from drc import config
    from drc.utils import get_logger
    log = get_logger("providers")
    cfg = config.load_tasks()[task]
    try:
        if cfg["suite"] == "libero":
            from drc.data.libero_adapter import load_replay_episodes
            return load_replay_episodes(cfg, n=max_episodes, n_obs_steps=n_obs_steps)
        import os
        from drc.data.robomimic_adapter import load_replay_episodes
        data_root = os.environ.get("ROBOMIMIC_DATA", "/kaggle/working/data/robomimic")
        return load_replay_episodes(cfg, data_root, n=max_episodes, n_obs_steps=n_obs_steps)
    except Exception as e:
        log.warning(f"replay episodes unavailable for {task} ({type(e).__name__}: {e}); M5/M7 -> NaN")
        return []

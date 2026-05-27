"""Environment factory exposing one rollout interface for every backend.

Common interface used by drc/rollouts.py and metric M5:
  reset_to(init_cond) -> obs
  get_observation()    -> {"image": (1, n_obs, C, H, W), "proprio": (1, n_obs, P)}
  step(action)         -> obs, reward, done, info{"success": bool}
  eef_pose()           -> (3,) end-effector position
  state_hash()         -> short deterministic hash of the reset state (SA-4 check)

Synthetic backend lives in drc.data.synthetic; real backends wrap a Robosuite
env and are only constructed on Kaggle.
"""
from __future__ import annotations

import os as _os

# Pin MuJoCo/PyOpenGL to GPU EGL offscreen rendering BEFORE any mujoco/robosuite import.
# Without this, on a headless node (Kaggle) MuJoCo's GL backend auto-selection blocks
# indefinitely the first time an offscreen render context is created — which is the metrics/
# rollout step, after training has already burned hours. setdefault lets the shell override.
_os.environ.setdefault("MUJOCO_GL", "egl")
_os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np


def _silence_egl_shutdown_noise():
    """robosuite's EGL context __del__ runs during interpreter shutdown, after EGL is already torn
    down, so it raises a harmless EGLError that Python prints as a multi-line 'Exception ignored in'
    traceback. It does not affect rendering or results — it just buries real errors in the log. Route
    those specific unraisable exceptions to /dev/null while leaving every other one untouched."""
    import sys

    prev = getattr(sys, "unraisablehook", None)

    def hook(unraisable):
        exc = unraisable.exc_value
        name = type(exc).__name__ if exc else ""
        msg = f"{name}: {exc}" if exc else ""
        if "EGLError" in name or "eglDestroyContext" in msg or "EGL_NOT_INITIALIZED" in msg:
            return  # swallow the cosmetic EGL teardown error
        if prev is not None:
            prev(unraisable)

    sys.unraisablehook = hook


_silence_egl_shutdown_noise()


def _ensure_libero_config():
    """Write ~/.libero/config.yaml with default package-relative paths so LIBERO never enters its
    interactive first-run setup (which prompts on stdin and hangs notebooks). Idempotent; no-op if
    LIBERO is absent or the config already exists. Uses find_spec to avoid executing libero.libero."""
    import importlib.util
    import os

    cfg_dir = os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero"))
    cfg_file = os.path.join(cfg_dir, "config.yaml")
    if os.path.exists(cfg_file):
        return
    try:
        spec = importlib.util.find_spec("libero.libero")
        import yaml
    except (ImportError, ModuleNotFoundError, ValueError):
        return  # LIBERO (or yaml) not installed -> nothing to configure
    if spec is None or not spec.origin:
        return
    root = os.path.dirname(os.path.abspath(spec.origin))
    cfg = {
        "benchmark_root": root,
        "bddl_files": os.path.join(root, "bddl_files"),
        "init_states": os.path.join(root, "init_files"),
        "datasets": os.path.join(os.path.dirname(root), "datasets"),
        "assets": os.path.join(root, "assets"),
    }
    os.makedirs(cfg_dir, exist_ok=True)
    with open(cfg_file, "w") as f:
        yaml.safe_dump(cfg, f)


def make_env(task_name: str, task_cfg: dict, synthetic: bool = False, n_obs_steps: int = 2):
    if synthetic:
        from drc.data.synthetic import SyntheticEnv

        return SyntheticEnv(n_obs_steps=n_obs_steps, max_steps=task_cfg.get("max_steps", 60))
    if task_cfg["suite"] == "libero":
        return _LiberoEnv(task_cfg, n_obs_steps)
    return _RobomimicEnv(task_cfg, n_obs_steps)


class _RobosuiteEnvBase:
    """Shared adapter turning a Robosuite obs dict into our stacked format."""

    def __init__(self, n_obs_steps: int):
        self.n_obs_steps = n_obs_steps
        self._hist = []
        self.env = None  # set by subclass

    # subclasses implement _raw_obs_to_dict and _make_env / reset
    def _push(self, raw):
        self._hist.append(raw)
        while len(self._hist) < self.n_obs_steps:
            self._hist.insert(0, raw)
        self._hist = self._hist[-self.n_obs_steps :]

    def get_observation(self):
        imgs = np.stack([o["image"] for o in self._hist], axis=0)[None]
        pros = np.stack([o["proprio"] for o in self._hist], axis=0)[None]
        return {"image": imgs.astype(np.float32), "proprio": pros.astype(np.float32)}

    def state_hash(self):
        import hashlib

        st = self.env.sim.get_state().flatten()
        return hashlib.sha256(np.asarray(st, dtype=np.float64).tobytes()).hexdigest()[:16]


class _LiberoEnv(_RobosuiteEnvBase):  # pragma: no cover - Kaggle only
    def __init__(self, task_cfg, n_obs_steps):
        super().__init__(n_obs_steps)
        _ensure_libero_config()   # write ~/.libero/config.yaml so LIBERO never prompts on stdin
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
        from drc.data.libero_adapter import _resize

        self._resize = _resize
        bench = benchmark.get_benchmark_dict()[task_cfg["benchmark"]]()
        task = bench.get_task(task_cfg["task_id"])
        bddl = bench.get_task_bddl_file_path(task_cfg["task_id"])
        self.env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
        self.task = task

    def _raw(self, obs):
        img = self._resize(obs["agentview_image"][::-1])  # flip per LIBERO convention
        pro = np.concatenate([obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"]])
        return {"image": img, "proprio": pro.astype(np.float32), "eef": obs["robot0_eef_pos"]}

    def reset_to(self, init_cond):
        self.env.reset()
        self.env.set_init_state(init_cond["state"])
        obs = self.env.env._get_observations()
        self._hist = []
        self._push(self._raw(obs))
        return self.get_observation()

    def eef_pose(self):
        return np.asarray(self._hist[-1]["eef"], dtype=np.float32)

    def step(self, action):
        obs, reward, done, info = self.env.step(np.asarray(action).reshape(-1))
        self._push(self._raw(obs))
        success = bool(self.env.check_success()) if hasattr(self.env, "check_success") else bool(done)
        return self.get_observation(), reward, success or done, {"success": success}


class _RobomimicEnv(_RobosuiteEnvBase):  # pragma: no cover - Kaggle only
    def __init__(self, task_cfg, n_obs_steps):
        super().__init__(n_obs_steps)
        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.file_utils as FileUtils
        import os
        from drc.data.libero_adapter import _resize

        self._resize = _resize
        from drc.data.robomimic_adapter import _find_image_hdf5

        data_root = os.environ.get("ROBOMIMIC_DATA", "/kaggle/working/data/robomimic")
        dataset = _find_image_hdf5(os.path.join(data_root, task_cfg["benchmark"], task_cfg["dataset"]))
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset)
        self.env = EnvUtils.create_env_from_metadata(env_meta, render=False, render_offscreen=True)
        self._cam = "agentview"

    def _raw(self, obs):
        img = self._resize((obs[f"{self._cam}_image"] * 255).astype(np.uint8))
        pro = np.concatenate(
            [obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"]]
        )
        return {"image": img, "proprio": pro.astype(np.float32), "eef": obs["robot0_eef_pos"]}

    def reset_to(self, init_cond):
        obs = self.env.reset_to({"states": init_cond["state"]})
        self._hist = []
        self._push(self._raw(obs))
        return self.get_observation()

    def eef_pose(self):
        return np.asarray(self._hist[-1]["eef"], dtype=np.float32)

    def step(self, action):
        obs, reward, done, info = self.env.step(np.asarray(action).reshape(-1))
        success = bool(self.env.is_success()["task"])
        self._push(self._raw(obs))
        return self.get_observation(), reward, success or done, {"success": success}

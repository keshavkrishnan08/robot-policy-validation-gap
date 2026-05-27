"""PolicyAdapter implementations for Part C (real-robot external validation).

These wrap public SIMPLER-comparison policies behind the minimal interface
drc.external_validation expects (predict / sample). Kaggle-only: they import heavy,
framework-specific deps lazily and are NOT runnable on the dev box. UNTESTED here
(no weights / GPU) — verify each against its upstream API on first Kaggle run.

Run each framework in its own session (TF for RT-1, JAX for Octo, PyTorch for OpenVLA);
write external/<policy>_metrics.json with compute_external_metrics, then combine offline.
"""
from __future__ import annotations

import numpy as np


class OpenVLAAdapter:  # pragma: no cover - Kaggle PyTorch env, needs openvla-7b weights
    """OpenVLA (HuggingFace, PyTorch). Model card: openvla/openvla-7b.

    predict(obs) returns the deterministic 7-DoF action; sample(obs) perturbs the decoding
    temperature to draw stochastic samples for the entropy/confidence metrics (M3/M8).
    """

    name = "openvla"

    def __init__(self, model_id="openvla/openvla-7b", device="cuda", unnorm_key="bridge_orig"):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.torch = torch
        self.device = device
        self.unnorm_key = unnorm_key
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
        ).to(device)

    def _prompt(self, instruction):
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"

    def predict(self, obs: dict) -> np.ndarray:
        from PIL import Image

        img = Image.fromarray(obs["image"])  # HxWx3 uint8
        inputs = self.processor(self._prompt(obs.get("instruction", "complete the task")), img).to(
            self.device, dtype=self.torch.bfloat16
        )
        return np.asarray(self.model.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False))

    def sample(self, obs: dict) -> np.ndarray:
        from PIL import Image

        img = Image.fromarray(obs["image"])
        inputs = self.processor(self._prompt(obs.get("instruction", "complete the task")), img).to(
            self.device, dtype=self.torch.bfloat16
        )
        return np.asarray(
            self.model.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=True, temperature=1.0)
        )


class SimplerWrappedAdapter:  # pragma: no cover - Kaggle, via SimplerEnv-OpenVLA wrappers
    """Generic adapter over a SimplerEnv-OpenVLA policy object (RT-1, RT-1-X, Octo, RT-2-X).

    Construct with a policy that exposes `step(obs)->action` (the SimplerEnv inference API);
    `sample` falls back to `predict` for deterministic policies (RT-1), which makes M3/M8
    degenerate for them — report those metrics only for policies that support sampling.
    """

    def __init__(self, policy, name):
        self.policy = policy
        self.name = name

    def predict(self, obs: dict) -> np.ndarray:
        out = self.policy.step(obs)
        return np.asarray(out[0] if isinstance(out, (tuple, list)) else out).reshape(-1)

    def sample(self, obs: dict) -> np.ndarray:
        fn = getattr(self.policy, "sample", None)
        if fn is None:
            raise NotImplementedError(f"{self.name} is deterministic; M3/M8 not applicable")
        out = fn(obs)
        return np.asarray(out[0] if isinstance(out, (tuple, list)) else out).reshape(-1)

"""Small shared helpers: paths, JSON/CSV IO, hashing, logging."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def path(*parts: str) -> str:
    return os.path.join(ROOT, *parts)


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def ckpt_path(task: str, seed: int, epoch: int) -> str:
    return path("checkpoints", task, str(seed), f"epoch_{epoch}.pt")


def metrics_path(task: str, seed: int, epoch: int) -> str:
    return path("metrics", task, str(seed), f"epoch_{epoch}_metrics.json")


def rollouts_path(task: str, seed: int, epoch: int) -> str:
    return path("rollouts", task, str(seed), f"epoch_{epoch}_rollouts.json")


def save_json(obj: Any, p: str) -> None:
    ensure_dir(os.path.dirname(p))
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(p: str) -> Any:
    with open(p) as f:
        return json.load(f)


def _json_default(o: Any):
    # numpy scalars/arrays -> native python for clean JSON.
    try:
        import numpy as np

        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    raise TypeError(f"not JSON serializable: {type(o)}")


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] %(name)s: %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger

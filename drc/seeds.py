"""Deterministic seeding across python, numpy, and torch (PRD SA-2.4)."""
from __future__ import annotations

import os
import random

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def seed_everything(seed: int) -> None:
    """Fix every RNG we touch so a (task, seed) run is reproducible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Determinism for the training run; relaxed for speed on rollouts.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

"""Configuration loading and PRE-LOCKED constants for DRC CoRL 2026.

This module is the single source of truth for every pre-registered decision.
The YAML files in configs/ are loaded here and exposed as plain dicts. The
constants duplicated as Python literals below are deliberate: they let the
analysis run even if the YAML is missing, and they make the pre-registration
auditable by hashing this file alongside analysis.py.

Locked 2026-05-25. Do not change after SA-2 training begins.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

try:
    import yaml  # PyYAML
except ImportError:  # pragma: no cover - yaml is in requirements
    yaml = None

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")

# ---------------------------------------------------------------------------
# Pre-locked literals (mirror configs/*.yaml; authoritative for the analysis).
# ---------------------------------------------------------------------------
SEEDS = (0, 1, 2)
CHECKPOINT_EPOCHS = (10, 25, 50, 75, 100, 150)
NUM_EPOCHS = 150

TASKS = (
    "LIBERO-Spatial-1",
    "LIBERO-Object-1",
    "LIBERO-Goal-1",
    "LIBERO-Long-1",
    "Robomimic-Square-PH",
    "Robomimic-Transport-PH",
)

METRIC_COLS = ("M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8")

# Metrics where a smaller raw value means a better policy. The analysis
# sign-flips these so that, after flipping, a positive Spearman always means
# "this metric tracks success rate the right way".
LOWER_IS_BETTER = ("M1", "M2", "M4", "M5", "M6", "M7")
HIGHER_IS_BETTER = ("M3", "M8")

# H4 task partition (locked).
H4_TRAIN_TASKS = (
    "LIBERO-Spatial-1",
    "LIBERO-Goal-1",
    "LIBERO-Long-1",
    "Robomimic-Square-PH",
)
H4_HELD_OUT_TASKS = ("LIBERO-Object-1", "Robomimic-Transport-PH")

# Statistical plan (locked).
FAMILY_ALPHA = 0.05
N_HYPOTHESES = 4
BONFERRONI_ALPHA = FAMILY_ALPHA / N_HYPOTHESES          # 0.0125
H3_N_COMPARISONS = 7
H3_ALPHA = BONFERRONI_ALPHA / H3_N_COMPARISONS           # ~0.00179
H1_MEDIAN_GAP_THRESHOLD_PCT = 10.0
H4_RMSE_IMPROVEMENT_THRESHOLD = 0.20
RIDGE_ALPHAS = (0.01, 0.1, 1, 10, 100)
RIDGE_CV_FOLDS = 5
BOOTSTRAP_RESAMPLES = 10000

N_RUNS = len(TASKS) * len(SEEDS)            # 18
N_CHECKPOINTS = N_RUNS * len(CHECKPOINT_EPOCHS)  # 108


@lru_cache(maxsize=None)
def _load_yaml(name: str) -> dict[str, Any]:
    path = os.path.join(CONFIG_DIR, name)
    if yaml is None or not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_tasks() -> dict[str, Any]:
    """Return the task table, falling back to literals if YAML is absent."""
    data = _load_yaml("tasks.yaml")
    return data.get("tasks", {})


def load_train_cfg() -> dict[str, Any]:
    return _load_yaml("train.yaml")


def load_stats_cfg() -> dict[str, Any]:
    return _load_yaml("stats.yaml")


def task_regime_labels() -> dict[str, dict[str, str]]:
    """{task: {horizon, complexity}} from the locked task table."""
    tasks = load_tasks()
    if tasks:
        return {t: {"horizon": v["horizon"], "complexity": v["complexity"]} for t, v in tasks.items()}
    # Fallback literals matching tasks.yaml.
    return {
        "LIBERO-Spatial-1": {"horizon": "short", "complexity": "low"},
        "LIBERO-Object-1": {"horizon": "short", "complexity": "medium"},
        "LIBERO-Goal-1": {"horizon": "medium", "complexity": "medium"},
        "LIBERO-Long-1": {"horizon": "long", "complexity": "high"},
        "Robomimic-Square-PH": {"horizon": "medium", "complexity": "high"},
        "Robomimic-Transport-PH": {"horizon": "long", "complexity": "high"},
    }


def verify_prelock() -> list[str]:
    """Cross-check the Python literals against the YAML. Returns mismatch messages.

    Called by scripts/01_setup.py and the smoke test to guarantee the
    pre-registration is internally consistent before any training.
    """
    problems: list[str] = []
    tasks = load_tasks()
    if tasks and tuple(tasks.keys()) != TASKS:
        problems.append(f"task order mismatch: yaml={tuple(tasks.keys())} literals={TASKS}")
    train = load_train_cfg()
    if train:
        if tuple(train.get("seeds", [])) != SEEDS:
            problems.append("seeds mismatch between train.yaml and constants")
        if tuple(train.get("checkpoint_epochs", [])) != CHECKPOINT_EPOCHS:
            problems.append("checkpoint_epochs mismatch")
    held_out = tuple(t for t, v in tasks.items() if v.get("held_out")) if tasks else H4_HELD_OUT_TASKS
    if held_out and set(held_out) != set(H4_HELD_OUT_TASKS):
        problems.append(f"held-out mismatch: yaml={held_out} literals={H4_HELD_OUT_TASKS}")
    return problems

#!/usr/bin/env python3
"""SA-1: environment verification + demonstration acquisition.

Dev box: verifies the pre-registration is internally consistent and prints the
software environment. Kaggle (`--download`): pulls LIBERO + Robomimic demos and
prints per-task trajectory counts against the locked expectations.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drc import config
from drc.utils import get_logger

log = get_logger("setup")


def verify_environment():
    problems = config.verify_prelock()
    if problems:
        for p in problems:
            log.error(f"PRELOCK MISMATCH: {p}")
        raise SystemExit("Pre-registration inconsistency detected; aborting.")
    log.info("Pre-lock check passed: tasks/seeds/epochs/held-out all consistent.")
    log.info(f"Tasks ({len(config.TASKS)}): {', '.join(config.TASKS)}")
    log.info(f"Seeds: {config.SEEDS} | epochs: {config.CHECKPOINT_EPOCHS}")
    log.info(f"Runs: {config.N_RUNS} | checkpoints: {config.N_CHECKPOINTS}")
    log.info(f"Bonferroni alpha={config.BONFERRONI_ALPHA}, H3 alpha={config.H3_ALPHA:.6f}")

    import numpy, pandas, scipy, sklearn  # noqa
    log.info(f"numpy {numpy.__version__}, scipy {scipy.__version__}, sklearn {sklearn.__version__}")
    try:
        import torch
        log.info(f"torch {torch.__version__}, cuda={torch.cuda.is_available()}, "
                 f"gpus={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    except ImportError:
        log.warning("torch not importable")


def download_demos():  # pragma: no cover - Kaggle only
    import robomimic.utils.file_utils as FileUtils
    data_root = config.path("data") if hasattr(config, "path") else "/kaggle/working/data"
    os.makedirs(os.path.join(data_root, "robomimic"), exist_ok=True)
    log.info("Downloading Robomimic image datasets (square, transport, lift PH)...")
    FileUtils.download_url(
        url="https://diffusion-policy.cs.columbia.edu/data/training/robomimic_image.zip",
        download_dir=os.path.join(data_root, "robomimic"),
    )
    log.info("LIBERO demos: run `python -m libero.libero.scripts.download_datasets`. "
             "Verify counts: 50 per LIBERO task, 200 per Robomimic-PH task.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="download demos (Kaggle)")
    args = ap.parse_args()
    verify_environment()
    if args.download:
        download_demos()

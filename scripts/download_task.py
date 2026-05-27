#!/usr/bin/env python3
"""Download (or clean) the demonstrations for ONE task only.

Kaggle has ~73 GB; the bulk LIBERO/Robomimic downloaders pull the entire benchmark (100s of GB).
This fetches just the data a single task needs, so the session can download -> train -> evaluate ->
delete per task and never exceed a few GB at once.

  python scripts/download_task.py LIBERO-Spatial-1            # download just this task's suite
  python scripts/download_task.py Robomimic-Square-PH --clean # delete it afterwards
"""
import argparse
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drc import config
from drc.utils import get_logger

log = get_logger("download")

ROBOMIMIC_DATA = os.environ.get("ROBOMIMIC_DATA", "/kaggle/working/data/robomimic")


def libero_suite_dir(suite):
    # LIBERO downloads into the package "datasets" dir, one subfolder per suite.
    try:
        from libero.libero import get_libero_path
        return os.path.join(get_libero_path("datasets"), suite)
    except Exception:
        return None


def main(task, clean):
    cfg = config.load_tasks()[task]
    if cfg["suite"] == "libero":
        suite = cfg["benchmark"]            # libero_spatial / libero_object / libero_goal / libero_10
        if clean:
            d = libero_suite_dir(suite)
            if d and os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True); log.info(f"removed {d}")
            return
        # download ONLY this suite (not all of LIBERO)
        cmd = [sys.executable, "-m", "libero.libero.scripts.download_datasets", "--datasets", suite]
        log.info("download: " + " ".join(cmd))
        subprocess.run(cmd, stdin=subprocess.DEVNULL, check=False)
    else:
        bench = cfg["benchmark"]             # lift / can / square / transport
        task_dir = os.path.join(ROBOMIMIC_DATA, bench)
        if clean:
            if os.path.isdir(task_dir):
                shutil.rmtree(task_dir, ignore_errors=True); log.info(f"removed {task_dir}")
            return
        # download ONLY this task's PH image dataset (not the full robomimic_image.zip)
        cmd = [sys.executable, "-m", "robomimic.scripts.download_datasets",
               "--tasks", bench, "--dataset_types", cfg.get("dataset", "ph"),
               "--hdf5_types", "image", "--download_dir", ROBOMIMIC_DATA]
        log.info("download: " + " ".join(cmd))
        subprocess.run(cmd, stdin=subprocess.DEVNULL, check=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=config.TASKS)
    ap.add_argument("--clean", action="store_true", help="delete this task's data instead of downloading")
    args = ap.parse_args()
    main(args.task, args.clean)

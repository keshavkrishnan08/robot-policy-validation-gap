#!/usr/bin/env python3
"""SA-2 driver: train one (task, seed) or the full 18-run sweep.

  python scripts/02_train.py --task LIBERO-Spatial-1 --seed 0
  python scripts/02_train.py --all                       # full sweep
  python scripts/02_train.py --all --synthetic --epochs 2 --device cpu   # smoke
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drc import config
from drc.train import train_run
from drc.utils import ckpt_path, get_logger
from scripts.providers import dataset_provider

log = get_logger("02_train")


def run_one(task, seed, arch, args):
    provider = dataset_provider(task, synthetic=args.synthetic)
    backbone = "smallcnn" if args.synthetic else args.backbone
    ckpt_epochs = [int(e) for e in args.checkpoint_epochs.split(",")] if args.checkpoint_epochs else None
    # Resume: if every checkpoint for this run already exists, skip it. Lets a re-run after a crash
    # (or a fresh 12h Kaggle session) pick up where it left off instead of retraining from scratch.
    want_epochs = ckpt_epochs or list(config.CHECKPOINT_EPOCHS)
    if not args.force and all(os.path.exists(ckpt_path(task, seed, e, arch)) for e in want_epochs):
        log.info(f"skip {task} s{seed} [{arch}] — all {len(want_epochs)} checkpoints already present")
        return
    train_run(
        task=task,
        seed=seed,
        arch=arch,
        dataset_provider=provider,
        backbone=backbone,
        device=args.device,
        epochs=args.epochs,
        checkpoint_epochs=ckpt_epochs,
        val_K=args.val_K,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=config.TASKS)
    ap.add_argument("--seed", type=int, choices=config.SEEDS)
    ap.add_argument("--arch", choices=config.ARCHITECTURES, default="diffusion")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--all_archs", action="store_true", help="with --all, sweep both architectures")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--checkpoint_epochs", default=None, help="comma list, overrides locked epochs")
    ap.add_argument("--force", action="store_true", help="retrain even if checkpoints already exist")
    ap.add_argument("--val_K", type=int, default=10)
    args = ap.parse_args()

    if args.all:
        archs = config.ARCHITECTURES if args.all_archs else (args.arch,)
        for arch in archs:
            for task in config.TASKS:
                for seed in config.SEEDS:
                    log.info(f"=== train {task} seed {seed} [{arch}] ===")
                    run_one(task, seed, arch, args)
    else:
        assert args.task is not None and args.seed is not None, "give --task and --seed, or --all"
        run_one(args.task, args.seed, args.arch, args)

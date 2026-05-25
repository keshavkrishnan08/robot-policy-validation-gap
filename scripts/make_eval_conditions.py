#!/usr/bin/env python3
"""Pre-fix and persist the 20 rollout initial conditions per task (PRD 8.3).

Run once before SA-4. The conditions are committed so every checkpoint of a
task is evaluated on the identical, paired initial conditions.
"""
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drc import config
from drc.utils import ensure_dir, get_logger, path

log = get_logger("eval_conditions")


def main(synthetic: bool, n: int):
    for i, task in enumerate(config.TASKS):
        if synthetic:
            from drc.data.synthetic import make_eval_conditions

            conds = make_eval_conditions(n, seed=1000 + i)
        else:  # pragma: no cover - Kaggle: sample from the sim init distribution
            from scripts.providers import env_for

            env = env_for(task, synthetic=False)
            conds = [{"state": env.env.sim.get_state().flatten().tolist()} for _ in range(n)]
        out = path("eval_conditions", f"{task}.pkl")
        ensure_dir(os.path.dirname(out))
        with open(out, "wb") as f:
            pickle.dump(conds, f)
        log.info(f"{task}: wrote {len(conds)} eval conditions -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    main(args.synthetic, args.n)

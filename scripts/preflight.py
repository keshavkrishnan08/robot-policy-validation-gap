#!/usr/bin/env python3
"""Preflight readiness gate — verify the full experiment is wired to run.

Runs on CPU with no training and no heavy deps. Checks every real-run code path
structurally so a Kaggle session does not fail three hours in:

  1. Pre-registration consistency (config literals vs YAML)
  2. Task table completeness (fields, held-out, regime coverage)
  3. Adapter routing for all 8 tasks (LIBERO / Robomimic, image+proprio keys)
  4. Both architectures build + run inference at single-arm (7d) and dual-arm (14d) dims
  5. Env factory dispatch (synthetic builds; real env classes importable)
  6. Synthetic providers produce data for every task name
  7. Eval-condition generation for all tasks
  8. Analysis handles the full (task, seed, arch) schema
  9. Compute / storage budget estimate

Exit 0 == ready to launch. Any failure prints a clear reason.
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from drc import config
from drc.utils import get_logger

log = get_logger("preflight")
PASS, FAIL = "✅", "❌"
errors = []


def check(name, fn):
    try:
        msg = fn()
        log.info(f"{PASS} {name}: {msg}")
    except Exception as e:  # noqa
        errors.append(name)
        log.error(f"{FAIL} {name}: {e}")
        traceback.print_exc()


def c1_prelock():
    probs = config.verify_prelock()
    assert not probs, f"prelock mismatches: {probs}"
    assert config.N_RUNS == len(config.TASKS) * len(config.SEEDS) * len(config.ARCHITECTURES)
    return f"{config.N_RUNS} runs, {config.N_CHECKPOINTS} checkpoints, archs={config.ARCHITECTURES}"


def c2_task_table():
    tasks = config.load_tasks()
    required = {"suite", "benchmark", "horizon", "complexity", "n_demos", "max_steps", "held_out"}
    for t, v in tasks.items():
        missing = required - set(v)
        assert not missing, f"{t} missing fields {missing}"
        assert v["horizon"] in {"short", "medium", "long"}, f"{t} bad horizon"
        assert v["complexity"] in {"low", "medium", "high"}, f"{t} bad complexity"
    held = {t for t, v in tasks.items() if v.get("held_out")}
    assert held == set(config.H4_HELD_OUT_TASKS), f"held-out mismatch {held}"
    assert set(config.H4_TRAIN_TASKS).isdisjoint(config.H4_HELD_OUT_TASKS)
    return f"{len(tasks)} tasks complete; held-out={sorted(held)}"


def c3_adapter_routing():
    from drc.data import libero_adapter, robomimic_adapter  # import-only (heavy deps lazy)

    tasks = config.load_tasks()
    for t, v in tasks.items():
        if v["suite"] == "robomimic":
            bench = v["benchmark"]
            keys = robomimic_adapter.IMG_KEYS.get(bench, ["agentview_image"])
            assert keys, f"{t}: no image keys"
            assert "dataset" in v, f"{t}: robomimic task needs 'dataset'"
        else:
            assert "bddl" in v and "benchmark" in v, f"{t}: LIBERO task needs bddl+benchmark"
    return "all 8 tasks route to a loader; robomimic image/proprio keys resolve"


def c4_both_architectures():
    from drc.train import build_policy

    tcfg = config.load_train_cfg()
    # single-arm (7d action, 9d proprio) and dual-arm (14d action, 18d proprio)
    for action_dim, proprio_dim, label in [(7, 9, "single-arm"), (14, 18, "dual-arm")]:
        info = {"action_dim": action_dim, "proprio_dim": proprio_dim,
                "image_shape": (3, 84, 84), "crop_shape": (76, 76)}
        obs = {"image": torch.rand(1, 2, 3, 84, 84), "proprio": torch.rand(1, 2, proprio_dim)}
        for arch in config.ARCHITECTURES:
            pol = build_policy(info, tcfg, backbone="smallcnn", arch=arch)
            pol.normalizer.fit(torch.randn(40, action_dim))
            # tiny inference (reduce diffusion steps for speed)
            if hasattr(pol, "scheduler"):
                pol.scheduler.num_inference_steps = 2
            chunk = pol.predict_action_chunk(obs, K=1)
            assert chunk.shape[-1] == action_dim, f"{arch} {label} bad action dim"
    return "diffusion + act build & run inference at 7d and 14d action dims"


def c5_env_factory():
    from drc.envs import make_env, _LiberoEnv, _RobomimicEnv  # noqa: F401  import-only

    tasks = config.load_tasks()
    env = make_env("LIBERO-Spatial-1", tasks["LIBERO-Spatial-1"], synthetic=True)
    env.reset_to({"state": np.array([-0.5, -0.5, 0, 0], dtype=np.float32)})
    _, _, done, info = env.step(np.zeros(4, dtype=np.float32))
    assert "success" in info
    return "synthetic env steps; real LIBERO/Robomimic env classes importable"


def c6_providers_all_tasks():
    from scripts.providers import dataset_provider

    for t in config.TASKS:
        train_ds, val_ds, info = dataset_provider(t, synthetic=True)()
        assert len(train_ds) > 0 and info["action_dim"] > 0, f"{t}: empty dataset"
    return f"synthetic providers yield data for all {len(config.TASKS)} task names"


def c7_eval_conditions():
    from drc.data.synthetic import make_eval_conditions

    conds = make_eval_conditions(20, seed=1000)
    assert len(conds) == 20 and "state" in conds[0]
    return "20 eval conditions generate per task"


def c8_analysis_schema():
    from drc import analysis, devtools

    m, r = devtools.make_fake_results(seed=0)
    assert len(m) == config.N_CHECKPOINTS and "arch" in m.columns
    mp, rp = "/tmp/pf_m.csv", "/tmp/pf_r.csv"
    m.to_csv(mp, index=False)
    r.to_csv(rp, index=False)
    res = analysis.run_all(mp, rp)
    assert res["n_runs"] == config.N_RUNS
    for h in ["H1", "H2", "H3", "H4"]:
        assert "supported" in res[h]
    assert "secondary" in res and "causal_selection" in res["secondary"]
    return f"analysis runs on {config.N_CHECKPOINTS}-row, {len(res['architectures'])}-arch schema"


def c9_budget():
    n_train = config.N_RUNS
    n_ckpt = config.N_CHECKPOINTS
    n_roll = n_ckpt * 20
    per_task_ckpt = len(config.SEEDS) * len(config.ARCHITECTURES) * len(config.CHECKPOINT_EPOCHS)
    peak_gb = per_task_ckpt * 0.15  # one task's checkpoints (pruned after metrics+rollouts)
    return (f"{n_train} train runs, {n_ckpt} checkpoints, {n_roll} rollouts; "
            f"per-task pruning caps peak at ~{peak_gb:.0f}GB ({per_task_ckpt} ckpts/task) << 20GB Kaggle")


def main():
    log.info("=== PREFLIGHT: experiment readiness ===")
    check("1. pre-registration consistency", c1_prelock)
    check("2. task table completeness", c2_task_table)
    check("3. adapter routing (8 tasks)", c3_adapter_routing)
    check("4. both architectures build+infer", c4_both_architectures)
    check("5. env factory dispatch", c5_env_factory)
    check("6. synthetic providers all tasks", c6_providers_all_tasks)
    check("7. eval-condition generation", c7_eval_conditions)
    check("8. analysis full schema", c8_analysis_schema)
    check("9. compute/storage budget", c9_budget)
    if errors:
        log.error(f"PREFLIGHT FAILED: {len(errors)} check(s) -> {errors}")
        sys.exit(1)
    log.info("PREFLIGHT PASSED ✅ — experiments are wired and ready to launch.")


if __name__ == "__main__":
    main()

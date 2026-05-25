#!/usr/bin/env python3
"""Tier 3 driver: combine per-policy offline-metric JSONs and correlate with the
published real-robot success rates (SIMPLER comparison set).

Per-policy metrics are produced on Kaggle, one framework-env per policy family
(TF for RT-1, JAX for Octo, PyTorch for OpenVLA), each writing
external/<policy>_metrics.json. This script (CPU) combines them and runs the
correlate-with-real-success analysis against external/real_success.json.

real_success.json maps policy -> real-world success rate, populated from SIMPLER's
released paired sim-real evaluation data. (Left to the user to fill from the source;
do NOT fabricate the numbers.)
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drc.external_validation import correlate_with_real_success
from drc.utils import get_logger, load_json, path, save_json

log = get_logger("06_external")


def main(args):
    metric_by_policy = {}
    for f in sorted(glob.glob(path("external", "*_metrics.json"))):
        policy = os.path.basename(f).replace("_metrics.json", "")
        metric_by_policy[policy] = load_json(f)
    if not metric_by_policy:
        raise SystemExit("No external/<policy>_metrics.json files found. Run the per-policy "
                         "metric jobs on Kaggle first (see TIER3_FEASIBILITY.md).")

    real_path = args.real or path("external", "real_success.json")
    if not os.path.exists(real_path):
        raise SystemExit(f"Missing {real_path}: populate with per-policy real-world success "
                         "rates from SIMPLER's released data (do not fabricate).")
    real_success = load_json(real_path)

    result = correlate_with_real_success(metric_by_policy, real_success)
    out = path("external", "external_validation_result.json")
    save_json(result, out)
    log.info(f"policies: {result['policies']}")
    log.info(f"best offline metric: {result['best_metric']} (|pearson|={result['best_pearson']})")
    log.info(f"MSE baseline pearson (SIMPLER): {result['mse_baseline_pearson']}")
    log.info(f"any offline metric beats MSE baseline: {result['any_metric_beats_mse']}")
    print(json.dumps(result["per_metric"], indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=None, help="path to real_success.json")
    main(ap.parse_args())

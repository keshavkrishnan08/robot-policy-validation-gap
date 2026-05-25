#!/usr/bin/env python3
"""SA-5 driver: run H1-H4, generate the four figures, run the power report, and
write results/results.json. Prints which outcome-matrix row the data falls into.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drc import analysis, figures, power
from drc.utils import get_logger, path, save_json, sha256_file

log = get_logger("05_analysis")


def main(args):
    metrics_csv = args.metrics or path("results", "metrics.csv")
    rollouts_csv = args.rollouts or path("results", "rollouts.csv")
    out_json = args.out or path("results", "results.json")

    results = analysis.run_all(metrics_csv, rollouts_csv)
    if not args.no_power:
        results["power"] = power.report(n_sims=args.power_sims)

    df = analysis.load_merged(metrics_csv, rollouts_csv)
    results["figures"] = figures.generate_all(results, df, path("figures"))

    save_json(results, out_json)
    log.info(f"results -> {out_json}  (sha256 {sha256_file(out_json)[:16]})")

    for h in ["H1", "H2", "H3", "H4"]:
        log.info(f"{h} supported={results[h]['supported']}")
    om = results["outcome_matrix"]
    log.info(f"OUTCOME: [{om['row']}] {om['title']}")
    log.info(f"  angle: {om['angle']}")
    print(json.dumps({k: results[k]["supported"] for k in ["H1", "H2", "H3", "H4"]}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--rollouts", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no_power", action="store_true")
    ap.add_argument("--power_sims", type=int, default=10000)
    main(ap.parse_args())

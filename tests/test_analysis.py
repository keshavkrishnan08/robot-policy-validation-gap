import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from drc import analysis, config, devtools


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    d = tmp_path_factory.mktemp("res")
    m, r = devtools.make_fake_results(seed=1)
    mp, rp = str(d / "m.csv"), str(d / "r.csv")
    m.to_csv(mp, index=False)
    r.to_csv(rp, index=False)
    return mp, rp


def test_run_all_structure(fake):
    res = analysis.run_all(*fake)
    assert res["n_observations"] == config.N_CHECKPOINTS
    assert res["n_runs"] == config.N_RUNS
    for h in ["H1", "H2", "H3", "H4"]:
        assert isinstance(res[h]["supported"], bool)
    assert "title" in res["outcome_matrix"]


def test_gap_nonnegative_by_construction(fake):
    df = analysis.load_merged(*fake)
    gaps = analysis.compute_gaps(df)
    # rollbest is the max success, vlbest is a specific checkpoint -> gap >= 0 always.
    assert (gaps["gap_pct"] >= -1e-9).all()


def test_signed_spearman_flips_lower_is_better():
    # A metric that decreases as success increases should yield positive signed rho
    # when it is in LOWER_IS_BETTER.
    success = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    metric = 1.0 - success  # perfectly anti-correlated
    rho_lower = analysis._signed_spearman(metric, success, "M1")  # M1 lower-is-better
    rho_higher = analysis._signed_spearman(metric, success, "M3")  # higher-is-better
    assert rho_lower > 0.9
    assert rho_higher < -0.9


def test_signed_spearman_handles_constant():
    success = np.ones(6)
    metric = np.arange(6.0)
    assert analysis._signed_spearman(metric, success, "M1") == 0.0


def test_classify_outcome_all_rows():
    # key passed as (H1, H2, H3, H4)
    assert "Adequate" in analysis.classify_outcome(False, True, True, True)["title"]  # H1=No
    assert "Offline Predictor" in analysis.classify_outcome(True, True, True, True)["title"]
    assert "Task-Specific" in analysis.classify_outcome(True, True, True, False)["title"]
    assert "Horizon Wall" in analysis.classify_outcome(True, True, False, False)["title"]
    assert "Considered Harmful" in analysis.classify_outcome(True, False, True, True)["title"]
    assert "Impossibility" in analysis.classify_outcome(True, False, False, False)["title"]
    # a non-tabulated combo falls back gracefully
    fb = analysis.classify_outcome(True, True, False, True)
    assert "title" in fb and "angle" in fb


def test_bca_ci_constant_returns_point():
    lo, hi = analysis.bca_ci(np.array([5.0, 5.0, 5.0]), np.median)
    assert lo == hi == 5.0


def test_h4_reports_loto(fake):
    df = analysis.load_merged(*fake)
    h4 = analysis.h4_analysis(df)
    assert set(h4["loto"].keys()) == set(config.TASKS)
    assert h4["composite_rmse"] >= 0

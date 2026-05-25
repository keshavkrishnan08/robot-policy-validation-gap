import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from drc import external_validation as ev


class DummyAdapter:
    """A toy policy: predicts expert action plus a fixed bias; samples add noise."""

    name = "dummy"

    def __init__(self, bias=0.0, noise=0.1, seed=0):
        self.bias = bias
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def predict(self, obs):
        return obs["expert"] + self.bias

    def sample(self, obs):
        return obs["expert"] + self.bias + self.rng.normal(0, self.noise, size=len(obs["expert"]))


def _demos(n=3, T=5, dim=4):
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        acts = rng.normal(size=(T, dim)).astype(np.float32)
        out.append({
            "obs_seq": [{"expert": a} for a in acts],
            "actions": acts,
            "initial_state": np.zeros(4, dtype=np.float32),
            "final_eef_pose": np.zeros(3, dtype=np.float32),
        })
    return out


def test_compute_external_metrics_finite():
    demos = _demos()
    m = ev.compute_external_metrics(DummyAdapter(bias=0.2), demos, env=None, n_sample=4)
    assert np.isfinite(m["M1"]) and m["M1"] > 0
    assert np.isfinite(m["M3"]) and np.isfinite(m["M8"])
    assert np.isnan(m["M5"])  # no env supplied


def test_m1_increases_with_bias():
    demos = _demos()
    low = ev.compute_external_metrics(DummyAdapter(bias=0.05), demos, n_sample=2)["M1"]
    high = ev.compute_external_metrics(DummyAdapter(bias=0.8), demos, n_sample=2)["M1"]
    assert high > low  # worse policy -> larger action error


def test_correlate_with_real_success_detects_predictive_metric():
    # 5 policies; M1 (lower better) perfectly anti-correlates with real success.
    real = {f"p{i}": sr for i, sr in enumerate([0.1, 0.3, 0.5, 0.7, 0.9])}
    metric_by_policy = {
        f"p{i}": {"M1": 1.0 - sr, "M3": 0.0, "M5": float("nan"), "M8": sr}
        for i, sr in enumerate([0.1, 0.3, 0.5, 0.7, 0.9])
    }
    res = ev.correlate_with_real_success(metric_by_policy, real)
    assert res["n_policies"] == 5
    # M1 lower-is-better and anti-correlated with success -> signed pearson ~ +1
    assert res["per_metric"]["M1"]["pearson"] > 0.95
    assert res["per_metric"]["M1"]["beats_mse_baseline"] is True
    assert res["any_metric_beats_mse"] is True


def test_correlate_handles_insufficient_n():
    real = {"a": 0.5, "b": 0.6}
    mbp = {"a": {"M1": 0.1, "M3": 0, "M5": float("nan"), "M8": 0},
           "b": {"M1": 0.2, "M3": 0, "M5": float("nan"), "M8": 0}}
    res = ev.correlate_with_real_success(mbp, real)
    assert res["per_metric"]["M1"]["spearman"] is None  # n<3 -> not computed

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drc import config


def test_prelock_consistent():
    assert config.verify_prelock() == []


def test_counts():
    assert len(config.TASKS) == 8
    assert len(config.ARCHITECTURES) == 2
    assert config.N_RUNS == 8 * 3 * 2   # tasks x seeds x architectures = 48
    assert config.N_CHECKPOINTS == config.N_RUNS * 6  # 288
    assert len(config.METRIC_COLS) == 8


def test_alphas():
    assert abs(config.BONFERRONI_ALPHA - 0.0125) < 1e-12
    assert abs(config.H3_ALPHA - 0.0125 / 7) < 1e-12


def test_held_out_partition_disjoint():
    assert set(config.H4_TRAIN_TASKS).isdisjoint(config.H4_HELD_OUT_TASKS)
    assert set(config.H4_TRAIN_TASKS) | set(config.H4_HELD_OUT_TASKS) == set(config.TASKS)


def test_regime_labels_cover_all_tasks():
    labels = config.task_regime_labels()
    assert set(labels) == set(config.TASKS)
    for v in labels.values():
        assert v["horizon"] in {"short", "medium", "long"}
        assert v["complexity"] in {"low", "medium", "high"}

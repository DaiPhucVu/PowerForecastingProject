"""
Tests for src/explain_lstm.py that need no training: the input groups and the permutation logic.

Permutation importance is checked on a model whose answer is known: a linear function of one input. Shuffling
an input it ignores must cost nothing, and shuffling the one it uses must cost something.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so "src.*" imports work like python -m

pytest.importorskip("keras")
from src import explain_lstm as ex
from src.models import lstm


def feature_names():
    cal = lstm.CALENDAR_COLS
    return lstm.MEASURES + cal + [f"next_{c}" for c in cal]


def test_groups_cover_every_lstm_feature_once():
    groups = ex.group_index(feature_names())
    used = sorted(i for cols in groups.values() for i in cols)
    assert used == list(range(len(feature_names())))
    assert set(groups) == set(ex.GROUPS)


def test_groups_reject_an_unknown_feature():
    with pytest.raises((AssertionError, KeyError)):
        ex.group_index(feature_names() + ["temperature"])


def test_permutation_ignored_input_costs_nothing_used_input_costs_something():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 14, 3)).astype("float32")
    y = 3 * X[:, -1, 0]                               # depends only on input 0 on the last day

    def predict(Xb):
        return 3 * Xb[:, -1, 0]

    res, base = ex.permutation_importance(predict, X, y, {"used": [0], "ignored": [1]}, repeats=3)
    rise = dict(zip(res["input"], res["mae_increase_kwh"]))
    assert base == pytest.approx(0.0)
    assert rise["ignored"] == pytest.approx(0.0)
    assert rise["used"] > 1.0


def test_permutation_by_step_only_touches_that_step():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 14, 2)).astype("float32")
    y = X[:, -1, 0]                                    # only the last window day matters

    def predict(Xb):
        return Xb[:, -1, 0]

    early, _ = ex.permutation_importance(predict, X, y, {"t-14": [0]}, steps=[0], repeats=3)
    last, _ = ex.permutation_importance(predict, X, y, {"t-1": [0]}, steps=[13], repeats=3)
    assert early.loc[0, "mae_increase_kwh"] == pytest.approx(0.0)
    assert last.loc[0, "mae_increase_kwh"] > 0.5


def test_permutation_leaves_the_input_untouched():
    X = np.arange(2 * 3 * 2, dtype="float32").reshape(2, 3, 2)
    before = X.copy()
    ex.permutation_importance(lambda Xb: Xb[:, -1, 0], X, X[:, -1, 0], {"a": [0]}, repeats=2)
    np.testing.assert_array_equal(X, before)


def test_xgb_features_map_to_lstm_groups():
    assert ex.xgb_group("other_wh_lag1") == "other_wh"
    assert ex.xgb_group("sub3_wh_lag7") == "sub3_wh"
    assert ex.xgb_group("lag_7") == "kwh"
    assert ex.xgb_group("roll_mean_28") == "kwh"
    assert ex.xgb_group("dow_sin") == "day_of_week"
    assert ex.xgb_group("is_weekend") == "day_of_week"
    assert ex.xgb_group("doy_cos") == "time_of_year"


def test_xgb_feature_groups_all_exist_in_the_lstm_groups():
    xgb = ["lag_1", "lag_7", "roll_mean_7", "roll_std_28", "dow_sin", "doy_cos", "month", "is_weekend",
           "other_wh_lag1", "sub1_wh_lag7", "voltage_lag1", "intensity_lag7", "reactive_kvarh_lag1"]
    assert {ex.xgb_group(f) for f in xgb} <= set(ex.GROUPS)

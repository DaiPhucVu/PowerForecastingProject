"""
Tests for splitting, metrics and the monthly rollup.

QA plan objectives: "Confirm fairness for all model comparison" and
"Confirm the accuracy result holds when the test window moves".
"""
import numpy as np
import pandas as pd
import pytest

import forecast_pipeline as fp
import monthly_report as mr
import rolling_validation as rv


# ------------------------------------------------------------------- splitting

def test_split_is_chronological(daily_frame):
    """Every training day must precede every validation day, and every
    validation day every test day. A shuffled split would still produce
    good-looking scores, which is exactly why this is tested."""
    feat = fp.make_features(daily_frame)
    train, val, test = fp.chrono_split(feat)

    assert train.date.max() < val.date.min()
    assert val.date.max() < test.date.min()


def test_split_loses_no_rows(daily_frame):
    feat = fp.make_features(daily_frame)
    train, val, test = fp.chrono_split(feat)
    assert len(train) + len(val) + len(test) == len(feat)


def test_split_proportions(daily_frame):
    """20% test, then 20% of the remainder as validation."""
    feat = fp.make_features(daily_frame)
    train, val, test = fp.chrono_split(feat)
    n = len(feat)

    assert len(test) == pytest.approx(0.20 * n, abs=1)
    assert len(val) == pytest.approx(0.20 * 0.80 * n, abs=1)


# --------------------------------------------------------------------- metrics

def test_mape_ignores_zero_actuals():
    """MAPE is undefined when the actual is zero. Those rows are masked out
    rather than producing inf, which would silently poison the mean."""
    y = np.array([0.0, 10.0, 20.0])
    p = np.array([5.0, 11.0, 18.0])

    got = fp.mape(y, p)
    expected = np.mean([1 / 10, 2 / 20]) * 100      # the zero row is excluded

    assert got == pytest.approx(expected)
    assert np.isfinite(got)


def test_mape_is_zero_for_perfect_predictions():
    y = np.array([10.0, 20.0, 30.0])
    assert fp.mape(y, y) == pytest.approx(0.0)


def test_perfect_prediction_scores_zero_on_everything():
    y = np.array([10.0, 20.0, 30.0])
    s = fp.score("perfect", y, y)
    assert s["MAE"] == pytest.approx(0.0)
    assert s["RMSE"] == pytest.approx(0.0)
    assert s["MAPE_%"] == pytest.approx(0.0)


def test_rmse_penalises_large_errors_more_than_mae():
    """One big miss should move RMSE further than MAE. This is why both are
    reported rather than just one."""
    y = np.array([10.0, 10.0, 10.0, 10.0])
    p = np.array([10.0, 10.0, 10.0, 18.0])
    s = fp.score("spiky", y, p)
    assert s["RMSE"] > s["MAE"]


# -------------------------------------------------------------- monthly rollup

def test_monthly_totals_sum_the_daily_values():
    dates = pd.date_range("2010-01-01", periods=62, freq="D")
    actual = np.full(62, 10.0)
    pred = np.full(62, 11.0)

    out = mr.to_monthly(dates, actual, pred)

    jan = out[out.month == "2010-01"].iloc[0]
    assert jan["days"] == 31
    assert jan["actual_kwh"] == pytest.approx(310.0)
    assert jan["predicted_kwh"] == pytest.approx(341.0)
    assert jan["error_kwh"] == pytest.approx(31.0)
    assert jan["error_pct"] == pytest.approx(10.0)


def test_partial_months_are_flagged_incomplete():
    """A test window starting mid-month produces a stub month. It must be
    marked so it never enters the averages as if it were a real month."""
    dates = pd.date_range("2010-01-29", periods=34, freq="D")   # 3 + 28 + 3
    out = mr.to_monthly(dates, np.full(34, 10.0), np.full(34, 10.0))

    jan = out[out.month == "2010-01"].iloc[0]
    assert jan["days"] == 3
    assert not jan["complete"]

    feb = out[out.month == "2010-02"].iloc[0]
    assert feb["complete"]


def test_min_days_in_month_threshold():
    assert mr.MIN_DAYS_IN_MONTH == 28


# ------------------------------------------------------- rolling-origin folds

def test_folds_never_train_on_the_future():
    """The core property of forward chaining: each fold's training block ends
    exactly where its test block starts."""
    for train_end, test_end in rv.folds(n=1000, k=5):
        assert train_end < test_end


def test_folds_expand_and_do_not_overlap():
    bounds = list(rv.folds(n=1000, k=5))

    train_ends = [t for t, _ in bounds]
    assert train_ends == sorted(train_ends)          # training grows
    for (_, prev_test_end), (next_train_end, _) in zip(bounds, bounds[1:]):
        assert prev_test_end == next_train_end       # blocks are contiguous


def test_folds_cover_the_tail_of_the_series():
    """The last fold must reach the final row, so no data is quietly unused."""
    bounds = list(rv.folds(n=1000, k=5))
    assert bounds[-1][1] == 1000


def test_first_fold_has_enough_history():
    bounds = list(rv.folds(n=1000, k=5))
    assert bounds[0][0] >= 400          # 40% minimum training block


def test_too_many_folds_is_refused():
    """Twenty folds on a short series would leave test blocks of a few days,
    which produces meaningless scores. Better to fail loudly."""
    with pytest.raises(SystemExit):
        list(rv.folds(n=300, k=20))

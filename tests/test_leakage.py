"""
Leakage tests (QA plan objective: "Confirm no leakage happens").

The pipeline already asserts this at runtime. These tests add the part a
runtime assertion cannot do: they deliberately break the rule and check the
assertion actually fires. An assertion nobody has ever seen fail is not
evidence of anything.
"""
import pandas as pd
import pytest

import forecast_pipeline as fp


def test_lag_1_equals_previous_day(daily_frame):
    feat = fp.make_features(daily_frame)
    prev = daily_frame[["date", "kwh"]].copy()
    prev["date"] = prev["date"] + pd.Timedelta(days=1)
    chk = feat[["date", "lag_1"]].merge(prev, on="date")

    assert len(chk) > 0
    assert (chk.lag_1 - chk.kwh).abs().max() < 1e-9


def test_lag_7_equals_a_week_earlier(daily_frame):
    feat = fp.make_features(daily_frame)
    prev = daily_frame[["date", "kwh"]].copy()
    prev["date"] = prev["date"] + pd.Timedelta(days=7)
    chk = feat[["date", "lag_7"]].merge(prev, on="date")

    assert (chk.lag_7 - chk.kwh).abs().max() < 1e-9


def test_same_day_columns_never_appear_raw(daily_frame):
    """Voltage, the sub-meters and the residual are only known once the day is
    over. They may appear as lags; the bare column names must not."""
    feat = fp.make_features(daily_frame)
    assert not set(fp.SAME_DAY_ONLY) & set(feat.columns)


def test_same_day_columns_are_available_as_lags(daily_frame):
    """The flip side: dropping them entirely would throw away real signal, so
    confirm the lagged versions survived."""
    feat = fp.make_features(daily_frame)
    for c in fp.SAME_DAY_ONLY:
        assert f"{c}_lag1" in feat.columns
        assert f"{c}_lag7" in feat.columns


def test_rolling_statistics_exclude_today(daily_frame):
    """roll_mean_7 must be built from the seven days BEFORE the target day.
    Including today would put the answer inside the feature."""
    feat = fp.make_features(daily_frame)
    row = feat.iloc[50]
    window = daily_frame.loc[daily_frame.date < row["date"], "kwh"].tail(7)

    assert row["roll_mean_7"] == pytest.approx(window.mean())


def test_assertion_fires_when_a_same_day_column_is_smuggled_in(daily_frame):
    """Tamper test. Add voltage to the feature table unlagged and confirm
    assert_no_leakage rejects it."""
    feat = fp.make_features(daily_frame)
    feat["voltage"] = 240.0

    with pytest.raises(AssertionError):
        fp.assert_no_leakage(daily_frame, feat)


def test_assertion_fires_when_lags_are_shifted_the_wrong_way(daily_frame):
    """Tamper test. Shift the target backwards instead of forwards - the
    classic off-by-one that turns a forecast into a lookup of tomorrow."""
    feat = fp.make_features(daily_frame)
    feat["lag_1"] = daily_frame["kwh"].shift(-1).iloc[len(daily_frame) - len(feat):].values

    with pytest.raises(AssertionError):
        fp.assert_no_leakage(daily_frame, feat)


def test_clean_feature_table_passes(daily_frame):
    """And the honest case must not raise."""
    feat = fp.make_features(daily_frame)
    fp.assert_no_leakage(daily_frame, feat)      # no exception = pass


def test_no_nulls_survive_feature_building(daily_frame):
    """Lags and rolling windows create nulls at the start of the series.
    They are dropped, not filled, so nothing should remain."""
    feat = fp.make_features(daily_frame)
    assert feat.notna().all().all()
    assert len(feat) < len(daily_frame)     # the warm-up rows were removed

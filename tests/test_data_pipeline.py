"""
Unit and regression tests for the raw-to-daily stage (feature F6 in the QA plan).

Objective: "Confirm the daily table can be rebuilt from raw without drift."
"""
import json

import pandas as pd
import pytest

import build_daily
from conftest import DATA, MANIFEST, needs_real_data


# --------------------------------------------------------------- unit tests

def test_kwh_conversion_divides_by_sixty(minute_frame):
    """Global_active_power is kW sampled per minute, so a day's energy is the
    sum of the readings divided by 60. Getting this wrong inflates daily
    consumption roughly 60-fold, which is the worst bug this project could
    ship, so it is checked against a hand-computed value."""
    kept, _ = build_daily.to_daily(minute_frame)

    day1 = kept.loc[kept.date == pd.Timestamp("2007-01-01")].iloc[0]
    assert day1["kwh"] == pytest.approx(1440 * 1.2 / 60)   # 28.8
    assert day1["kwh"] == pytest.approx(28.8)


def test_partial_day_is_dropped(minute_frame):
    """A day with 100 of 1440 minutes looks like a very low consumption day.
    It must be dropped, not kept and not interpolated."""
    kept, dropped = build_daily.to_daily(minute_frame)

    assert len(kept) == 1
    assert kept.date.iloc[0] == pd.Timestamp("2007-01-01")
    assert pd.Timestamp("2007-01-02") in dropped.index


def test_coverage_threshold_is_ninety_percent():
    """The rule is documented in the data dictionary and the report, so it is
    pinned here. Changing it should break a test, not slip through."""
    assert build_daily.MIN_COVERAGE == 0.90
    assert build_daily.MINUTES_PER_DAY == 1440


def test_day_just_above_threshold_is_kept():
    """1296 minutes is exactly 90%, so it must survive."""
    ts = pd.date_range("2007-03-01 00:00:00", periods=1296, freq="min")
    df = pd.DataFrame({"timestamp": ts, "Global_active_power": 1.0,
                       "Global_reactive_power": 0.0, "Voltage": 240.0,
                       "Global_intensity": 4.0, "Sub_metering_1": 0.0,
                       "Sub_metering_2": 0.0, "Sub_metering_3": 0.0})
    kept, dropped = build_daily.to_daily(df)
    assert len(kept) == 1 and len(dropped) == 0


def test_other_wh_is_a_residual(minute_frame):
    """other_wh is total active energy minus the three sub-meters. It is a
    residual, not a metered circuit, and the report says so - this checks the
    arithmetic matches that description."""
    kept, _ = build_daily.to_daily(minute_frame)
    r = kept.iloc[0]
    assert r["other_wh"] == pytest.approx(
        r["kwh"] * 1000 - r["sub1_wh"] - r["sub2_wh"] - r["sub3_wh"])


def test_helper_columns_do_not_reach_the_output(minute_frame):
    """minutes_seen and coverage are working columns. If they leaked into the
    daily table they would become features, and coverage is not something a
    forecast can know in advance."""
    kept, _ = build_daily.to_daily(minute_frame)
    assert "minutes_seen" not in kept.columns
    assert "coverage" not in kept.columns


# ------------------------------------------------- regression against manifest

@needs_real_data
def test_daily_table_matches_manifest():
    """The regression test named in the Week 8 QA plan: the committed manifest
    and the built table must agree on row count and date range. If a second
    member rebuilds and this fails, the cleaning rules have drifted."""
    daily = pd.read_csv(DATA, parse_dates=["date"])
    man = json.loads(MANIFEST.read_text())

    assert len(daily) == man["daily_rows"]
    assert str(daily.date.min().date()) == man["date_from"]
    assert str(daily.date.max().date()) == man["date_to"]
    assert man["min_coverage_rule"] == build_daily.MIN_COVERAGE


@needs_real_data
def test_daily_table_is_sane():
    """Order, uniqueness, no nulls in the target, and a plausible magnitude.
    A single household averaging hundreds of kWh a day would mean the /60
    conversion had been lost again."""
    daily = pd.read_csv(DATA, parse_dates=["date"])

    assert daily.date.is_monotonic_increasing
    assert not daily.date.duplicated().any()
    assert daily.kwh.notna().all()
    assert (daily.kwh > 0).all()
    assert 5 < daily.kwh.mean() < 100, \
        f"mean of {daily.kwh.mean():.1f} kWh/day is not a plausible household"

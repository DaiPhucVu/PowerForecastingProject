"""
Shared fixtures. Lets the tests import from src/ without installing anything.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Where the real generated files live, if they have been built on this machine.
DATA = ROOT / "data" / "household_daily.csv"
MANIFEST = ROOT / "data" / "manifest.json"


@pytest.fixture
def minute_frame():
    """Two days of minute-level readings in the raw file's shape.

    Day 1 is complete (1440 minutes). Day 2 has only 100 minutes, so the
    coverage rule must drop it. Global_active_power is a constant 1.2 kW,
    which makes the kWh conversion checkable by hand:
    1440 minutes x 1.2 kW / 60 = 28.8 kWh.
    """
    full = pd.date_range("2007-01-01 00:00:00", periods=1440, freq="min")
    part = pd.date_range("2007-01-02 00:00:00", periods=100, freq="min")
    ts = full.append(part)

    return pd.DataFrame({
        "timestamp": ts,
        "Global_active_power": 1.2,
        "Global_reactive_power": 0.1,
        "Voltage": 240.0,
        "Global_intensity": 5.0,
        "Sub_metering_1": 1.0,
        "Sub_metering_2": 2.0,
        "Sub_metering_3": 3.0,
    })


@pytest.fixture
def daily_frame():
    """A daily table with the real schema, long enough to build features from.

    Values are synthetic. Nothing here is a project result - these tests check
    that the code behaves correctly, not what the household consumed.
    """
    rng = np.random.default_rng(0)
    n = 400
    date = pd.date_range("2007-01-01", periods=n, freq="D")
    kwh = 26 + 6 * np.sin(2 * np.pi * date.dayofyear / 365.25) \
            + rng.normal(0, 3, n)
    kwh = np.clip(kwh, 1, None)

    df = pd.DataFrame({
        "date": date,
        "kwh": kwh,
        "reactive_kvarh": kwh * 0.08,
        "voltage": 240 + rng.normal(0, 2, n),
        "intensity": kwh / 5,
        "sub1_wh": rng.uniform(0, 2000, n),
        "sub2_wh": rng.uniform(0, 3000, n),
        "sub3_wh": rng.uniform(0, 9000, n),
    })
    df["other_wh"] = df.kwh * 1000 - df.sub1_wh - df.sub2_wh - df.sub3_wh
    return df


needs_real_data = pytest.mark.skipif(
    not DATA.exists(),
    reason="data/household_daily.csv not built on this machine - "
           "run src/build_daily.py first")

"""Turn the raw files in data/ into the daily table.

Run from the repo root: python -m src.build_daily
Reads  data/household_power_consumption.txt  (UCI, one row per minute)
Writes data/household_daily.csv              (one row per day, see data_dictionary.md)

Nothing here uses a later day to fill an earlier one:
- a day's energy comes only from that day's minutes,
- a day with too few readings is filled from the same weekday one week earlier.
"""

from pathlib import Path

import pandas as pd

from src.config import SEED, set_global_seed

RAW_PATH = Path("data/household_power_consumption.txt")
DAILY_PATH = Path("data/household_daily.csv")

MIN_COVERAGE = 0.8   # a day needs readings for at least 80% of its minutes
MINUTES_PER_DAY = 1440


def load_minutes(path: Path = RAW_PATH) -> pd.Series:
    """Global active power (kW), one value per minute, NaN where the meter had no reading."""
    raw = pd.read_csv(path, sep=";", na_values="?", usecols=["Date", "Time", "Global_active_power"])
    index = pd.to_datetime(raw["Date"] + " " + raw["Time"], format="%d/%m/%Y %H:%M:%S")
    return pd.Series(raw["Global_active_power"].to_numpy(), index=index, name="kw").sort_index()


def build_daily(minutes: pd.Series) -> pd.DataFrame:
    # Keep whole calendar days only; recording starts and stops part-way through a day.
    first = minutes.index[0].normalize() + pd.Timedelta(days=1)
    last = minutes.index[-1].normalize() - pd.Timedelta(minutes=1)
    minutes = minutes.loc[first:last]

    grouped = minutes.groupby(minutes.index.normalize())
    daily = pd.DataFrame({
        # Mean power (kW) x 24 h = energy (kWh). Using the mean scales up days with a few missing minutes.
        "energy_kwh": grouped.mean() * 24,
        "coverage": grouped.count() / MINUTES_PER_DAY,
    })
    daily = daily.asfreq("D")
    daily["coverage"] = daily["coverage"].fillna(0.0)
    daily.index.name = "date"

    daily["imputed"] = daily["coverage"] < MIN_COVERAGE
    daily.loc[daily["imputed"], "energy_kwh"] = float("nan")
    # Walk forward in time so a run of missing days can reuse a value filled earlier.
    for day in daily.index[daily["imputed"]]:
        daily.loc[day, "energy_kwh"] = daily.loc[day - pd.Timedelta(days=7), "energy_kwh"]
    assert daily["energy_kwh"].notna().all()
    return daily


def check_days_use_no_future(minutes: pd.Series) -> None:
    """Scramble every minute after a cut-off; the daily table up to the cut-off must not change."""
    cutoff = minutes.index[len(minutes) // 2].normalize()
    tampered = minutes.copy()
    tampered[tampered.index >= cutoff] = 99.0
    before = build_daily(minutes).loc[: cutoff - pd.Timedelta(days=1)]
    after = build_daily(tampered).loc[: cutoff - pd.Timedelta(days=1)]
    pd.testing.assert_frame_equal(before, after)


def main() -> None:
    set_global_seed(SEED)
    minutes = load_minutes()
    check_days_use_no_future(minutes)
    daily = build_daily(minutes)
    daily.to_csv(DAILY_PATH)
    print(f"{len(daily)} days, {daily.index[0].date()} to {daily.index[-1].date()}, "
          f"{int(daily['imputed'].sum())} imputed -> {DAILY_PATH}")


if __name__ == "__main__":
    main()

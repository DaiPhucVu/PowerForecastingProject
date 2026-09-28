"""The team's daily table, rebuilt from the raw UCI file exactly as notebooks/household_energy_model.ipynb does.

Run from the repo root: python -m src.team_daily
Reads  data/household_power_consumption.txt  (UCI, one row per minute)
Writes data/household_daily_team.csv         (one row per kept day)

Rules (same as the notebook, so every model scores the same days):
- kwh = sum of the day's Global_active_power readings (kW, one per minute) / 60
- a day is kept only if at least 90% of its 1,440 minutes have a reading; other days are dropped,
  not filled, so the table has date gaps
- the other columns are the day's other meter readings, in the notebook's units
"""

from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/household_power_consumption.txt")
TEAM_DAILY_PATH = Path("data/household_daily_team.csv")

MIN_COVERAGE = 0.9
MINUTES_PER_DAY = 1440

POWER_COLS = [
    "Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
    "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
]


def build_team_daily(raw_path: Path = RAW_PATH) -> pd.DataFrame:
    raw = pd.read_csv(raw_path, sep=";", na_values=["?"], low_memory=False, dtype=str)
    raw["timestamp"] = pd.to_datetime(raw["Date"] + " " + raw["Time"], format="%d/%m/%Y %H:%M:%S")
    for col in POWER_COLS:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.drop(columns=["Date", "Time"]).sort_values("timestamp")

    g = raw.set_index("timestamp").resample("D")
    daily = pd.DataFrame({
        "kwh": g.Global_active_power.sum(min_count=1) / 60,
        "reactive_kvarh": g.Global_reactive_power.sum(min_count=1) / 60,
        "voltage": g.Voltage.mean(),
        "intensity": g.Global_intensity.mean(),
        "sub1_wh": g.Sub_metering_1.sum(min_count=1),
        "sub2_wh": g.Sub_metering_2.sum(min_count=1),
        "sub3_wh": g.Sub_metering_3.sum(min_count=1),
        "mins": g.Global_active_power.count(),
    })
    daily["other_wh"] = daily.kwh * 1000 - daily.sub1_wh - daily.sub2_wh - daily.sub3_wh

    full = daily[daily.mins >= MIN_COVERAGE * MINUTES_PER_DAY].drop(columns="mins")
    full.index.name = "date"
    return full


def load_team_daily(path: Path = TEAM_DAILY_PATH) -> pd.DataFrame:
    """The cached table, built from the raw file first if it isn't there yet."""
    if not path.exists():
        build_team_daily().to_csv(path)
    return pd.read_csv(path, parse_dates=["date"], index_col="date")


def main() -> None:
    daily = build_team_daily()
    daily.to_csv(TEAM_DAILY_PATH)
    print(f"{len(daily)} days, {daily.index[0].date()} to {daily.index[-1].date()} -> {TEAM_DAILY_PATH}")


if __name__ == "__main__":
    main()

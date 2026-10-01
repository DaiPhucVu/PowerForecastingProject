"""
Build the daily table from the raw UCI household power file.

Usage:  python build_daily.py --raw raw/household_power_consumption.txt

Writes data/household_daily.csv, data/manifest.json, data/data_dictionary.md
"""
import argparse, hashlib, json
from pathlib import Path

import pandas as pd

MIN_COVERAGE = 0.90          # keep a day only if this share of its minutes exists
MINUTES_PER_DAY = 1440

NUMERIC = ["Global_active_power", "Global_reactive_power", "Voltage",
           "Global_intensity", "Sub_metering_1", "Sub_metering_2",
           "Sub_metering_3"]


def load_raw(path):
    # ; separated, ? for missing, Date and Time in separate columns.
    # Combining them by hand keeps this working on any pandas version.
    df = pd.read_csv(path, sep=";", na_values=["?"], low_memory=False, dtype=str)
    df["timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"],
                                     format="%d/%m/%Y %H:%M:%S")
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return (df.drop(columns=["Date", "Time"])
              .sort_values("timestamp").reset_index(drop=True))


def to_daily(df):
    g = df.set_index("timestamp").resample("D")

    out = pd.DataFrame({
        # Global_active_power is kW sampled each minute, so a day's sum is
        # kW-minutes. Divide by 60 for kWh.
        "kwh":            g["Global_active_power"].sum(min_count=1) / 60.0,
        "reactive_kvarh": g["Global_reactive_power"].sum(min_count=1) / 60.0,
        "voltage":        g["Voltage"].mean(),
        "intensity":      g["Global_intensity"].mean(),
        "sub1_wh":        g["Sub_metering_1"].sum(min_count=1),
        "sub2_wh":        g["Sub_metering_2"].sum(min_count=1),
        "sub3_wh":        g["Sub_metering_3"].sum(min_count=1),
        "minutes_seen":   g["Global_active_power"].count(),
    })
    out["coverage"] = out["minutes_seen"] / MINUTES_PER_DAY
    # active energy the three sub-meters do not account for
    out["other_wh"] = (out["kwh"] * 1000.0
                       - out["sub1_wh"] - out["sub2_wh"] - out["sub3_wh"])

    kept    = out[out["coverage"] >= MIN_COVERAGE].copy()
    dropped = out[out["coverage"] <  MIN_COVERAGE]

    kept = kept.drop(columns=["minutes_seen", "coverage"])
    kept.index.name = "date"
    return kept.reset_index(), dropped


def main(raw_path, outdir):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    print("reading raw file ...")
    df = load_raw(raw_path)
    blank = df[NUMERIC].isna().all(axis=1).sum()
    expected = int((df.timestamp.max() - df.timestamp.min()).total_seconds() // 60) + 1
    print(f"  rows               : {len(df):,}")
    print(f"  fully-blank rows   : {blank:,} ({blank/len(df)*100:.2f}%)")
    print(f"  range              : {df.timestamp.min()} -> {df.timestamp.max()}")
    print(f"  missing timestamps : {expected - len(df):,}")

    daily, dropped = to_daily(df)
    interior = int((daily["date"].diff().dt.days > 1).sum())
    print(f"  kept (>={MIN_COVERAGE:.0%})        : {len(daily):,}")
    print(f"  dropped (partial)  : {len(dropped):,}")
    print(f"  interior day gaps  : {interior}")

    daily.to_csv(outdir / "household_daily.csv", index=False)

    digest = hashlib.sha256(Path(raw_path).read_bytes()).hexdigest()
    (outdir / "manifest.json").write_text(json.dumps({
        "source": "UCI ML Repository - Individual household electric power consumption",
        "source_url": "https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption",
        "doi": "10.24432/C58K54",
        "raw_file": Path(raw_path).name,
        "raw_sha256": digest,
        "raw_rows": int(len(df)),
        "daily_rows": int(len(daily)),
        "date_from": str(daily.date.min().date()),
        "date_to": str(daily.date.max().date()),
        "min_coverage_rule": MIN_COVERAGE,
        "days_dropped_partial": int(len(dropped)),
        "interior_gaps": interior,
    }, indent=2))

    (outdir / "data_dictionary.md").write_text(f"""# Data dictionary - household_daily.csv

Source: UCI Individual household electric power consumption (Hebrail & Berard 2012),
one house in Sceaux, France, sampled every minute.
Raw SHA-256: `{digest[:16]}...`

Built by `build_daily.py`. A calendar day is kept only if at least
{MIN_COVERAGE:.0%} of its 1440 minutes were recorded; {len(dropped)} partial days were dropped.
No interpolation is applied.

| Column | Unit | Meaning |
|---|---|---|
| `date` | date | Calendar day |
| `kwh` | kWh | **Target.** Daily active energy. Minute kW readings summed, divided by 60. |
| `reactive_kvarh` | kVArh | Daily reactive energy, same conversion |
| `voltage` | V | Mean voltage across the day |
| `intensity` | A | Mean current intensity across the day |
| `sub1_wh` | Wh | Sub-meter 1: kitchen (dishwasher, oven, microwave) |
| `sub2_wh` | Wh | Sub-meter 2: laundry (washing machine, dryer, fridge, light) |
| `sub3_wh` | Wh | Sub-meter 3: water heater and air conditioner |
| `other_wh` | Wh | Active energy not captured by the three sub-meters (residual, not a metered circuit) |

Coverage: {daily.date.min().date()} to {daily.date.max().date()}, {len(daily)} days.
Note: {interior} interior gaps remain where whole days were dropped, so lag features
computed by row position can reach back more than the stated number of calendar days.
""")

    print(f"\nwrote {outdir}/household_daily.csv ({len(daily)} rows)")
    print(f"      {outdir}/manifest.json")
    print(f"      {outdir}/data_dictionary.md")
    return daily


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--outdir", default="data")
    a = ap.parse_args()
    main(a.raw, a.outdir)

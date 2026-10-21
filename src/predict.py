"""
Produce actual forecasts for days that have not happened yet.

This is the part a client uses. Everything else in the project measures the
models; this runs them forward.

Two things happen:

  Tomorrow      One step ahead with the `next_day` model, which can use
                yesterday's end-of-day readings because they exist.

  The next N    Recursive forecasting with the `horizon` model. Each predicted
                day becomes the next day's lag. Error compounds as it goes, so
                the output is honest about that rather than presenting day 30
                with the same confidence as day 1.

Feature rows are built from the TAIL of the series by position, which is how
make_features builds them during training - lag_1 means "the previous row", not
"the previous calendar day". At the end of the series those are the same thing,
because the dropped partial days sit in the interior.

Usage:  python predict.py --data data/household_daily.csv
        python predict.py --data data/household_daily.csv --days 30

Writes results/forecast.csv, results/forecast_monthly.csv, results/forecast.png
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_pipeline import TARGET, load

LAGS = (1, 2, 3, 7, 14)
WINDOWS = (7, 14, 28)


def calendar_features(date):
    d = pd.Timestamp(date)
    return {
        "dow": d.dayofweek,
        "month": d.month,
        "doy": d.dayofyear,
        "is_weekend": int(d.dayofweek >= 5),
        "dow_sin": np.sin(2 * np.pi * d.dayofweek / 7),
        "dow_cos": np.cos(2 * np.pi * d.dayofweek / 7),
        "doy_sin": np.sin(2 * np.pi * d.dayofyear / 365.25),
        "doy_cos": np.cos(2 * np.pi * d.dayofyear / 365.25),
    }


def history_features(series, date):
    """Target lags and rolling statistics for `date`, from the values that
    precede it. `series` is a plain list of past kWh values in time order."""
    row = {}
    for lag in LAGS:
        row[f"lag_{lag}"] = series[-lag]
    for w in WINDOWS:
        window = series[-w:]
        row[f"roll_mean_{w}"] = float(np.mean(window))
        row[f"roll_std_{w}"] = float(np.std(window, ddof=1))
    row.update(calendar_features(date))
    return row


def same_day_lag_features(daily, date):
    """Lags of the end-of-day columns, for a one-step-ahead forecast only.

    For `date` = the day after the last observed day, lag1 is the last observed
    row and lag7 is seven rows back - both real measurements.
    """
    row = {}
    for c in ("reactive_kvarh", "voltage", "intensity",
              "sub1_wh", "sub2_wh", "sub3_wh", "other_wh"):
        if c not in daily.columns:
            continue
        row[f"{c}_lag1"] = float(daily[c].iloc[-1])
        row[f"{c}_lag7"] = float(daily[c].iloc[-7])
    return row


def load_model(path):
    import joblib
    bundle = joblib.load(path)
    return bundle["model"], bundle["meta"]


def forecast(daily, days, models_dir):
    hist = daily[TARGET].tolist()
    last = pd.Timestamp(daily["date"].iloc[-1])

    nd_model, nd_meta = load_model(Path(models_dir) / "next_day.joblib")
    hz_model, hz_meta = load_model(Path(models_dir) / "horizon.joblib")

    rows = []

    # --- step 1: tomorrow, with the more accurate model -------------------
    d1 = last + pd.Timedelta(days=1)
    row = history_features(hist, d1)
    row.update(same_day_lag_features(daily, d1))
    X = pd.DataFrame([row])[nd_meta["features"]]
    p1 = float(nd_model.predict(X)[0])

    rows.append({"date": d1, "step": 1, "predicted_kwh": p1,
                 "model": "next_day", "expected_MAE": nd_meta["val_MAE"]})
    hist.append(p1)

    # --- steps 2..N: recursive, reduced feature set ------------------------
    # Error compounds, so the quoted uncertainty is widened with the square
    # root of the step count. That is the standard assumption for errors that
    # accumulate roughly independently - it is an estimate, not a measurement,
    # and is labelled as such in the output.
    for step in range(2, days + 1):
        d = last + pd.Timedelta(days=step)
        X = pd.DataFrame([history_features(hist, d)])[hz_meta["features"]]
        p = float(hz_model.predict(X)[0])

        rows.append({"date": d, "step": step, "predicted_kwh": p,
                     "model": "horizon",
                     "expected_MAE": hz_meta["val_MAE"] * np.sqrt(step)})
        hist.append(p)

    out = pd.DataFrame(rows)
    out["lower_kwh"] = (out.predicted_kwh - out.expected_MAE).clip(lower=0)
    out["upper_kwh"] = out.predicted_kwh + out.expected_MAE
    return out, nd_meta, hz_meta


def to_monthly(fc, last_observed):
    """Monthly totals from the daily forecasts, flagging any month the forecast
    only partly covers - a stub month is not a monthly total."""
    f = fc.copy()
    f["month"] = f.date.dt.to_period("M")

    g = (f.groupby("month")
          .agg(days_forecast=("predicted_kwh", "size"),
               predicted_kwh=("predicted_kwh", "sum"),
               lower_kwh=("lower_kwh", "sum"),
               upper_kwh=("upper_kwh", "sum"))
          .reset_index())

    g["days_in_month"] = g.month.dt.days_in_month
    g["complete"] = g.days_forecast == g.days_in_month
    g["month"] = g.month.astype(str)
    return g


def main(path, models_dir, outdir, days):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    daily = load(path)
    last = pd.Timestamp(daily["date"].iloc[-1])
    print(f"history: {len(daily)} days to {last.date()}")

    fc, nd_meta, hz_meta = forecast(daily, days, models_dir)

    print(f"\nTOMORROW ({fc.date.iloc[0].date()}): "
          f"{fc.predicted_kwh.iloc[0]:.2f} kWh "
          f"(+/- {fc.expected_MAE.iloc[0]:.2f} expected error)")

    print(f"\nnext {days} days")
    show = fc.head(10)[["date", "step", "predicted_kwh",
                        "lower_kwh", "upper_kwh", "model"]].copy()
    show["date"] = show.date.dt.date
    print(show.round(2).to_string(index=False))
    if days > 10:
        print(f"  ... {days - 10} more rows in results/forecast.csv")

    total = fc.predicted_kwh.sum()
    print(f"\ntotal over {days} days: {total:.1f} kWh "
          f"({total / days:.2f} kWh/day average)")

    monthly = to_monthly(fc, last)
    print("\nmonthly view")
    print(monthly.round(1).to_string(index=False))
    if not monthly.complete.all():
        part = monthly.loc[~monthly.complete, "month"].tolist()
        print(f"  partial months (forecast does not cover the whole month): "
              f"{', '.join(part)}")

    fc_out = fc.copy()
    fc_out["date"] = fc_out.date.dt.date
    fc_out.round(4).to_csv(outdir / "forecast.csv", index=False)
    monthly.round(4).to_csv(outdir / "forecast_monthly.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tail = daily.tail(60)
        fig, ax = plt.subplots(figsize=(12, 3.6))
        ax.plot(tail.date, tail[TARGET], lw=0.9, color="#333333",
                label="observed")
        ax.plot(fc.date, fc.predicted_kwh, lw=1.4, color="#c0392b",
                label="forecast")
        ax.fill_between(fc.date, fc.lower_kwh, fc.upper_kwh,
                        color="#c0392b", alpha=0.15,
                        label="expected error band")
        ax.axvline(last, color="#999999", ls=":", lw=1)
        ax.set_ylabel("kWh/day")
        ax.set_title(f"Last 60 observed days and the next {days} forecast")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / "forecast.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"\n(plot skipped: {e})")

    print(f"\nwrote -> {outdir}/forecast.csv, forecast_monthly.csv, forecast.png")
    print("\nNote: the band widens with the square root of the step count. "
          "It is an\nestimate of accumulating error, not a measured prediction "
          "interval.")
    return fc, monthly


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", default="models")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--days", type=int, default=14)
    a = ap.parse_args()
    main(a.data, a.models, a.outdir, a.days)

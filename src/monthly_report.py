"""
Roll the daily forecasts up to a monthly view - the level the client reports at.

The model is trained on days because there are ~1,400 of them and only 47 months.
The client plans at month scale. This script closes that gap: it trains the same
XGBoost used in forecast_pipeline.py, then aggregates the test-set predictions
and actuals by calendar month and scores them there.

Daily errors partly cancel when summed over a month - overshooting Tuesday and
undershooting Wednesday nets out - so monthly percentage error should come out
below daily percentage error. That is the argument for the whole design, so the
script prints both side by side rather than only the monthly figure.

Usage:  python monthly_report.py --data data/household_daily.csv

Writes results/monthly_report.csv, results/monthly_vs_daily.csv,
       results/monthly_plot.png
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from forecast_pipeline import (SEED, TARGET, load, make_features,
                               assert_no_leakage, chrono_split, mape)

# A month needs at least this many days present to be scored. The first and last
# months of the test window are usually partial, and a month holding four days
# is not a monthly total.
MIN_DAYS_IN_MONTH = 28


def fit_xgb(Xtr, ytr, Xva, yva):
    """Same two-stage fit as forecast_pipeline: pick n_estimators on the
    validation fold, then refit on train+val."""
    probe = XGBRegressor(n_estimators=2000, learning_rate=0.05, max_depth=4,
                         subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                         random_state=SEED, n_jobs=1,
                         early_stopping_rounds=50, eval_metric="mae")
    probe.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    best_n = probe.best_iteration + 1

    model = XGBRegressor(n_estimators=best_n, learning_rate=0.05, max_depth=4,
                         subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                         random_state=SEED, n_jobs=1)
    model.fit(pd.concat([Xtr, Xva]), pd.concat([ytr, yva]))
    return model, best_n


def to_monthly(dates, actual, predicted):
    df = pd.DataFrame({"date": pd.to_datetime(dates),
                       "actual": np.asarray(actual, float),
                       "predicted": np.asarray(predicted, float)})
    df["month"] = df["date"].dt.to_period("M")

    out = (df.groupby("month")
             .agg(days=("actual", "size"),
                  actual_kwh=("actual", "sum"),
                  predicted_kwh=("predicted", "sum"))
             .reset_index())

    out["error_kwh"] = out["predicted_kwh"] - out["actual_kwh"]
    out["abs_error_kwh"] = out["error_kwh"].abs()
    out["error_pct"] = out["error_kwh"] / out["actual_kwh"] * 100
    out["complete"] = out["days"] >= MIN_DAYS_IN_MONTH
    out["month"] = out["month"].astype(str)
    return out


def plot(monthly, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    full = monthly[monthly["complete"]]
    if full.empty:
        print("  (no complete months, skipping plot)")
        return

    x = np.arange(len(full))
    fig, ax = plt.subplots(figsize=(max(6, len(full) * 0.9), 4))
    ax.bar(x - 0.2, full["actual_kwh"],    width=0.4, label="actual")
    ax.bar(x + 0.2, full["predicted_kwh"], width=0.4, label="predicted")
    ax.set_xticks(x)
    ax.set_xticklabels(full["month"], rotation=45, ha="right")
    ax.set_ylabel("kWh per month")
    ax.set_title("Monthly consumption: actual vs predicted (test period)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(path, outdir):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    raw = load(path)
    feat = make_features(raw)
    assert_no_leakage(raw, feat)

    train, val, test = chrono_split(feat)
    drop = ["date", TARGET]
    model, best_n = fit_xgb(train.drop(columns=drop), train[TARGET],
                            val.drop(columns=drop),   val[TARGET])

    pred = model.predict(test.drop(columns=drop))
    yte = test[TARGET].to_numpy()

    monthly = to_monthly(test["date"], yte, pred)
    full = monthly[monthly["complete"]]

    # Daily figures on the same test set, for the comparison that matters.
    daily_mae = float(np.mean(np.abs(pred - yte)))
    daily_mape = mape(yte, pred)

    if full.empty:
        raise SystemExit(
            f"No month in the test window has {MIN_DAYS_IN_MONTH}+ days. "
            "The test set is too short to report monthly - widen test_frac.")

    monthly_mae = float(full["abs_error_kwh"].mean())
    monthly_mape = float(full["error_pct"].abs().mean())

    compare = pd.DataFrame([
        {"level": "daily",   "unit": "kWh/day",
         "MAE": round(daily_mae, 3),   "MAPE_%": round(daily_mape, 2),
         "n": len(test)},
        {"level": "monthly", "unit": "kWh/month",
         "MAE": round(monthly_mae, 3), "MAPE_%": round(monthly_mape, 2),
         "n": int(len(full))},
    ])

    print(f"\nXGBoost (n={best_n}) | test = {len(test)} days "
          f"from {test.date.min().date()}")
    print("\nMonthly totals")
    print(monthly.round(2).to_string(index=False))
    if (~monthly["complete"]).any():
        dropped = monthly.loc[~monthly["complete"], "month"].tolist()
        print(f"\nPartial months excluded from the scores: {', '.join(dropped)}")

    print("\n" + "=" * 52)
    print("DAILY vs MONTHLY on the same predictions")
    print("=" * 52)
    print(compare.to_string(index=False))

    gap = daily_mape - monthly_mape
    if gap > 0:
        print(f"\nMonthly MAPE is {gap:.2f} points lower than daily. Day-level "
              f"errors partly cancel when summed over a month, which is why the\n"
              f"model is fitted on days and reported on months.")
    else:
        print(f"\nMonthly MAPE is NOT lower than daily ({monthly_mape:.2f} vs "
              f"{daily_mape:.2f}). Worth reporting honestly - it means the\n"
              f"daily errors are biased in one direction rather than cancelling.")

    monthly.to_csv(outdir / "monthly_report.csv", index=False)
    compare.to_csv(outdir / "monthly_vs_daily.csv", index=False)
    plot(monthly, outdir / "monthly_plot.png")

    print(f"\nwrote -> {outdir}/monthly_report.csv, monthly_vs_daily.csv, "
          f"monthly_plot.png")
    return monthly, compare


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()
    main(a.data, a.outdir)

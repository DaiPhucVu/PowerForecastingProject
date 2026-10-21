"""
Feature-group ablation: which groups of features actually earn their place.

SHAP says `other_wh_lag1` is the largest single contributor. But `other_wh` is
computed as total active energy minus the three sub-meters, so that column is a
linear combination of yesterday's total and its metered parts - all of which are
also in the feature table. SHAP splits credit between correlated features more
or less arbitrarily, so a large SHAP value does not prove a feature is carrying
information nothing else carries.

Removing a group and retraining does prove it. If the error barely moves, the
group was redundant. If it gets clearly worse, the group held signal of its own.

Each group is dropped across all five rolling-origin folds rather than one test
window, so the answer is not an artefact of where the split happened to fall.

Usage:  python ablation.py --data data/household_daily.csv
        python ablation.py --data data/household_daily.csv --folds 5

Writes results/ablation.csv, results/ablation_plot.png
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from forecast_pipeline import (SEED, TARGET, load, make_features,
                               assert_no_leakage, mape)
from rolling_validation import folds

# Each group is a list of regex patterns matched against column names.
GROUPS = {
    "all features": [],
    "no residual lags": [r"^other_wh_"],
    "no sub-meter lags": [r"^sub[123]_wh_"],
    "no electrical lags": [r"^voltage_", r"^intensity_", r"^reactive_kvarh_"],
    "no calendar": [r"^dow", r"^month$", r"^doy", r"^is_weekend$"],
    "no rolling stats": [r"^roll_"],
    "target lags only": [r"^(?!lag_\d+$).*"],      # keep lag_1..lag_14, drop all else
}


def columns_for(group, all_cols):
    """Apply a group's drop patterns and return the surviving columns."""
    patterns = GROUPS[group]
    keep = []
    for c in all_cols:
        if any(re.match(p, c) for p in patterns):
            continue
        keep.append(c)
    return keep


def fit_xgb(Xtr, ytr):
    """Early stopping on the last 15% of the training block - still in time
    order, so nothing after the split is seen."""
    cut = int(len(Xtr) * 0.85)
    probe = XGBRegressor(n_estimators=1500, learning_rate=0.05, max_depth=4,
                         subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                         random_state=SEED, n_jobs=1,
                         early_stopping_rounds=50, eval_metric="mae")
    probe.fit(Xtr.iloc[:cut], ytr.iloc[:cut],
              eval_set=[(Xtr.iloc[cut:], ytr.iloc[cut:])], verbose=False)

    m = XGBRegressor(n_estimators=probe.best_iteration + 1, learning_rate=0.05,
                     max_depth=4, subsample=0.8, colsample_bytree=0.8,
                     reg_lambda=1.0, random_state=SEED, n_jobs=1)
    m.fit(Xtr, ytr)
    return m


def main(path, outdir, k):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    raw = load(path)
    feat = make_features(raw)
    assert_no_leakage(raw, feat)

    all_cols = [c for c in feat.columns if c not in ("date", TARGET)]
    n = len(feat)
    print(f"{n} rows | {len(all_cols)} features | {k} folds\n")

    rows = []
    for group in GROUPS:
        cols = columns_for(group, all_cols)
        if not cols:
            print(f"{group:<20} skipped - no features left")
            continue

        maes, mapes = [], []
        for train_end, test_end in folds(n, k):
            tr = feat.iloc[:train_end]
            te = feat.iloc[train_end:test_end]
            m = fit_xgb(tr[cols], tr[TARGET])
            p = m.predict(te[cols])
            maes.append(float(np.mean(np.abs(p - te[TARGET].to_numpy()))))
            mapes.append(mape(te[TARGET], p))

        rows.append({
            "feature_set": group,
            "n_features": len(cols),
            "dropped": len(all_cols) - len(cols),
            "mean_MAE": float(np.mean(maes)),
            "std_MAE": float(np.std(maes, ddof=1)),
            "mean_MAPE": float(np.mean(mapes)),
        })
        print(f"{group:<20} {len(cols):>3} features  "
              f"MAE {np.mean(maes):.3f} +/- {np.std(maes, ddof=1):.3f}")

    res = pd.DataFrame(rows)
    base = res.loc[res.feature_set == "all features"].iloc[0]
    res["MAE_change"] = res.mean_MAE - base.mean_MAE
    res["MAE_change_%"] = res.MAE_change / base.mean_MAE * 100

    # Is the change bigger than the fold-to-fold noise of the full model? If
    # not, the group is redundant and should not be called a key driver.
    res["beyond_noise"] = res.MAE_change.abs() > base.std_MAE
    res = res.round(3)

    print("\n" + "=" * 86)
    print(f"ABLATION  (XGBoost, mean of {k} rolling-origin folds, "
          f"full-model fold-to-fold std = {base.std_MAE:.3f})")
    print("=" * 86)
    print(res.to_string(index=False))

    print("\nReading it:")
    for _, r in res.iterrows():
        if r.feature_set == "all features":
            continue
        if not r.beyond_noise:
            print(f"  {r.feature_set:<20} redundant - error moved "
                  f"{r.MAE_change:+.3f} kWh, inside the noise")
        elif r.MAE_change > 0:
            print(f"  {r.feature_set:<20} CARRIES SIGNAL - error rose "
                  f"{r.MAE_change:+.3f} kWh ({r['MAE_change_%']:+.1f}%)")
        else:
            print(f"  {r.feature_set:<20} HURT the model - error FELL "
                  f"{r.MAE_change:+.3f} kWh when removed")

    res.to_csv(outdir / "ablation.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot = res[res.feature_set != "all features"].sort_values("MAE_change")
        colours = ["#888888" if not b else "#c0392b"
                   for b in plot.beyond_noise]
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.barh(plot.feature_set, plot.MAE_change, color=colours)
        ax.axvline(0, color="black", lw=0.8)
        ax.axvline(base.std_MAE, color="black", ls=":", lw=0.8,
                   label="fold-to-fold noise")
        ax.axvline(-base.std_MAE, color="black", ls=":", lw=0.8)
        ax.set_xlabel("change in MAE when the group is removed (kWh)")
        ax.set_title("Feature-group ablation")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / "ablation_plot.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"\n(plot skipped: {e})")

    print(f"\nwrote -> {outdir}/ablation.csv, ablation_plot.png")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    main(a.data, a.outdir, a.folds)

"""
Explainability for the tree model, using TreeSHAP.

Gain importance (what forecast_pipeline.py already writes) ranks features but
says nothing about direction or size. It tells the client that other_wh_lag1
matters; it does not tell them whether a high value pushes consumption up or
down, or by how many kWh.

SHAP values do. Each one is a contribution in kWh to a single day's
prediction, and they sum to the difference between that prediction and the
average prediction. That makes them reportable in the client's own units
rather than as a unitless score.

Usage:  python explain.py --data data/household_daily.csv
        python explain.py --data data/household_daily.csv --model rf

Writes results/shap_summary.csv       mean |SHAP| and direction per feature
       results/shap_drivers.md        the same thing in plain English
       results/shap_beeswarm.png      distribution of effects
       results/shap_waterfall_*.png   two single days explained end to end
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_pipeline import (SEED, TARGET, load, make_features,
                               assert_no_leakage, chrono_split)

# Readable names for the report. Anything not listed falls back to the raw
# column name, so this map can stay partial.
PRETTY = {
    "lag_1": "yesterday's consumption",
    "lag_2": "consumption two days ago",
    "lag_3": "consumption three days ago",
    "lag_7": "same day last week",
    "lag_14": "same day two weeks ago",
    "roll_mean_7": "average of the last 7 days",
    "roll_mean_14": "average of the last 14 days",
    "roll_mean_28": "average of the last 28 days",
    "roll_std_7": "volatility over the last 7 days",
    "roll_std_14": "volatility over the last 14 days",
    "roll_std_28": "volatility over the last 28 days",
    "dow": "day of week",
    "dow_sin": "day of week (cyclical)",
    "dow_cos": "day of week (cyclical)",
    "doy": "day of year",
    "doy_sin": "time of year (cyclical)",
    "doy_cos": "time of year (cyclical)",
    "month": "month",
    "is_weekend": "weekend or weekday",
    "other_wh_lag1": "yesterday's unmetered load (residual)",
    "other_wh_lag7": "unmetered load same day last week",
    "sub1_wh_lag1": "yesterday's kitchen circuit",
    "sub2_wh_lag1": "yesterday's laundry circuit",
    "sub3_wh_lag1": "yesterday's water heater and A/C",
    "sub1_wh_lag7": "kitchen circuit same day last week",
    "sub2_wh_lag7": "laundry circuit same day last week",
    "sub3_wh_lag7": "water heater and A/C same day last week",
    "voltage_lag1": "yesterday's mean voltage",
    "voltage_lag7": "mean voltage same day last week",
    "intensity_lag1": "yesterday's mean current",
    "intensity_lag7": "mean current same day last week",
    "reactive_kvarh_lag1": "yesterday's reactive energy",
    "reactive_kvarh_lag7": "reactive energy same day last week",
}


def pretty(name):
    return PRETTY.get(name, name.replace("_", " "))


def build_model(kind, Xtr, ytr, Xva, yva):
    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                                  random_state=SEED, n_jobs=1)
        m.fit(pd.concat([Xtr, Xva]), pd.concat([ytr, yva]))
        return m, "Random Forest"

    from xgboost import XGBRegressor
    probe = XGBRegressor(n_estimators=2000, learning_rate=0.05, max_depth=4,
                         subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                         random_state=SEED, n_jobs=1,
                         early_stopping_rounds=50, eval_metric="mae")
    probe.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    best_n = probe.best_iteration + 1

    m = XGBRegressor(n_estimators=best_n, learning_rate=0.05, max_depth=4,
                     subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                     random_state=SEED, n_jobs=1)
    m.fit(pd.concat([Xtr, Xva]), pd.concat([ytr, yva]))
    return m, f"XGBoost (n={best_n})"


def summarise(shap_values, X):
    """Per-feature: average size of effect, and whether high feature values
    push the prediction up or down.

    Direction is the correlation between the feature's value and its own SHAP
    value. Positive means a high value raises the forecast. Near zero means
    the effect is not monotonic - worth saying rather than hiding.
    """
    rows = []
    for i, col in enumerate(X.columns):
        sv = shap_values[:, i]
        xv = X[col].to_numpy(float)

        if np.std(xv) < 1e-12 or np.std(sv) < 1e-12:
            corr = 0.0
        else:
            corr = float(np.corrcoef(xv, sv)[0, 1])

        rows.append({
            "feature": col,
            "plain_name": pretty(col),
            "mean_abs_shap_kwh": float(np.mean(np.abs(sv))),
            "max_abs_shap_kwh": float(np.max(np.abs(sv))),
            "direction_corr": corr,
            "direction": ("higher raises the forecast" if corr > 0.2
                          else "higher lowers the forecast" if corr < -0.2
                          else "mixed / not monotonic"),
        })

    return (pd.DataFrame(rows)
            .sort_values("mean_abs_shap_kwh", ascending=False)
            .reset_index(drop=True))


def write_drivers(summary, model_name, base_value, n_rows, path, top=6):
    lines = [
        "# What drives the daily forecast",
        "",
        f"Model: {model_name}. Explained on {n_rows} held-out test days using "
        f"TreeSHAP.",
        "",
        f"The model's average prediction across these days is "
        f"**{base_value:.2f} kWh**. Each figure below is how far that one "
        f"feature moves a day's forecast away from that average, in kWh, "
        f"averaged over all the test days.",
        "",
    ]
    for i, r in summary.head(top).iterrows():
        lines.append(
            f"{i + 1}. **{r['plain_name']}** - moves the forecast by "
            f"{r['mean_abs_shap_kwh']:.2f} kWh on an average day, up to "
            f"{r['max_abs_shap_kwh']:.2f} kWh on the most affected day. "
            f"{r['direction'].capitalize()}."
        )

    lines += [
        "",
        "## Reading this",
        "",
        "SHAP values are additive: for any single day, the base value plus "
        "every feature's contribution equals that day's prediction exactly. "
        "So these numbers can be checked, not just trusted.",
        "",
        "A feature marked *mixed / not monotonic* matters, but not in one "
        "direction - typically because its effect depends on another feature. "
        "Calendar features behave this way by nature.",
        "",
    ]

    if not summary.empty and "other_wh" in summary.iloc[0]["feature"]:
        lines += [
            "## Caveat on the top driver",
            "",
            "`other_wh` is a **residual**: total active energy minus the three "
            "sub-metered circuits. It is not a measured appliance group. Its "
            "importance says that load outside the three monitored circuits "
            "carries much of the day-to-day variation, which is a finding "
            "about the metering coverage of this house, not about a specific "
            "appliance.",
            "",
        ]

    Path(path).write_text("\n".join(lines))


def main(path, outdir, kind, sample):
    try:
        import shap
    except ImportError:
        raise SystemExit("shap is not installed. Run:  pip install shap")

    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    raw = load(path)
    feat = make_features(raw)
    assert_no_leakage(raw, feat)
    train, val, test = chrono_split(feat)

    drop = ["date", TARGET]
    model, model_name = build_model(
        kind, train.drop(columns=drop), train[TARGET],
        val.drop(columns=drop), val[TARGET])

    Xte = test.drop(columns=drop)
    if sample and len(Xte) > sample:
        Xte = Xte.iloc[:sample]
    print(f"{model_name} | explaining {len(Xte)} test days")

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xte)
    base = float(np.ravel(explainer.expected_value)[0])

    # Additivity check. If this fails the explanation is not faithful to the
    # model and should not be reported.
    recon = base + sv.sum(axis=1)
    worst = float(np.abs(recon - model.predict(Xte)).max())
    assert worst < 1e-2, f"SHAP values do not reconstruct predictions ({worst})"
    print(f"  additivity check passed (max deviation {worst:.2e} kWh)")

    summary = summarise(sv, Xte)
    summary.round(4).to_csv(outdir / "shap_summary.csv", index=False)
    write_drivers(summary, model_name, base, len(Xte),
                  outdir / "shap_drivers.md")

    print(f"\n  base value: {base:.2f} kWh")
    print("\n  top drivers (mean |SHAP|, kWh):")
    for _, r in summary.head(8).iterrows():
        print(f"    {r['mean_abs_shap_kwh']:6.3f}  {r['plain_name']:<42}"
              f" {r['direction']}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        shap.summary_plot(sv, Xte, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(outdir / "shap_beeswarm.png", dpi=150)
        plt.close()

        # Two individual days: the highest and lowest actual consumption in
        # the explained window. These are the slides that land in a demo.
        y = test[TARGET].iloc[:len(Xte)]
        for label, idx in (("highest_day", int(np.argmax(y.to_numpy()))),
                           ("lowest_day", int(np.argmin(y.to_numpy())))):
            ex = shap.Explanation(values=sv[idx], base_values=base,
                                  data=Xte.iloc[idx],
                                  feature_names=[pretty(c) for c in Xte.columns])
            shap.plots.waterfall(ex, max_display=12, show=False)
            plt.tight_layout()
            plt.savefig(outdir / f"shap_waterfall_{label}.png", dpi=150)
            plt.close()
        print("\n  wrote beeswarm and two waterfall plots")
    except Exception as e:
        print(f"\n  (plots skipped: {e})")

    print(f"\nwrote -> {outdir}/shap_summary.csv, shap_drivers.md, *.png")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--model", default="xgb", choices=["xgb", "rf"])
    ap.add_argument("--sample", type=int, default=0,
                    help="explain only the first N test days (0 = all)")
    a = ap.parse_args()
    main(a.data, a.outdir, a.model, a.sample)

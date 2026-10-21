"""
Train the final models on all available history and save them to disk.

Everything else in this project measures models on held-out history. This step
produces the artefact a forecast actually comes from: a fitted model, the exact
feature list it expects, and a record of what it was trained on.

Two models are saved, because the feature set does not support both jobs.

  next_day   Every feature, including lags of the end-of-day columns
             (voltage, sub-meters, residual). Yesterday's values for those ARE
             known when forecasting tomorrow, so this model is the most
             accurate - but only one day ahead.

  horizon    Target lags, rolling statistics and calendar only. Nothing derived
             from the end-of-day columns, because forecasting day+2 would need
             day+1's sub-meter readings, which will not exist. This model can
             therefore be run recursively for any number of days.

The ablation showed the reduced feature set costs about 8% MAE, which is the
price of forecasting further than one day out. Worth knowing rather than
discovering it when the client asks for a month.

Usage:  python train_model.py --data data/household_daily.csv

Writes models/next_day.joblib, models/horizon.joblib, models/model_card.md
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from forecast_pipeline import (SEED, TARGET, SAME_DAY_ONLY, load,
                               make_features, assert_no_leakage, mape)

PARAMS = dict(learning_rate=0.05, max_depth=4, subsample=0.8,
              colsample_bytree=0.8, reg_lambda=1.0,
              random_state=SEED, n_jobs=1)


def feature_sets(feat):
    """Split the columns into the two usable sets."""
    cols = [c for c in feat.columns if c not in ("date", TARGET)]

    # Anything derived from a column only known at the end of a day.
    derived = tuple(f"{c}_" for c in SAME_DAY_ONLY)
    horizon = [c for c in cols if not c.startswith(derived)]

    return {"next_day": cols, "horizon": horizon}


def tune_then_refit(feat, cols, val_frac=0.15):
    """Choose n_estimators on a chronological holdout, then refit on everything.

    The holdout exists only to pick the tree count. Once that number is fixed
    the model is retrained on all available history, because a forecast for
    tomorrow should use every day up to today - not stop short to preserve a
    test set that has already served its purpose.
    """
    X, y = feat[cols], feat[TARGET]
    cut = int(len(feat) * (1 - val_frac))

    probe = XGBRegressor(n_estimators=2000, early_stopping_rounds=50,
                         eval_metric="mae", **PARAMS)
    probe.fit(X.iloc[:cut], y.iloc[:cut],
              eval_set=[(X.iloc[cut:], y.iloc[cut:])], verbose=False)
    best_n = probe.best_iteration + 1

    # Holdout score at that tree count - the honest expectation for new days.
    p = probe.predict(X.iloc[cut:])
    holdout = {
        "val_days": int(len(feat) - cut),
        "val_from": str(feat.date.iloc[cut].date()),
        "val_MAE": float(np.mean(np.abs(p - y.iloc[cut:].to_numpy()))),
        "val_MAPE": float(mape(y.iloc[cut:], p)),
    }

    final = XGBRegressor(n_estimators=best_n, **PARAMS)
    final.fit(X, y)
    return final, best_n, holdout


def main(path, outdir):
    import joblib

    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    raw = load(path)
    feat = make_features(raw)
    assert_no_leakage(raw, feat)

    sets = feature_sets(feat)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    cards = {}

    for name, cols in sets.items():
        model, best_n, holdout = tune_then_refit(feat, cols)

        meta = {
            "name": name,
            "features": cols,
            "n_features": len(cols),
            "n_estimators": int(best_n),
            "target": TARGET,
            "trained_rows": int(len(feat)),
            "trained_from": str(feat.date.min().date()),
            "trained_to": str(feat.date.max().date()),
            "data_file": Path(path).name,
            "data_sha256_12": digest[:12],
            "seed": SEED,
            "params": PARAMS,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **holdout,
        }

        joblib.dump({"model": model, "meta": meta}, outdir / f"{name}.joblib")
        cards[name] = meta

        print(f"{name:<9} {len(cols):>2} features, {best_n:>3} trees | "
              f"holdout MAE {holdout['val_MAE']:.3f} "
              f"MAPE {holdout['val_MAPE']:.1f}% "
              f"on {holdout['val_days']} days from {holdout['val_from']}")

    nd, hz = cards["next_day"], cards["horizon"]
    cost = (hz["val_MAE"] - nd["val_MAE"]) / nd["val_MAE"] * 100

    (outdir / "model_card.md").write_text(f"""# Model card

Household daily electricity forecasting, Forecasting 5 (ICT30016).
Client: ARC Research Hub for Integrated Energy Storage Solutions.

Trained {nd['trained_at']} on `{nd['data_file']}`
(SHA-256 `{nd['data_sha256_12']}...`), {nd['trained_rows']} days from
{nd['trained_from']} to {nd['trained_to']}. Seed {SEED}.

## The two models

| | `next_day` | `horizon` |
|---|---|---|
| Features | {nd['n_features']} | {hz['n_features']} |
| Trees | {nd['n_estimators']} | {hz['n_estimators']} |
| Holdout MAE (kWh) | {nd['val_MAE']:.3f} | {hz['val_MAE']:.3f} |
| Holdout MAPE | {nd['val_MAPE']:.1f}% | {hz['val_MAPE']:.1f}% |
| Forecast range | 1 day | any |

`next_day` uses lags of the end-of-day columns (voltage, current, reactive
energy, the three sub-meters and the residual). Yesterday's values for those
are known when forecasting tomorrow, so it is the more accurate model - but
only one step ahead.

`horizon` uses target lags, rolling statistics and calendar features only, so
it can be run recursively: each prediction becomes the next step's lag. It
costs {cost:+.1f}% MAE against `next_day`, which is the measured price of
forecasting further than one day out.

## Intended use

Short-horizon forecasts of total daily active energy for one household, to
inform storage sizing and dispatch. Daily predictions aggregate to a monthly
view with lower percentage error than the daily figures.

## Limits

- **One household.** Trained on a single house in Sceaux, France
  (UCI, Hebrail & Berard 2012). Nothing here establishes transfer to other
  households, other climates or Australian consumption patterns.
- **No weather input.** Temperature drives household consumption and is not in
  this dataset. Lagged consumption absorbs some of it indirectly.
- **Recursive error growth.** `horizon` feeds its own predictions back in, so
  accuracy decays with distance. The holdout figure above is a one-step score
  and is optimistic for longer horizons.
- **Retraining.** Both models are fixed at the date above. Consumption habits
  drift, so they should be retrained when new data arrives rather than trusted
  indefinitely.

## Validation performed

Chronological splits throughout; no random k-fold (Bergmeir & Benitez 2012).
Leakage assertions run on every build. Rolling-origin validation over five
folds, a feature-group ablation, and TreeSHAP attribution with an additivity
check. See `results/` and `tests/`.
""")

    print(f"\nhorizon model costs {cost:+.1f}% MAE vs next_day - the price of "
          f"forecasting past tomorrow")
    print(f"\nwrote -> {outdir}/next_day.joblib, horizon.joblib, model_card.md")
    return cards


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", default="models")
    a = ap.parse_args()
    main(a.data, a.outdir)

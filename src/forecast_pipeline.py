"""
Features, chronological split, baseline + ML models, comparison table.

Usage:  python forecast_pipeline.py --data data/household_daily.csv

Writes results/<model>_test_predictions.csv for ridge, random_forest and xgboost
(the shared format, see README, "Dashboard"), results/xgb_feature_importance.csv and
results/pipeline_metadata.json. The comparison table across every model, including
the LSTM, is built by src/evaluate.py.
"""
import argparse, hashlib, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

SEED = 42
TARGET = "kwh"

# Known only at the END of a day, so they may appear as lags but never raw.
SAME_DAY_ONLY = ["reactive_kvarh", "voltage", "intensity",
                 "sub1_wh", "sub2_wh", "sub3_wh", "other_wh"]


def load(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def make_features(df):
    out = df[["date", TARGET]].copy()

    for lag in (1, 2, 3, 7, 14):
        out[f"lag_{lag}"] = df[TARGET].shift(lag)
    for w in (7, 14, 28):
        out[f"roll_mean_{w}"] = df[TARGET].shift(1).rolling(w).mean()
        out[f"roll_std_{w}"]  = df[TARGET].shift(1).rolling(w).std()

    for c in [c for c in SAME_DAY_ONLY if c in df.columns]:
        out[f"{c}_lag1"] = df[c].shift(1)
        out[f"{c}_lag7"] = df[c].shift(7)

    d = out["date"].dt
    out["dow"]        = d.dayofweek
    out["month"]      = d.month
    out["doy"]        = d.dayofyear
    out["is_weekend"] = (d.dayofweek >= 5).astype(int)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7)
    out["doy_sin"] = np.sin(2 * np.pi * out["doy"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["doy"] / 365.25)

    return out.dropna().reset_index(drop=True)


def assert_no_leakage(raw, feat):
    prev = raw[["date", TARGET]].copy()
    prev["date"] = prev["date"] + pd.Timedelta(days=1)
    chk = feat[["date", "lag_1"]].merge(prev, on="date", how="inner")
    worst = (chk["lag_1"] - chk[TARGET]).abs().max()
    assert worst < 1e-9, f"lag_1 does not match the previous day (max diff {worst})"
    assert not set(SAME_DAY_ONLY) & set(feat.columns), \
        "a same-day column reached the feature table"
    print("  leakage checks passed")


def chrono_split(feat, test_frac=0.20, val_frac=0.20):
    n = len(feat)
    n_test = int(round(n * test_frac))
    trval, test = feat.iloc[: n - n_test], feat.iloc[n - n_test:]
    n_val = int(round(len(trval) * val_frac))
    return (trval.iloc[: len(trval) - n_val],
            trval.iloc[len(trval) - n_val:],
            test)


def mape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = y != 0                     # MAPE is undefined at zero
    return np.mean(np.abs((y[m] - p[m]) / y[m])) * 100


def score(name, y, p):
    return {"model": name,
            "MAE":  mean_absolute_error(y, p),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAPE_%": mape(y, p)}


def main(path, outdir):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    raw = load(path)
    print(f"loaded {len(raw)} days  {raw.date.min().date()} -> {raw.date.max().date()}")

    feat = make_features(raw)
    assert_no_leakage(raw, feat)
    print(f"  feature table: {len(feat)} rows x {feat.shape[1]-2} features")

    train, val, test = chrono_split(feat)
    print(f"  train {len(train)} | val {len(val)} | test {len(test)}"
          f"  (test starts {test.date.min().date()})")

    drop = ["date", TARGET]
    Xtr, ytr = train.drop(columns=drop), train[TARGET]
    Xva, yva = val.drop(columns=drop),   val[TARGET]
    Xte, yte = test.drop(columns=drop),  test[TARGET]
    Xfull, yfull = pd.concat([Xtr, Xva]), pd.concat([ytr, yva])

    rows = [score("Seasonal naive (lag 7)", yte, test["lag_7"])]
    predictions = {}

    predictions["ridge"] = Ridge(alpha=1.0).fit(Xfull, yfull).predict(Xte)
    rows.append(score("Ridge regression", yte, predictions["ridge"]))

    rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                               random_state=SEED, n_jobs=1).fit(Xfull, yfull)
    predictions["random_forest"] = rf.predict(Xte)
    rows.append(score("Random Forest", yte, predictions["random_forest"]))

    # pick n_estimators on the validation fold, then refit on train+val
    probe = XGBRegressor(n_estimators=2000, learning_rate=0.05, max_depth=4,
                         subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                         random_state=SEED, n_jobs=1,
                         early_stopping_rounds=50, eval_metric="mae")
    probe.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    best_n = probe.best_iteration + 1

    xgb = XGBRegressor(n_estimators=best_n, learning_rate=0.05, max_depth=4,
                       subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                       random_state=SEED, n_jobs=1).fit(Xfull, yfull)
    predictions["xgboost"] = xgb.predict(Xte)
    rows.append(score(f"XGBoost (n={best_n})", yte, predictions["xgboost"]))

    res  = pd.DataFrame(rows).round(3)
    base = res.loc[0, "MAE"]
    res["MAE_vs_baseline_%"] = ((base - res["MAE"]) / base * 100).round(1)
    res = res.sort_values("MAE").reset_index(drop=True)

    print("\n" + "=" * 68)
    print(f"COMPARISON  (test = last {len(test)} days, target = daily kWh)")
    print("=" * 68)
    print(res.to_string(index=False))

    for model, pred in predictions.items():
        pd.DataFrame({"date": test["date"], "actual_kwh": yte, f"{model}_kwh": pred}) \
          .to_csv(outdir / f"{model}_test_predictions.csv", index=False)
    (pd.Series(xgb.feature_importances_, index=Xfull.columns)
       .sort_values(ascending=False).head(15).round(4)
       .to_csv(outdir / "xgb_feature_importance.csv", header=["importance"]))

    (outdir / "pipeline_metadata.json").write_text(json.dumps({
        "data_file": Path(path).name,
        "data_sha256_12": hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12],
        "seed": SEED,
        "rows_after_features": len(feat),
        "train": len(train), "val": len(val), "test": len(test),
        "test_start": str(test.date.min().date()),
        "xgb_best_n_estimators": int(best_n),
    }, indent=2))

    print(f"\nwrote -> {outdir}/" + ", ".join(f"{m}_test_predictions.csv" for m in predictions)
          + ", xgb_feature_importance.csv, pipeline_metadata.json")
    print("build the comparison table with: python -m src.evaluate")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()
    main(a.data, a.outdir)

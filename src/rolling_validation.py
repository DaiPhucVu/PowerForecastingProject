"""
Rolling-origin (forward-chaining) validation.

forecast_pipeline.py scores every model on ONE test window - the last 20% of
days. That gives a single number with no sense of how much it would move if the
window had fallen elsewhere. If that window happens to be a mild spring, the
model looks better than it is.

This script walks the origin forward instead. Fold 1 trains on the earliest
block and tests on the next; fold 2 trains on everything fold 1 saw plus its
test block, and tests on the block after that; and so on. Training data only
ever precedes test data, so no future information leaks backwards.

Each model then has K scores instead of one, and the spread across folds is the
honest statement of reliability.

Usage:  python rolling_validation.py --data data/household_daily.csv
        python rolling_validation.py --data data/household_daily.csv --folds 6

Writes results/rolling_folds.csv      (every model on every fold)
       results/rolling_summary.csv    (mean, std, worst, win rate)
       results/rolling_plot.png
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from forecast_pipeline import (SEED, TARGET, load, make_features,
                               assert_no_leakage, mape)

BASELINE = "Seasonal naive (lag 7)"


def folds(n, k, min_train_frac=0.40):
    """Expanding-window fold boundaries.

    The first fold must already have a usable amount of history, so training
    starts at min_train_frac of the data and the remainder is divided into k
    equal test blocks.

    Yields (train_end, test_end) index pairs; train is [0, train_end).
    """
    start = int(round(n * min_train_frac))
    block = (n - start) // k
    if block < 20:
        raise SystemExit(
            f"{k} folds leaves only {block} days per test block. "
            f"Use fewer folds, or a longer series.")
    for i in range(k):
        train_end = start + i * block
        test_end = train_end + block if i < k - 1 else n
        yield train_end, test_end


def fit_models(Xtr, ytr):
    """Train the three learned models on one fold's training block.

    XGBoost needs a validation slice to stop on. It is taken as the LAST 15% of
    the training block, not a random sample, so it stays after everything the
    model trains on and before everything it is tested on.
    """
    cut = int(len(Xtr) * 0.85)
    Xf, yf = Xtr.iloc[:cut], ytr.iloc[:cut]
    Xv, yv = Xtr.iloc[cut:], ytr.iloc[cut:]

    probe = XGBRegressor(n_estimators=2000, learning_rate=0.05, max_depth=4,
                         subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                         random_state=SEED, n_jobs=1,
                         early_stopping_rounds=50, eval_metric="mae")
    probe.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
    best_n = probe.best_iteration + 1

    return {
        "Ridge regression": Ridge(alpha=1.0).fit(Xtr, ytr),
        "Random Forest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2,
            random_state=SEED, n_jobs=1).fit(Xtr, ytr),
        "XGBoost": XGBRegressor(
            n_estimators=best_n, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=SEED, n_jobs=1).fit(Xtr, ytr),
    }, best_n


def main(path, outdir, k):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    raw = load(path)
    feat = make_features(raw)
    assert_no_leakage(raw, feat)
    n = len(feat)
    print(f"{n} rows after features | {k} expanding-window folds\n")

    drop = ["date", TARGET]
    rows = []

    for i, (train_end, test_end) in enumerate(folds(n, k), start=1):
        tr = feat.iloc[:train_end]
        te = feat.iloc[train_end:test_end]

        Xtr, ytr = tr.drop(columns=drop), tr[TARGET]
        Xte, yte = te.drop(columns=drop), te[TARGET]

        preds = {BASELINE: te["lag_7"].to_numpy()}
        models, best_n = fit_models(Xtr, ytr)
        for name, m in models.items():
            preds[name] = m.predict(Xte)

        print(f"fold {i}: train {len(tr):>4} days -> test {len(te):>3} days "
              f"({te.date.min().date()} to {te.date.max().date()}), "
              f"xgb n={best_n}")

        for name, p in preds.items():
            rows.append({
                "fold": i,
                "model": name,
                "train_days": len(tr),
                "test_days": len(te),
                "test_from": str(te.date.min().date()),
                "MAE": float(np.mean(np.abs(p - yte.to_numpy()))),
                "MAPE_%": mape(yte, p),
            })

    per_fold = pd.DataFrame(rows)

    # Improvement over the baseline, computed WITHIN each fold - a baseline that
    # happens to be easy in one window must not flatter another window's model.
    base = (per_fold[per_fold.model == BASELINE]
            .set_index("fold")["MAE"].rename("base_MAE"))
    per_fold = per_fold.join(base, on="fold")
    per_fold["vs_baseline_%"] = ((per_fold["base_MAE"] - per_fold["MAE"])
                                 / per_fold["base_MAE"] * 100)

    # Win rate: in how many folds did this model beat the baseline?
    wins = (per_fold.assign(won=per_fold["MAE"] < per_fold["base_MAE"])
                    .groupby("model")["won"].mean() * 100)

    summary = (per_fold.groupby("model")
               .agg(mean_MAE=("MAE", "mean"),
                    std_MAE=("MAE", "std"),
                    worst_MAE=("MAE", "max"),
                    mean_MAPE=("MAPE_%", "mean"),
                    mean_vs_baseline=("vs_baseline_%", "mean"))
               .join(wins.rename("beat_baseline_%"))
               .sort_values("mean_MAE")
               .round(3)
               .reset_index())

    print("\n" + "=" * 78)
    print(f"PER FOLD (MAE, kWh/day)")
    print("=" * 78)
    print(per_fold.pivot(index="model", columns="fold", values="MAE")
                  .round(3).to_string())

    print("\n" + "=" * 78)
    print(f"ACROSS {k} FOLDS")
    print("=" * 78)
    print(summary.to_string(index=False))

    best = summary.iloc[0]
    print(f"\nLowest mean MAE: {best['model']} at {best['mean_MAE']:.3f} "
          f"+/- {best['std_MAE']:.3f} kWh, beating the baseline in "
          f"{best['beat_baseline_%']:.0f}% of folds.")
    if best["std_MAE"] > best["mean_MAE"] * 0.25:
        print("The spread is wide relative to the mean, so quote the range "
              "rather than the single figure.")

    per_fold.round(4).to_csv(outdir / "rolling_folds.csv", index=False)
    summary.to_csv(outdir / "rolling_summary.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        for name, g in per_fold.groupby("model"):
            style = dict(lw=2.2, ls="--") if name == BASELINE else dict(lw=1.6)
            ax.plot(g["fold"], g["MAE"], marker="o", label=name, **style)
        ax.set_xlabel("fold (origin moving forward in time)")
        ax.set_ylabel("MAE (kWh/day)")
        ax.set_title("Rolling-origin validation")
        ax.set_xticks(range(1, k + 1))
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / "rolling_plot.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"(plot skipped: {e})")

    print(f"\nwrote -> {outdir}/rolling_folds.csv, rolling_summary.csv, "
          f"rolling_plot.png")
    return per_fold, summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    main(a.data, a.outdir, a.folds)

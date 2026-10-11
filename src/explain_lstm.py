"""Explainability for the LSTM: which inputs it uses, how far back it looks, and how that compares to XGBoost.

Run from the repo root:
    python -m src.explain_lstm
    python -m src.explain_lstm --shap-samples 200    # quicker, noisier SHAP values

Build the daily table first:
    python -m src.build_daily --raw data/household_power_consumption.txt

The LSTM is retrained exactly as in src/models/lstm.py (same days, windows, scalers, 5 seeds), and what is
explained is the average of the 5 copies: the forecast that goes in the comparison table.

Two methods, because each answers something the other can't:

1. Permutation importance (model-agnostic). Shuffle one input across the test days and measure how much the
   test MAE rises, in kWh. It explains the ensemble directly and needs no assumptions about the network.
   - By input group: a group's whole 14-day history is shuffled together, so a test day gets another
     day's history for that input.
   - By day of the window (t-1 ... t-14): only the meter readings on that one day are shuffled, so the
     answer is "how much does the forecast rely on readings from k days ago".
   kwh and intensity are near-duplicates (kWh is about voltage x current x 24h), so the model can read one
   when the other is shuffled. They are also shuffled together as "total load", which is the fairer figure.

2. SHAP, using GradientExplainer (expected gradients). One contribution per test day, window day and input,
   in kWh, so it gives direction and single-day explanations. Values are computed on each copy and averaged,
   which is valid because the ensemble forecast is the average of the copies. Expected gradients are
   estimated by sampling, so unlike TreeSHAP they only approximately add up to the prediction; the gap is
   reported rather than hidden.

Writes results/lstm_permutation_importance.csv   MAE increase per input group
       results/lstm_lookback_importance.csv      importance per window day, both methods
       results/lstm_shap_summary.csv             mean |SHAP| and direction per input group
       results/lstm_shap_daily.csv               SHAP per test day and input group (dashboard's household view)
       results/lstm_drivers.md                   the same in plain English, with the XGBoost comparison
       results/lstm_permutation_importance.png
       results/lstm_lookback.png
       results/lstm_shap_waterfall_{highest,lowest}_day.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import SEED
from src.models import lstm

RESULTS_DIR = Path("results")
XGB_SHAP_PATH = RESULTS_DIR / "shap_summary.csv"     # written by src/explain.py

N_REPEATS = 10        # shuffles per input; the reported figure is their mean, the spread their std
N_BACKGROUND = 200    # training windows SHAP integrates from

# Input groups, in plain English. Every step feature belongs to exactly one group (checked in group_index).
# Calendar columns are grouped by meaning, not by position: the window is 14 consecutive rows, so the window
# days' calendar already gives away the target day's (next_* in lstm.step_features), and splitting them
# would divide one piece of information between two groups.
GROUPS = {
    "kwh": "daily consumption (kWh)",
    "intensity": "mean current",
    "reactive_kvarh": "reactive energy",
    "voltage": "mean voltage",
    "sub1_wh": "kitchen circuit",
    "sub2_wh": "laundry circuit",
    "sub3_wh": "water heater and A/C",
    "other_wh": "unmetered load (residual)",
    "day_of_week": "day of week",
    "time_of_year": "time of year",
}
UNITS = {"kwh": "kWh", "intensity": "A", "reactive_kvarh": "kvarh", "voltage": "V",
         "sub1_wh": "Wh", "sub2_wh": "Wh", "sub3_wh": "Wh", "other_wh": "Wh"}
CALENDAR_GROUPS = {"day_of_week": ["dow_sin", "dow_cos", "weekend"], "time_of_year": ["doy_sin", "doy_cos"]}
GROUP_MEMBERS = {
    **{m: [m] for m in lstm.MEASURES},
    **{g: cols + [f"next_{c}" for c in cols] for g, cols in CALENDAR_GROUPS.items()},
}
# Shuffled together as well as separately, because each can stand in for the other.
COMBINED = {"total_load": ("total load (kWh and current together)", ["kwh", "intensity"])}

# XGBoost feature -> LSTM group, for the side-by-side ranking. lag_* and roll_* are built from kwh.
def xgb_group(feature: str) -> str:
    for m in lstm.MEASURES:
        if feature.startswith(m + "_"):
            return m
    if feature.startswith(("lag_", "roll_")):
        return "kwh"
    if feature.startswith("dow") or feature == "is_weekend":
        return "day_of_week"
    if feature.startswith("doy") or feature == "month":
        return "time_of_year"
    return feature


def group_index(feature_names: list[str]) -> dict[str, list[int]]:
    """Column positions of each group's features, checking the groups cover every feature exactly once."""
    pos = {f: i for i, f in enumerate(feature_names)}
    index = {g: [pos[f] for f in members] for g, members in GROUP_MEMBERS.items()}
    used = sorted(i for cols in index.values() for i in cols)
    assert used == list(range(len(feature_names))), \
        f"input groups don't cover the LSTM's features exactly once: {feature_names}"
    return index


def ensemble_predict_kwh(models, y_scaler, X) -> np.ndarray:
    """The LSTM's forecast in kWh: the average of the copies, unscaled (same as lstm.run)."""
    scaled = np.mean([m.predict(X, batch_size=256, verbose=0).ravel() for m in models], axis=0)
    return y_scaler.inverse_transform(scaled.reshape(-1, 1)).ravel()


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def permutation_importance(predict, X, y, column_sets: dict, steps=None, repeats=N_REPEATS, seed=SEED):
    """MAE increase (kWh) when the given columns are shuffled across samples.

    column_sets maps a name to feature positions. steps limits the shuffle to those window positions
    (default: the whole window). Rows are shuffled as blocks, so each sample receives another sample's
    values for those columns and steps, keeping the values themselves realistic.
    """
    rng = np.random.default_rng(seed)
    base = mae(y, predict(X))
    every = np.arange(len(X))
    steps = np.arange(X.shape[1]) if steps is None else np.asarray(steps)
    rows = []
    for name, cols in column_sets.items():
        rises = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[np.ix_(every, steps, cols)] = X[np.ix_(rng.permutation(len(X)), steps, cols)]
            rises.append(mae(y, predict(Xp)) - base)
        rows.append({"input": name, "mae_increase_kwh": float(np.mean(rises)),
                     "std_kwh": float(np.std(rises))})
    return pd.DataFrame(rows), base


def gradient_shap_kwh(models, y_scaler, X_background, X_explain, nsamples: int, seed=SEED) -> np.ndarray:
    """Expected-gradients SHAP values in kWh, shape (days, window, features), averaged over the copies."""
    import shap

    per_copy = []
    for i, m in enumerate(models):
        explainer = shap.GradientExplainer(m, X_background)
        sv = np.asarray(explainer.shap_values(X_explain, nsamples=nsamples, rseed=seed + i))
        per_copy.append(sv.reshape(X_explain.shape))
    # StandardScaler is linear, so scaled contributions times its scale are exact kWh contributions.
    return np.mean(per_copy, axis=0) * float(y_scaler.scale_[0])


def shap_by_group(sv: np.ndarray, groups: dict[str, list[int]]) -> pd.DataFrame:
    """(days, groups) contributions in kWh, each group summed over its features and the whole window."""
    return pd.DataFrame({g: sv[:, :, cols].sum(axis=(1, 2)) for g, cols in groups.items()})


def shap_summary(contrib: pd.DataFrame, raw_X: np.ndarray, groups: dict[str, list[int]]) -> pd.DataFrame:
    """Mean |SHAP| per group, and direction: correlation between the group's input on the last window day
    (yesterday, in its own units) and the group's contribution. Same thresholds as src/explain.py."""
    rows = []
    for g, cols in groups.items():
        c = contrib[g].to_numpy()
        x = raw_X[:, -1, cols[0]]
        corr = 0.0 if np.std(x) < 1e-12 or np.std(c) < 1e-12 else float(np.corrcoef(x, c)[0, 1])
        direction = ("higher raises the forecast" if corr > 0.2 else
                     "higher lowers the forecast" if corr < -0.2 else "mixed / not monotonic")
        if g in CALENDAR_GROUPS:
            direction = "calendar effect (no single direction)"
        rows.append({"input": g, "plain_name": GROUPS[g],
                     "mean_abs_shap_kwh": float(np.mean(np.abs(c))),
                     "max_abs_shap_kwh": float(np.max(np.abs(c))),
                     "direction_corr": corr, "direction": direction})
    return pd.DataFrame(rows).sort_values("mean_abs_shap_kwh", ascending=False).reset_index(drop=True)


def compare_with_xgb(lstm_summary: pd.DataFrame, path: Path = XGB_SHAP_PATH):
    """Rank of each input group under the LSTM and under XGBoost's TreeSHAP (src/explain.py), or None.

    XGBoost's per-feature mean |SHAP| are added up per group. A sum of means is an upper bound on the
    group's mean |sum|, so treat the XGBoost column as a ranking, not an exact kWh figure."""
    if not path.exists():
        return None
    xgb = pd.read_csv(path)
    xgb = xgb.assign(input=xgb["feature"].map(xgb_group)).groupby("input")["mean_abs_shap_kwh"].sum()
    out = lstm_summary[["input", "plain_name", "mean_abs_shap_kwh"]].rename(
        columns={"mean_abs_shap_kwh": "lstm_kwh"})
    out["xgb_kwh"] = out["input"].map(xgb).fillna(0.0)
    out["lstm_rank"] = out["lstm_kwh"].rank(ascending=False, method="min").astype(int)
    out["xgb_rank"] = out["xgb_kwh"].rank(ascending=False, method="min").astype(int)
    return out


def write_drivers(perm, lookback, summary, comparison, base_kwh, gap, n_days, path, top=6):
    lines = [
        "# What drives the LSTM's daily forecast",
        "",
        f"Model: LSTM, average of {lstm.N_SEEDS} copies, {lstm.WINDOW}-day input window. "
        f"Explained on {n_days} held-out test days.",
        "",
        "## Which inputs it relies on (permutation importance)",
        "",
        "How much the test MAE rises, in kWh, when one input's history is shuffled across days. "
        "Bigger means the forecast depends on it more.",
        "",
    ]
    for i, r in perm.head(top).iterrows():
        lines.append(f"{i + 1}. **{r['plain_name']}**: MAE rises by {r['mae_increase_kwh']:.2f} kWh "
                     f"(± {r['std_kwh']:.2f}).")
    t = lookback.sort_values("perm_mae_increase_kwh", ascending=False).iloc[0]
    lines += [
        "",
        "## How far back it looks",
        "",
        f"Shuffling only the meter readings from one day of the window hurts most for **{t['day']}** "
        f"(MAE +{t['perm_mae_increase_kwh']:.2f} kWh). See `lstm_lookback.png` for every day.",
        "",
        "## Direction and size (SHAP)",
        "",
        f"Average forecast over the background days: **{base_kwh:.2f} kWh**. Each figure is how far an input "
        "(its whole 14-day history) moves a day's forecast away from that average, averaged over the test days.",
        "",
    ]
    for i, r in summary.head(top).iterrows():
        lines.append(f"{i + 1}. **{r['plain_name']}**: {r['mean_abs_shap_kwh']:.2f} kWh on an average day, "
                     f"up to {r['max_abs_shap_kwh']:.2f} kWh. {r['direction'].capitalize()}.")
    lines += [
        "",
        f"Additivity check: base value plus the SHAP values is within {gap['mean']:.2f} kWh of the forecast on "
        f"an average day (worst {gap['max']:.2f} kWh), against a typical {gap['spread']:.2f} kWh gap between "
        "forecast and base value. Expected gradients are sampled, so this is approximate, unlike TreeSHAP.",
        "",
    ]
    if comparison is not None:
        lines += ["## Compared with XGBoost", "",
                  "| Input | LSTM rank | XGBoost rank |", "|---|---|---|"]
        for _, r in comparison.sort_values("lstm_rank").iterrows():
            lines.append(f"| {r['plain_name']} | {r['lstm_rank']} | {r['xgb_rank']} |")
        agree = comparison.nsmallest(3, "lstm_rank")["input"].isin(comparison.nsmallest(3, "xgb_rank")["input"])
        lines += ["", f"{int(agree.sum())} of the LSTM's top 3 inputs are also in XGBoost's top 3. XGBoost's "
                  "ranking adds up its features per group (from `shap_summary.csv`), so compare ranks, not kWh. "
                  "Mean current and daily consumption carry the same information, so a model can lean on either: "
                  "read their ranks together.", ""]
    lines += [
        "## Reading this",
        "",
        "- Permutation importance says what the model *uses*; SHAP adds *which way* and *by how much* per day.",
        "- Daily consumption and mean current carry almost the same information, so either alone looks less "
        "important than both together (see *total load*).",
        "- `other_wh` is a residual (total minus the three sub-meters), not a measured appliance group.",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def plots(perm, lookback, contrib, base_kwh, days, actual, raw_X, groups, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = perm.sort_values("mae_increase_kwh")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(p["plain_name"], p["mae_increase_kwh"], xerr=p["std_kwh"], color="#4C72B0")
    ax.set_xlabel("Test MAE increase when shuffled (kWh)")
    ax.set_title("LSTM permutation importance")
    fig.tight_layout(); fig.savefig(outdir / "lstm_permutation_importance.png", dpi=150); plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.bar(lookback["day"], lookback["perm_mae_increase_kwh"], color="#4C72B0", label="permutation")
    ax1.set_ylabel("MAE increase when shuffled (kWh)")
    ax1.set_xlabel("Day of the input window (t-1 = the day before the forecast)")
    ax2 = ax1.twinx()
    ax2.plot(lookback["day"], lookback["mean_abs_shap_kwh"], color="#DD8452", marker="o", label="mean |SHAP|")
    ax2.set_ylabel("mean |SHAP| (kWh)")
    ax1.set_title("How far back the LSTM looks (meter readings only)")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.88))
    fig.tight_layout(); fig.savefig(outdir / "lstm_lookback.png", dpi=150); plt.close(fig)

    try:
        import shap
    except ImportError:
        return
    for label, idx in (("highest_day", int(np.argmax(actual))), ("lowest_day", int(np.argmin(actual)))):
        # Label meter inputs with yesterday's reading; a calendar group has no single value worth showing.
        names = [GROUPS[g] if g in CALENDAR_GROUPS else
                 f"{GROUPS[g]} (yesterday {raw_X[idx, -1, cols[0]]:,.1f} {UNITS[g]})"
                 for g, cols in groups.items()]
        ex = shap.Explanation(values=contrib.iloc[idx].to_numpy(), base_values=base_kwh, feature_names=names)
        shap.plots.waterfall(ex, max_display=12, show=False)
        plt.gcf().set_size_inches(11, 6)
        plt.title(f"LSTM forecast for {pd.Timestamp(days[idx]).date()} (actual {actual[idx]:.1f} kWh)")
        plt.tight_layout(); plt.savefig(outdir / f"lstm_shap_waterfall_{label}.png", dpi=150); plt.close()


def main(outdir: Path, shap_samples: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    print("Training the LSTM (same protocol as src/models/lstm.py) ...")
    result = lstm.run(verbose=False)
    models, y_scaler = result["models"], result["y_scaler"]
    X_te, y_te, names = result["X_test"], result["actual"], result["feature_names"]
    groups = group_index(names)

    def predict(X):
        return ensemble_predict_kwh(models, y_scaler, X)

    # Raw (unscaled) windows for the same test days, to show inputs in their own units.
    daily = lstm.load_daily()
    _, _, test_rows = lstm.team_split(daily)
    raw_X, _ = lstm.make_windows(lstm.step_features(daily).to_numpy(), daily[lstm.TARGET].to_numpy(), test_rows)

    check = mae(result["predicted"], predict(X_te))
    assert check < 1e-3, f"re-predicting the test days doesn't reproduce the LSTM's forecast ({check})"
    print(f"LSTM test MAE {result['lstm']['mae']:.3f} kWh on {len(y_te)} days")

    # 1. Permutation importance, by input group and by window day.
    column_sets = dict(groups)
    for name, (_, members) in COMBINED.items():
        column_sets[name] = [i for m in members for i in groups[m]]
    perm, base_mae = permutation_importance(predict, X_te, y_te, column_sets)
    labels = {**GROUPS, **{k: v[0] for k, v in COMBINED.items()}}
    perm.insert(1, "plain_name", perm["input"].map(labels))
    perm = perm.sort_values("mae_increase_kwh", ascending=False).reset_index(drop=True)
    perm.round(4).to_csv(outdir / "lstm_permutation_importance.csv", index=False)

    measure_cols = [i for m in lstm.MEASURES for i in groups[m]]
    lookback_rows = []
    for k in range(1, lstm.WINDOW + 1):
        step = lstm.WINDOW - k                       # position in the window; the last one is t-1
        r, _ = permutation_importance(predict, X_te, y_te, {f"t-{k}": measure_cols}, steps=[step])
        lookback_rows.append({"day": f"t-{k}", "perm_mae_increase_kwh": r.loc[0, "mae_increase_kwh"],
                              "perm_std_kwh": r.loc[0, "std_kwh"]})
    lookback = pd.DataFrame(lookback_rows)
    print("\nPermutation importance (MAE increase, kWh):")
    print(perm[["plain_name", "mae_increase_kwh", "std_kwh"]].round(3).to_string(index=False))

    # 2. SHAP (expected gradients), averaged over the copies.
    rng = np.random.default_rng(SEED)
    X_fit = result["X_fit"]
    background = X_fit[rng.choice(len(X_fit), size=min(N_BACKGROUND, len(X_fit)), replace=False)]
    print(f"\nSHAP: {len(models)} copies x {len(X_te)} days x {shap_samples} samples ...")
    sv = gradient_shap_kwh(models, y_scaler, background, X_te, shap_samples)
    base_kwh = float(predict(background).mean())

    preds = predict(X_te)
    gap_days = np.abs(base_kwh + sv.sum(axis=(1, 2)) - preds)
    gap = {"mean": float(gap_days.mean()), "max": float(gap_days.max()),
           "spread": float(np.abs(preds - base_kwh).mean())}
    print(f"  additivity gap: mean {gap['mean']:.3f} kWh, worst {gap['max']:.3f} kWh "
          f"(typical forecast - base: {gap['spread']:.3f} kWh)")

    contrib = shap_by_group(sv, groups)
    # One row per test day, for the dashboard's household view. "unexplained_kwh" is the additivity gap, so
    # base + every contribution + unexplained adds up to the forecast exactly.
    per_day = contrib.copy()
    per_day.insert(0, "date", pd.DatetimeIndex(result["days"]).date)
    per_day.insert(1, "actual_kwh", y_te)
    per_day.insert(2, "predicted_kwh", preds)
    per_day.insert(3, "base_kwh", base_kwh)
    per_day["unexplained_kwh"] = preds - base_kwh - contrib.sum(axis=1).to_numpy()
    per_day.round(4).to_csv(outdir / "lstm_shap_daily.csv", index=False)

    summary = shap_summary(contrib, raw_X, groups)
    summary.round(4).to_csv(outdir / "lstm_shap_summary.csv", index=False)
    lookback["mean_abs_shap_kwh"] = [float(np.abs(sv[:, lstm.WINDOW - k, measure_cols].sum(axis=1)).mean())
                                     for k in range(1, lstm.WINDOW + 1)]
    lookback.round(4).to_csv(outdir / "lstm_lookback_importance.csv", index=False)

    comparison = compare_with_xgb(summary)
    write_drivers(perm, lookback, summary, comparison, base_kwh, gap, len(y_te), outdir / "lstm_drivers.md")
    plots(perm, lookback, contrib, base_kwh, result["days"], y_te, raw_X, groups, outdir)

    print("\nSHAP, mean |contribution| (kWh):")
    print(summary[["plain_name", "mean_abs_shap_kwh", "direction"]].round(3).to_string(index=False))
    if comparison is not None:
        print("\nRank vs XGBoost:")
        print(comparison.sort_values("lstm_rank")[["plain_name", "lstm_rank", "xgb_rank"]].to_string(index=False))
    print(f"\nwrote -> {outdir}/lstm_permutation_importance.csv, lstm_lookback_importance.csv, "
          "lstm_shap_summary.csv, lstm_shap_daily.csv, lstm_drivers.md, lstm_*.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--shap-samples", type=int, default=500,
                    help="expected-gradients samples per test day (more = closer additivity, slower)")
    a = ap.parse_args()
    main(a.outdir, a.shap_samples)

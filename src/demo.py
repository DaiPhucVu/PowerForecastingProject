"""
Live demonstration driver.

Everything the panel sees runs from artefacts already on disk - saved models,
result files, the manifest. Nothing here reparses the two million raw rows, so
each section returns in under a few seconds instead of five minutes.

Run one section at a time so you can talk between them:

    python demo.py provenance     where the numbers come from
    python demo.py forecast       the product: tomorrow, and the next 14 days
    python demo.py accuracy       is it good enough, and how sure are we
    python demo.py leakage        proof the guard actually fires
    python demo.py all            every section, back to back

Add --check to verify the demo will work before you stand up.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The repository root is the script's parent directory, unless you are
# running from somewhere that already has a data/ folder (Colab, a subfolder) -
# then trust where you are standing.
_here = Path.cwd()
ROOT = _here if (_here / "data").is_dir() else Path(__file__).resolve().parents[1]
DATA, RESULTS, MODELS = ROOT / "data", ROOT / "results", ROOT / "models"

W = 72
def rule(ch="="):  print(ch * W)
def head(t):       print(); rule(); print(f"  {t.upper()}"); rule()
def item(k, v):    print(f"  {k:<34} {v}")
def note(*lines):
    print()
    for t in lines:
        print(f"  {t}")


def need(path, how):
    if not Path(path).exists():
        print(f"\n  MISSING: {path}")
        print(f"  Run first:  {how}\n")
        sys.exit(1)
    return Path(path)


# ------------------------------------------------------------- provenance
def provenance():
    head("1. Where these numbers come from")
    man = json.loads(need(DATA / "manifest.json",
                          "python src/build_daily.py --raw raw/...txt").read_text())

    item("Source", "UCI household power consumption")
    item("Raw readings", f'{man["raw_rows"]:,} minute-level rows')
    item("Raw file SHA-256", man["raw_sha256"][:32] + "...")
    item("Usable days after cleaning", f'{man["daily_rows"]:,}')
    item("Covering", f'{man["date_from"]} to {man["date_to"]}')
    item("Coverage rule",
         f'keep a day only if >= {man.get("min_coverage_rule", 0.9):.0%} '
         f'of its 1,440 minutes recorded')
    item("Partial days dropped", man.get("days_dropped_partial", "see manifest"))

    note("The checksum is the point. Anyone can rebuild this table from the",
         "original file and confirm they got the same thing we did.")


# --------------------------------------------------------------- forecast
def forecast():
    head("2. The product - a forecast for days that have not happened")
    need(MODELS / "next_day.joblib",
         "python src/train_model.py --data data/household_daily.csv")
    fc = pd.read_csv(need(RESULTS / "forecast.csv",
                          "python src/predict.py --data data/household_daily.csv"))

    import joblib
    meta = joblib.load(MODELS / "next_day.joblib")["meta"]
    item("Model", f'XGBoost, {meta["n_features"]} features, '
                  f'{meta["n_estimators"]} trees')
    item("Trained on", f'{meta["trained_rows"]:,} days to {meta["trained_to"]}')

    r = fc.iloc[0]
    print()
    rule("-")
    print(f"  TOMORROW ({r['date']}):   {r['predicted_kwh']:.2f} kWh"
          f"   (+/- {r['expected_MAE']:.2f})")
    rule("-")

    print("\n  Next seven days")
    for _, x in fc.head(7).iterrows():
        bar = "#" * int(round(x["predicted_kwh"] / 1.2))
        print(f"   {x['date']}  {x['predicted_kwh']:6.2f} kWh  {bar}")

    mth = RESULTS / "forecast_monthly.csv"
    if mth.exists():
        m = pd.read_csv(mth)
        print("\n  Rolled up to months - the level the client plans at")
        for _, x in m.iterrows():
            flag = "" if x["complete"] else "   (partial month)"
            print(f"   {x['month']}  {x['predicted_kwh']:8.1f} kWh"
                  f"   range {x['lower_kwh']:.0f}-{x['upper_kwh']:.0f}{flag}")

    note("Day one uses the full model. Everything after is recursive, so the",
         "band widens with the square root of the step count - that is an",
         "estimate of accumulating error, not a measured interval.")


# --------------------------------------------------------------- accuracy
def accuracy():
    head("3. Is it good enough, and how sure are we")

    cmp_ = pd.read_csv(need(RESULTS / "comparison_table.csv",
                            "python src/forecast_pipeline.py --data data/household_daily.csv"))
    print("  Single held-out split")
    for _, r in cmp_.sort_values("MAE").iterrows():
        print(f"   {str(r['model']):<28} MAE {r['MAE']:6.3f}"
              f"   MAPE {r['MAPE_%']:5.1f}%")

    roll = RESULTS / "rolling_summary.csv"
    if roll.exists():
        rs = pd.read_csv(roll)
        print("\n  Mean of five rolling-origin folds - the window moves through time")
        for _, r in rs.iterrows():
            print(f"   {str(r['model']):<28} MAE {r['mean_MAE']:6.3f}"
                  f" +/- {r['std_MAE']:.3f}"
                  f"   beat baseline in {r['beat_baseline_%']:.0f}% of folds")

        ml = rs[~rs.model.str.contains("naive", case=False)]
        spread = ml.mean_MAE.max() - ml.mean_MAE.min()
        noise = ml.std_MAE.mean()
        print()
        rule("-")
        print(f"  Spread between models {spread:.3f} kWh."
              f"   Fold-to-fold noise {noise:.3f} kWh.")
        if spread < 0.25 * noise:
            print("  The models are indistinguishable. The robust finding is the")
            print("  improvement over the baseline, not a ranking.")
        rule("-")

    mvd = RESULTS / "monthly_vs_daily.csv"
    if mvd.exists():
        m = pd.read_csv(mvd).set_index("level")
        print(f"\n  Daily error   {m.loc['daily','MAPE_%']:.1f}% MAPE")
        print(f"  Monthly error {m.loc['monthly','MAPE_%']:.1f}% MAPE"
              f"   <- daily errors partly cancel when summed")


# ---------------------------------------------------------------- leakage
def leakage():
    head("4. Proof the leakage guard actually fires")
    sys.path.insert(0, str(ROOT / "src"))
    import forecast_pipeline as fp

    daily = pd.read_csv(need(DATA / "household_daily.csv",
                             "python src/build_daily.py --raw raw/...txt"),
                        parse_dates=["date"])
    feat = fp.make_features(daily)

    print("  Honest feature table:")
    fp.assert_no_leakage(daily, feat)
    print("    -> passes\n")

    print("  Now I smuggle in today's voltage - a column we only know once the")
    print("  day is over. If the guard is real, this must fail.\n")
    tampered = feat.copy()
    tampered["voltage"] = 240.0
    try:
        fp.assert_no_leakage(daily, tampered)
        print("    -> NO ERROR. The guard did not fire. That would be a bug.")
    except AssertionError as e:
        print(f"    -> AssertionError: {e}")
        print("\n    The pipeline refuses to run. This assertion executes on")
        print("    every build, and a test breaks it deliberately to confirm it.")


# ------------------------------------------------------------- pre-flight
def check():
    head("Pre-flight")
    wanted = [
        (DATA / "manifest.json", "build_daily.py"),
        (DATA / "household_daily.csv", "build_daily.py"),
        (RESULTS / "comparison_table.csv", "forecast_pipeline.py"),
        (RESULTS / "rolling_summary.csv", "rolling_validation.py"),
        (RESULTS / "monthly_vs_daily.csv", "monthly_report.py"),
        (MODELS / "next_day.joblib", "train_model.py"),
        (RESULTS / "forecast.csv", "predict.py"),
        (RESULTS / "dashboard.html", "build_dashboard.py"),
    ]
    missing = []
    for p, how in wanted:
        ok = p.exists()
        print(f"  [{'ok' if ok else '  '}] {p.relative_to(ROOT)}"
              f"{'' if ok else '      <- run ' + how}")
        if not ok:
            missing.append(how)

    print()
    if missing:
        print(f"  NOT READY. Run: {', '.join(dict.fromkeys(missing))}")
        sys.exit(1)
    print("  Ready. Every section will run from disk in seconds.")


SECTIONS = {"provenance": provenance, "forecast": forecast,
            "accuracy": accuracy, "leakage": leakage}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("section", nargs="?", default="all",
                    choices=list(SECTIONS) + ["all"])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    if a.check:
        check()
    elif a.section == "all":
        for fn in SECTIONS.values():
            fn()
        print()
    else:
        SECTIONS[a.section]()
        print()

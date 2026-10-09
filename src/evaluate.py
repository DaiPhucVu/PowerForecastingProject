"""Compare models. Writes results/comparison_table.csv and results/run_metadata.json.

Run from the repo root, after the models have written their predictions:
    python -m src.evaluate

It scores every results/<model>_test_predictions.csv (the shared format, see README, "Dashboard").
Unlike the dashboard, which only warns, it refuses to build the table if any model was tested on
other days or other actual values than the pipeline's test split, because those scores wouldn't
be comparable.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src import forecast_pipeline as pipeline
from src.config import SEED

DAILY_PATH = Path("data/household_daily.csv")   # written by src/build_daily.py
RESULTS_DIR = Path("results")
PATTERN = "*_test_predictions.csv"
NAIVE = "naive"           # the baseline every model is measured against
TABLE_PATH = RESULTS_DIR / "comparison_table.csv"
METADATA_PATH = RESULTS_DIR / "run_metadata.json"
PIPELINE_METADATA_PATH = RESULTS_DIR / "pipeline_metadata.json"   # written by src/forecast_pipeline.py


def sha256_12(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def team_test_days(path: Path = DAILY_PATH) -> pd.DataFrame:
    """The pipeline's test days and their actual kWh, from its own make_features + chrono_split."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Build it first: "
                                "python -m src.build_daily --raw data/household_power_consumption.txt")
    _, _, test = pipeline.chrono_split(pipeline.make_features(pipeline.load(path)))
    return test[["date", pipeline.TARGET]].rename(columns={pipeline.TARGET: "actual_kwh"}).reset_index(drop=True)


def load_predictions(path: Path, expected: pd.DataFrame) -> tuple[str, pd.Series]:
    """One model's test forecasts, checked against the team's test days and actual values."""
    df = pd.read_csv(path, parse_dates=["date"])
    pred_cols = [c for c in df.columns if c.endswith("_kwh") and c != "actual_kwh"]
    if "date" not in df.columns or "actual_kwh" not in df.columns or len(pred_cols) != 1:
        raise ValueError(f"{path.name}: needs columns date, actual_kwh and exactly one <model>_kwh "
                         f"(found {list(df.columns)})")
    model = pred_cols[0].removesuffix("_kwh")

    df = df.sort_values("date").reset_index(drop=True)
    if not df["date"].equals(expected["date"]):
        raise ValueError(f"{path.name}: tested on {len(df)} days from {df.date.min().date()}, but the team's "
                         f"test split is {len(expected)} days from {expected.date.min().date()}. "
                         "Re-run the model on the current daily table.")
    worst = (df["actual_kwh"] - expected["actual_kwh"]).abs().max()
    if worst > 1e-6:
        raise ValueError(f"{path.name}: actual values differ from the daily table by up to {worst:.3f} kWh. "
                         "It was probably built from a different daily table.")
    if df[pred_cols[0]].isna().any():
        raise ValueError(f"{path.name}: {df[pred_cols[0]].isna().sum()} test days have no forecast.")
    return model, df[pred_cols[0]]


def score(y, p) -> dict:
    return {"MAE": mean_absolute_error(y, p),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAPE": pipeline.mape(y, p)}


def main() -> None:
    expected = team_test_days()
    files = sorted(RESULTS_DIR.glob(PATTERN))
    loaded = [(load_predictions(f, expected), f) for f in files]
    predictions = {model: p for (model, p), _ in loaded}
    if NAIVE not in predictions:
        raise FileNotFoundError(f"No {RESULTS_DIR / (NAIVE + '_test_predictions.csv')}, so there is no baseline "
                                "to compare against. Run python -m src.models.naive first.")

    y = expected["actual_kwh"]
    table = pd.DataFrame([{"model": m, **score(y, p)} for m, p in predictions.items()])
    naive_mae = table.loc[table.model == NAIVE, "MAE"].item()
    table["vs_naive_%"] = ((naive_mae - table["MAE"]) / naive_mae * 100).round(1)
    table = table.round({"MAE": 3, "RMSE": 3, "MAPE": 3}).sort_values("MAE").reset_index(drop=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(TABLE_PATH, index=False)
    METADATA_PATH.write_text(json.dumps({
        "data_file": DAILY_PATH.name,
        "data_sha256_12": sha256_12(DAILY_PATH),
        "seed": SEED,
        "test_days": len(expected),
        "test_start": str(expected.date.min().date()),
        "test_end": str(expected.date.max().date()),
        "baseline": NAIVE,
        "prediction_files": {model: {"file": f.name, "sha256_12": sha256_12(f)} for (model, _), f in loaded},
        # split sizes and XGBoost's tree count, if the pipeline has been run
        "pipeline": (json.loads(PIPELINE_METADATA_PATH.read_text())
                     if PIPELINE_METADATA_PATH.exists() else None),
    }, indent=2))

    print(f"COMPARISON  (test = {len(expected)} days, {expected.date.min().date()} to "
          f"{expected.date.max().date()}, target = daily kWh)")
    print(table.to_string(index=False))
    print(f"\nwrote -> {TABLE_PATH}, {METADATA_PATH}")


if __name__ == "__main__":
    main()

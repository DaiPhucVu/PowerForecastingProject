"""GRU forecaster for daily household energy (kWh): the LSTM with its recurrent layer swapped for a GRU.

Run from the repo root:
    python -m src.models.gru

Build the daily table first:
    python -m src.build_daily --raw data/household_power_consumption.txt

Everything except the layer comes from src/models/lstm.py, so the two are a like-for-like comparison:
the same daily table, train / val / test days, input windows, leakage checks, scaling, epoch search on
val, refit on train+val, and the average of N_SEEDS copies. See that file for how each rule is kept.

Writes results/gru_test_predictions.csv in the shared prediction format (see README, "Dashboard").
"""

from pathlib import Path

import keras
import pandas as pd

from src.models.lstm import LSTMForecaster, run

PREDICTIONS_PATH = Path("results/gru_test_predictions.csv")


class GRUForecaster(LSTMForecaster):
    NAME = "GRU"
    LAYER = keras.layers.GRU


def main() -> None:
    result = run(forecaster=GRUForecaster)
    PREDICTIONS_PATH.parent.mkdir(exist_ok=True)
    pd.DataFrame({"date": result["days"], "actual_kwh": result["actual"], "gru_kwh": result["predicted"]}
                 ).to_csv(PREDICTIONS_PATH, index=False)
    print(f"\nSaved test predictions to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()

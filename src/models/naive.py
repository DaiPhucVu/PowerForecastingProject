"""Seasonal naive baseline: tomorrow's kWh = the value 7 rows earlier (the notebook's "naive (lag7)").

Run from the repo root: python -m src.models.naive
Writes results/naive_test_predictions.csv in the shared prediction format (see README, "Dashboard").

It scores exactly the pipeline's test days, because it uses the same table and split helpers as the LSTM
(which take them from src/forecast_pipeline.py).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.lstm import TARGET, load_daily, team_split

PREDICTIONS_PATH = Path("results/naive_test_predictions.csv")
SEASON = 7   # rows, like the team's lag_7


def main() -> None:
    daily = load_daily()
    kwh = daily[TARGET]
    _, _, test_rows = team_split(daily)

    predicted = kwh.shift(SEASON).to_numpy()[test_rows]
    out = pd.DataFrame({
        "date": daily.index[test_rows],
        "actual_kwh": kwh.to_numpy()[test_rows],
        "naive_kwh": predicted,
    })
    PREDICTIONS_PATH.parent.mkdir(exist_ok=True)
    out.to_csv(PREDICTIONS_PATH, index=False)
    print(f"{len(out)} test days, MAE {np.abs(out.actual_kwh - out.naive_kwh).mean():.3f} kWh -> {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()

"""LSTM forecaster for daily household energy (kWh), predicting the next row of the team's daily table.

Run from the repo root:
    python -m src.models.lstm

It uses the team's data and protocol from notebooks/household_energy_model.ipynb, so its numbers can be
put in the same table as the notebook's models:
- Data: the team's daily table (src/team_daily.py): days with under 90% of minutes are dropped.
- Days scored: the same rows the notebook's feature table keeps (it drops the first 28 rows).
- Split: by row, in time order. First 80% is train+val, last 20% is test; val is the last 20% of train+val.
- Final fit: find the number of epochs on val, then retrain from scratch on train+val (as the notebook does
  for XGBoost) and score on test.
- Baseline: naive = the value 7 rows earlier (the notebook's "naive (lag7)").
- Inputs: same information as the tree models, but as a window of the last WINDOW rows.

How this file keeps future data away from the model:
- The window for a target row holds only earlier rows.
- The only thing the model sees about the row it predicts is its calendar date (day of week, time of
  year). Dates are known in advance and are never computed from energy data.
- Scalers are fitted on the fitting period only.
Each rule is checked by an assertion, so a violation stops the run instead of producing a score that looks
too good.

Like the team's lag features, windows count rows, not calendar days. The table has 7 date gaps, so a few
windows span a gap and "the next row" is sometimes several days later.
"""

from pathlib import Path

import keras
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from src.config import SEED, set_global_seed
from src.team_daily import load_team_daily

TARGET = "kwh"
# What the meter recorded on day s. Known at the end of day s, so safe inside a window that ends on s.
MEASURES = ["kwh", "reactive_kvarh", "voltage", "intensity", "sub1_wh", "sub2_wh", "sub3_wh", "other_wh"]

WINDOW = 14           # rows of history the LSTM sees
WARMUP = 28           # the notebook's feature table starts at row 28 (28-day rolling mean); we score the same rows
TEST_FRACTION = 0.2   # the latest 20% of rows is the test set
VAL_FRACTION = 0.2    # the last 20% of the remaining 80% is for finding the epoch count

PREDICTIONS_PATH = Path("results/lstm_test_predictions.csv")


def team_split(target_rows: np.ndarray):
    """Row positions of train / val / test, cut the same way as the notebook (cell 22)."""
    n = len(target_rows)
    n_trainval = int(n * (1 - TEST_FRACTION))
    n_train = int(n_trainval * (1 - VAL_FRACTION))
    train, val, test = target_rows[:n_train], target_rows[n_train:n_trainval], target_rows[n_trainval:]
    assert train.max() < val.min() and val.max() < test.min(), "splits overlap in time"
    return train, val, test


def calendar_columns(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Worked out from the dates alone, never from energy readings."""
    dow, doy = dates.dayofweek, dates.dayofyear
    return pd.DataFrame({
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "weekend": (dow >= 5).astype(float),
    }, index=dates)


CALENDAR_COLS = list(calendar_columns(pd.date_range("2007-01-01", periods=3)).columns)


def step_features(daily: pd.DataFrame) -> pd.DataFrame:
    """One row per day s: that day's meter readings and calendar, plus the calendar of the next row.

    Row s uses only readings from day s. next_* columns hold the date of the following row, which is the
    row being predicted when s is the last day of the window. Dates are known in advance, so this is allowed.
    The last day has no following row, so it is dropped.
    """
    cal = calendar_columns(daily.index)
    nxt = cal.shift(-1).add_prefix("next_")
    out = pd.concat([daily[MEASURES], cal, nxt], axis=1)
    return out.iloc[:-1]


def make_windows(feats: np.ndarray, target: np.ndarray, target_rows: np.ndarray):
    """Input = the WINDOW steps just before each target row; label = the target row's own kwh."""
    assert WINDOW >= 1
    assert target_rows.min() >= WINDOW, "not enough history before the first target"
    assert target_rows.max() <= len(feats), "target has no step before it"
    X = np.stack([feats[j - WINDOW : j] for j in target_rows])
    last_step = target_rows - 1
    assert (last_step < target_rows).all(), "window touches the row it predicts"
    return X.astype("float32"), target[target_rows].astype("float32")


def check_features_use_no_future(daily: pd.DataFrame) -> None:
    """Two tamper tests on the energy data.

    1. Scramble every reading from a cut-off on: windows for targets up to the cut-off must not change.
    2. Scramble every reading: calendar columns must not change at all, which proves the look-ahead
       next_* columns hold dates, not energy data.
    """
    cut = len(daily) // 2
    tampered = daily.copy()
    tampered.iloc[cut:, tampered.columns.get_indexer(MEASURES)] = -1.0
    rows = np.arange(WARMUP, cut + 1)
    before = make_windows(step_features(daily).to_numpy(), daily[TARGET].to_numpy(), rows)[0]
    after = make_windows(step_features(tampered).to_numpy(), tampered[TARGET].to_numpy(), rows)[0]
    np.testing.assert_array_equal(before, after)

    scrambled = daily.copy()
    rng = np.random.default_rng(SEED)
    for col in MEASURES:
        scrambled[col] = rng.permutation(scrambled[col].to_numpy())
    calendar = CALENDAR_COLS + [f"next_{c}" for c in CALENDAR_COLS]
    pd.testing.assert_frame_equal(step_features(daily)[calendar], step_features(scrambled)[calendar])


class LSTMForecaster:
    def __init__(self, units: int = 32, dropout: float = 0.2, batch_size: int = 32):
        self.units = units
        self.dropout = dropout
        self.batch_size = batch_size
        self.model = None

    def _build(self, input_shape):
        set_global_seed(SEED)
        model = keras.Sequential([
            keras.Input(shape=input_shape),
            keras.layers.LSTM(self.units),
            keras.layers.Dropout(self.dropout),
            keras.layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        return model

    def find_epochs(self, X_train, y_train, X_val, y_val, max_epochs: int = 300, patience: int = 25) -> int:
        """Train with early stopping on the validation period and return the best epoch count."""
        model = self._build(X_train.shape[1:])
        stop = keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
        # Shuffling training windows is safe: each window already holds only its own past.
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=max_epochs,
                            batch_size=self.batch_size, callbacks=[stop], verbose=0)
        return int(np.argmin(history.history["val_loss"])) + 1

    def fit(self, X, y, epochs: int):
        """Fresh model, fixed number of epochs (no validation data left to watch)."""
        self.model = self._build(X.shape[1:])
        self.model.fit(X, y, epochs=epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict(self, X):
        return self.model.predict(X, batch_size=self.batch_size, verbose=0).ravel()


def mape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = y != 0                      # undefined at 0, same as the notebook
    return float(np.abs((y[m] - p[m]) / y[m]).mean() * 100)


def score(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": mape(y_true, y_pred),
    }


def prepare(feats: pd.DataFrame, kwh: pd.Series, fit_last_row: int):
    """Scale features and target with scalers fitted on rows 0..fit_last_row only."""
    x_scaler = StandardScaler().fit(feats.iloc[: fit_last_row + 1])
    y_scaler = StandardScaler().fit(kwh.iloc[: fit_last_row + 1].to_frame())
    scaled_x = x_scaler.transform(feats)
    scaled_y = y_scaler.transform(kwh.to_frame()).ravel()
    return scaled_x, scaled_y, y_scaler


def run(verbose: bool = True) -> dict:
    daily = load_team_daily()
    check_features_use_no_future(daily)

    feats = step_features(daily)
    kwh = daily[TARGET]
    target_rows = np.arange(WARMUP, len(daily))
    train_rows, val_rows, test_rows = team_split(target_rows)
    if verbose:
        d = daily.index
        print(f"rows train {len(train_rows)} ({d[train_rows[0]].date()} to {d[train_rows[-1]].date()})  "
              f"val {len(val_rows)} (to {d[val_rows[-1]].date()})  "
              f"test {len(test_rows)} ({d[test_rows[0]].date()} to {d[test_rows[-1]].date()})")

    # Stage 1: scalers see only the training period; validation picks the epoch count.
    fx, fy, _ = prepare(feats, kwh, fit_last_row=train_rows[-1])
    X_tr, y_tr = make_windows(fx, fy, train_rows)
    X_va, y_va = make_windows(fx, fy, val_rows)
    epochs = LSTMForecaster().find_epochs(X_tr, y_tr, X_va, y_va)

    # Stage 2: retrain on train+val for that many epochs, then score the untouched test rows.
    trainval_rows = np.concatenate([train_rows, val_rows])
    fx, fy, y_scaler = prepare(feats, kwh, fit_last_row=val_rows[-1])
    X_fit, y_fit = make_windows(fx, fy, trainval_rows)
    X_te, _ = make_windows(fx, fy, test_rows)
    model = LSTMForecaster().fit(X_fit, y_fit, epochs)

    y_true = kwh.to_numpy()[test_rows]
    y_pred = y_scaler.inverse_transform(model.predict(X_te).reshape(-1, 1)).ravel()

    naive = kwh.shift(7).to_numpy()[test_rows]        # the notebook's "naive (lag7)"
    yesterday = kwh.shift(1).to_numpy()[test_rows]
    result = {
        "epochs": epochs,
        "days": daily.index[test_rows],
        "actual": y_true,
        "predicted": y_pred,
        "lstm": score(y_true, y_pred),
        "naive": score(y_true, naive),
        "yesterday": score(y_true, yesterday),
    }
    result["gain_vs_naive"] = (1 - result["lstm"]["mae"] / result["naive"]["mae"]) * 100

    if verbose:
        print(f"epochs chosen on val: {epochs}")
        print("\nTest scores (same days as the notebook's models)")
        for name, key in [("LSTM", "lstm"), ("naive (lag7)", "naive"), ("yesterday", "yesterday")]:
            s = result[key]
            print(f"{name:<14} MAE {s['mae']:6.3f}   RMSE {s['rmse']:6.3f}   MAPE {s['mape']:5.1f}%")
        print(f"LSTM MAE vs naive: {result['gain_vs_naive']:+.1f}% (target: at least 15% lower)")
        print("\nMonthly roll-up of the LSTM's forecasts (complete months only)")
        print(monthly_rollup(result["days"], y_true, y_pred).round(1).to_string())
    return result


def monthly_rollup(days: pd.DatetimeIndex, y_true, y_pred) -> pd.DataFrame:
    """Add forecasts up to calendar months. Only months where every day of the month was forecast count."""
    daily = pd.DataFrame({"actual": y_true, "predicted": y_pred}, index=days)
    months = daily.resample("MS").agg(["sum", "count"])
    full = months[("actual", "count")] == months.index.days_in_month
    out = pd.DataFrame({
        "actual_kwh": months.loc[full, ("actual", "sum")],
        "predicted_kwh": months.loc[full, ("predicted", "sum")],
    })
    out["error_pct"] = (out["predicted_kwh"] - out["actual_kwh"]) / out["actual_kwh"] * 100
    return out


def main() -> None:
    result = run()
    PREDICTIONS_PATH.parent.mkdir(exist_ok=True)
    pd.DataFrame({"date": result["days"], "actual_kwh": result["actual"], "lstm_kwh": result["predicted"]}
                 ).to_csv(PREDICTIONS_PATH, index=False)
    print(f"\nSaved test predictions to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()

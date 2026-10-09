# PowerForecastingProject

## Layout

```
data/        manifest.json, data_dictionary.md (raw files are not committed)
src/         build_daily.py, forecast_pipeline.py, evaluate.py, config.py, dashboard.py
src/models/  lstm.py, gru.py, naive.py
notebooks/   exploration only, outputs cleared before committing
results/     <model>_test_predictions.csv, comparison_table.csv, run_metadata.json
docs/        worklogs, report drafts
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (source .venv/bin/activate elsewhere)
pip install -r requirements.txt
```

Run the scripts as modules from the repo root, e.g. `python -m src.build_daily`.

## Running everything

Download the raw file from the URL in `data/manifest.json`, extract `household_power_consumption.txt`
into `data/`, and check its SHA-256 matches `raw_sha256`. Then, in order:

```bash
python -m src.build_daily --raw data/household_power_consumption.txt   # writes data/household_daily.csv
python -m src.forecast_pipeline --data data/household_daily.csv        # ridge, random_forest, xgboost predictions
python -m src.models.lstm        # writes results/lstm_test_predictions.csv
python -m src.models.gru         # writes results/gru_test_predictions.csv
python -m src.models.naive       # writes results/naive_test_predictions.csv
python -m src.evaluate           # writes results/comparison_table.csv and results/run_metadata.json
```

`build_daily` also rewrites `data/manifest.json` and `data/data_dictionary.md`. `evaluate` stops with
an error if any model was scored on different days or actual values from the others.

## Dashboard

Shows every model's test-period forecasts against the actual kWh, daily or added up per month,
with MAE / RMSE / MAPE and the gain over the seasonal naive baseline. Run the steps above first.

```bash
python -m streamlit run src/dashboard.py
```

It opens in your browser at http://localhost:8501. Stop it with `Ctrl+C` in the terminal.

### Adding your model

Save your test predictions as `results/<model>_test_predictions.csv`, for example
`results/xgboost_test_predictions.csv`, with exactly these columns:

| Column | Meaning |
|--------|---------|
| `date` | Test day, `YYYY-MM-DD` |
| `actual_kwh` | What the meter recorded that day (kWh) |
| `<model>_kwh` | Your model's forecast for that day (kWh), e.g. `xgboost_kwh` |

One row per test day. From the notebook, after fitting:

```python
pd.DataFrame({"date": te.date, "actual_kwh": yte, "xgboost_kwh": xgb.predict(Xte)}) \
  .to_csv("results/xgboost_test_predictions.csv", index=False)
```

Refresh the page and the model appears. Every model must use the same test days and the same
actual values (the team daily table and split). If yours doesn't, the dashboard shows a warning,
because the scores wouldn't be comparable.

## Rules

- `main` is protected. Every change goes through a pull request.
- The random seed lives in `src/config.py` (`SEED = 42`). Import it; never type `42` anywhere else.
- Clear notebook outputs before committing.

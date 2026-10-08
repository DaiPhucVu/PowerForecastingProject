# Household Energy Usage Prediction

Next-day household electricity forecasting for the **ARC Research Hub for Integrated
Energy Storage Solutions** (UNSW Sydney).

Team **Forecasting 5** — ICT30016 ICT Innovation Project, Swinburne University of Technology.

---

## The problem

The client needs short-horizon forecasts of household electricity consumption to inform
storage sizing and dispatch decisions. A forecast is only useful if it beats the obvious
guess — "this Monday will look like last Monday" — and if the client can see *why* a
particular day was predicted high or low.

So the project has two targets, not one: beat a seasonal-naive baseline on error, and
produce driver explanations a non-specialist can act on.

## Approach in one paragraph

We model at **daily** resolution and report at **monthly** resolution. The dataset covers
47 months, so modelling monthly directly leaves roughly 47 training rows — not enough to
fit anything. Modelling daily gives about 1,400 rows, and daily predictions aggregate up
to a monthly view with lower error than the daily figures, because day-to-day noise
cancels out. That one decision is the spine of the design.

## Dataset

UCI *Individual household electric power consumption* — one house in Sceaux, France,
sampled every minute from December 2006 to November 2010.

- Source: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
- DOI: 10.24432/C58K54
- Raw: 2,075,259 minute-level rows, 1.25% with no values, zero missing timestamps

`Global_active_power` is **kW sampled per minute**, not energy. A day's total energy in
kWh is therefore the sum of the minute readings divided by 60. Skipping that division
inflates daily consumption from a realistic ~26 kWh to ~1,571 kWh, so it is the single
most important conversion in the pipeline.

A calendar day is kept only if **at least 90% of its 1,440 minutes** were recorded.
Partial days are dropped rather than interpolated: a day with three hours of data looks
like an extremely low-consumption day, which would poison both the target and the lags.

**Result: 1,418 usable days.** Exact counts, date range and the raw file's SHA-256 are in
`data/manifest.json`.

## Results

Test set is the final 20% of days in chronological order. No shuffling at any point.

| Model | MAE (kWh) | RMSE | MAPE | vs baseline |
|---|---|---|---|---|
| Seasonal naive (lag-7) | 6.02 | — | — | baseline |
| Ridge regression | — | — | — | — |
| Random Forest | — | — | — | — |
| **XGBoost** | **4.04** | — | **18.5%** | **−33.0%** |

> Fill the blanks from `results/comparison_table.csv` after running the notebook.

**Chosen model: XGBoost.** The other three are reported rather than delivered — the
baseline gives the improvement figure its meaning, Ridge shows the problem is not
trivially linear, and Random Forest shows the gain comes from gradient boosting
specifically rather than from any tree ensemble.

### A finding worth recording

An earlier export of this dataset gave only 728 usable days. On that smaller sample Ridge
was the best model and **every** machine learning model lost to the seasonal-naive
baseline on MAPE. After recovering the full raw file and rebuilding to 1,418 days,
XGBoost wins and all models beat the baseline on all three metrics.

The models did not change. The sample size did. This is a concrete demonstration that
apparent model quality in small-sample time-series work can be an artefact of the sample.

### Top driver

Feature importances are in `results/xgb_feature_importance.csv`.

Note that `other_wh` is a **residual** — total active energy minus the three sub-metered
circuits — not a metered circuit in its own right. If it ranks highly, that says
unmetered load carries much of the day-to-day signal, which is itself useful to the
client, but it should not be described as a measured appliance group.

## Avoiding leakage

Seven columns in the daily table are only known once a day has finished: `reactive_kvarh`,
`voltage`, `intensity`, `sub1_wh`, `sub2_wh`, `sub3_wh`, `other_wh`. Using any of them to
predict the same day's consumption would be using the answer to compute the answer.

They appear in the feature table **only as lags** (previous day, previous week), and this
is enforced rather than trusted:

- `lag_1` is asserted equal to the previous day's actual consumption for every row
- the raw same-day column names are asserted absent from the feature table

Both assertions run every time the pipeline executes. Splits are chronological
(train → validation → test in time order), following the argument in Bergmeir & Benítez
(2012) that random k-fold cross-validation is invalid for dependent data.

## Repository layout

```
├── src/
│   ├── build_daily.py          raw .txt → daily table + manifest + data dictionary
│   └── forecast_pipeline.py    features → split → models → comparison table
├── notebooks/
│   └── household_energy_model.ipynb    Colab notebook, runs the whole thing end to end
├── data/
│   ├── manifest.json           provenance: raw checksum, row counts, date range
│   ├── data_dictionary.md      every column, its unit and its meaning
│   └── household_daily.csv     generated, not committed
├── results/
│   ├── comparison_table.csv
│   ├── xgb_feature_importance.csv
│   └── feature_importance.png
└── docs/                       specification, innovation report, weekly work logs
```

The raw 127 MB file and the generated daily CSV are **not** in the repository. They are
reproducible from the two scripts, and `manifest.json` carries the raw file's SHA-256 so
anyone can confirm they rebuilt from the same source.

## Reproducing the results

Everything is seeded (`SEED = 42`) and `n_jobs=1` in `forecast_pipeline.py`, so a rerun on
the same input produces identical numbers.

### In Colab (easiest)

Open `notebooks/household_energy_model.ipynb`, run the cells in order, and upload
`individualhouseholdelectricpowerconsumption.zip` when prompted. Runtime is a few minutes,
most of it parsing the raw file.

### Locally

```bash
pip install -r requirements.txt

# download and unzip the raw file into raw/ first
python src/build_daily.py --raw raw/household_power_consumption.txt
python src/forecast_pipeline.py --data data/household_daily.csv
```

`build_daily.py` writes `data/household_daily.csv`, `data/manifest.json` and
`data/data_dictionary.md`. `forecast_pipeline.py` writes `results/comparison_table.csv`,
`results/xgb_feature_importance.csv` and `results/run_metadata.json`.

If your numbers differ from the table above, your run is the truth — the dataset or the
environment has changed and the table needs updating.

## Known limitations

- **One household.** Results describe a single house in France. They do not establish that
  the approach transfers to other households, other climates, or Australian consumption
  patterns. Any generalisation claim needs a second dataset.
- **Interior gaps.** Days dropped by the coverage rule leave gaps, so lag features computed
  by row position can reach back further than the stated number of calendar days.
- **Single split.** The reported figures come from one test window. Rolling-origin
  validation across several folds is outstanding.
- **No weather data.** Temperature is a known driver of household consumption and is not
  in this dataset. Lagged consumption absorbs some of it indirectly.

## Status

| Stage | State |
|---|---|
| Daily dataset rebuilt from raw, documented, checksummed | Done |
| Leakage-free feature engineering with assertions | Done |
| Baseline + Ridge + Random Forest + XGBoost comparison | Done |
| Monthly rollup of daily forecasts | Outstanding |
| Rolling-origin validation | Outstanding |
| SHAP explainability | Outstanding |
| LSTM / GRU sequence model trial | Outstanding |

## Team

| Member | Role |
|---|---|
| Rothavisal Keo | Scrum Master and Data Lead |
| Kimly | Modelling |
| Isaiah | Validation |
| Dai Phuc | Explainability and reporting |

## References

Bergmeir, C. & Benítez, J.M. (2012) 'On the use of cross-validation for time series
predictor evaluation', *Information Sciences*, 191, pp. 192–213.
https://doi.org/10.1016/j.ins.2011.12.028

Hebrail, G. & Berard, A. (2012) *Individual household electric power consumption*. UCI
Machine Learning Repository. https://doi.org/10.24432/C58K54
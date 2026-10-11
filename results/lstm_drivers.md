# What drives the LSTM's daily forecast

Model: LSTM, average of 5 copies, 14-day input window. Explained on 278 held-out test days.

## Which inputs it relies on (permutation importance)

How much the test MAE rises, in kWh, when one input's history is shuffled across days. Bigger means the forecast depends on it more.

1. **time of year**: MAE rises by 0.79 kWh (± 0.19).
2. **total load (kWh and current together)**: MAE rises by 0.74 kWh (± 0.10).
3. **water heater and A/C**: MAE rises by 0.40 kWh (± 0.07).
4. **day of week**: MAE rises by 0.31 kWh (± 0.11).
5. **mean current**: MAE rises by 0.29 kWh (± 0.08).
6. **daily consumption (kWh)**: MAE rises by 0.27 kWh (± 0.05).

## How far back it looks

Shuffling only the meter readings from one day of the window hurts most for **t-1** (MAE +0.48 kWh). See `lstm_lookback.png` for every day.

## Direction and size (SHAP)

Average forecast over the background days: **27.22 kWh**. Each figure is how far an input (its whole 14-day history) moves a day's forecast away from that average, averaged over the test days.

1. **time of year**: 2.84 kWh on an average day, up to 6.85 kWh. Calendar effect (no single direction).
2. **day of week**: 1.85 kWh on an average day, up to 5.38 kWh. Calendar effect (no single direction).
3. **mean current**: 0.94 kWh on an average day, up to 4.88 kWh. Higher raises the forecast.
4. **water heater and A/C**: 0.78 kWh on an average day, up to 2.17 kWh. Higher raises the forecast.
5. **daily consumption (kWh)**: 0.66 kWh on an average day, up to 3.15 kWh. Higher raises the forecast.
6. **unmetered load (residual)**: 0.53 kWh on an average day, up to 1.58 kWh. Higher raises the forecast.

Additivity check: base value plus the SHAP values is within 0.13 kWh of the forecast on an average day (worst 0.47 kWh), against a typical 5.16 kWh gap between forecast and base value. Expected gradients are sampled, so this is approximate, unlike TreeSHAP.

## Compared with XGBoost

| Input | LSTM rank | XGBoost rank |
|---|---|---|
| time of year | 1 | 4 |
| day of week | 2 | 3 |
| mean current | 3 | 10 |
| water heater and A/C | 4 | 6 |
| daily consumption (kWh) | 5 | 1 |
| unmetered load (residual) | 6 | 2 |
| kitchen circuit | 7 | 7 |
| laundry circuit | 8 | 5 |
| reactive energy | 9 | 9 |
| mean voltage | 10 | 8 |

1 of the LSTM's top 3 inputs are also in XGBoost's top 3. XGBoost's ranking adds up its features per group (from `shap_summary.csv`), so compare ranks, not kWh. Mean current and daily consumption carry the same information, so a model can lean on either: read their ranks together.

## Reading this

- Permutation importance says what the model *uses*; SHAP adds *which way* and *by how much* per day.
- Daily consumption and mean current carry almost the same information, so either alone looks less important than both together (see *total load*).
- `other_wh` is a residual (total minus the three sub-meters), not a measured appliance group.

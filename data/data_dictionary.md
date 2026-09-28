# Data dictionary

The daily table that `src/build_daily.py` produces: `data/household_daily.csv`,
built from the UCI Individual Household Electric Power Consumption data (one reading per minute).

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| date | date | — | Calendar day. Whole days only, 2006-12-17 to 2010-11-25, no gaps |
| energy_kwh | float | kWh | Energy used that day: mean `Global_active_power` (kW) over the day's readings × 24 h |
| coverage | float | 0–1 | Share of the day's 1,440 minutes that had a reading |
| imputed | bool | — | True when coverage < 0.8. `energy_kwh` is then copied from the same weekday one week earlier. Don't use these days as prediction targets |

# What drives the daily forecast

Model: XGBoost (n=101). Explained on 278 held-out test days using TreeSHAP.

The model's average prediction across these days is **26.31 kWh**. Each figure below is how far that one feature moves a day's forecast away from that average, in kWh, averaged over all the test days.

1. **yesterday's unmetered load (residual)** - moves the forecast by 2.83 kWh on an average day, up to 7.34 kWh on the most affected day. Higher raises the forecast.
2. **time of year (cyclical)** - moves the forecast by 1.12 kWh on an average day, up to 2.75 kWh on the most affected day. Higher raises the forecast.
3. **day of week** - moves the forecast by 0.82 kWh on an average day, up to 2.11 kWh on the most affected day. Higher raises the forecast.
4. **average of the last 7 days** - moves the forecast by 0.57 kWh on an average day, up to 1.85 kWh on the most affected day. Higher raises the forecast.
5. **consumption three days ago** - moves the forecast by 0.48 kWh on an average day, up to 1.23 kWh on the most affected day. Higher raises the forecast.
6. **yesterday's consumption** - moves the forecast by 0.40 kWh on an average day, up to 2.37 kWh on the most affected day. Higher raises the forecast.

## Reading this

SHAP values are additive: for any single day, the base value plus every feature's contribution equals that day's prediction exactly. So these numbers can be checked, not just trusted.

A feature marked *mixed / not monotonic* matters, but not in one direction - typically because its effect depends on another feature. Calendar features behave this way by nature.

## Caveat on the top driver

`other_wh` is a **residual**: total active energy minus the three sub-metered circuits. It is not a measured appliance group. Its importance says that load outside the three monitored circuits carries much of the day-to-day variation, which is a finding about the metering coverage of this house, not about a specific appliance.

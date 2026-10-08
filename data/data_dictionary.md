# Data dictionary - household_daily.csv

Source: UCI Individual household electric power consumption (Hebrail & Berard 2012),
one house in Sceaux, France, sampled every minute.
Raw SHA-256: `4259c9d7ece5dbee...`

Built by `build_daily.py`. A calendar day is kept only if at least
90% of its 1440 minutes were recorded; 24 partial days were dropped.
No interpolation is applied.

| Column | Unit | Meaning |
|---|---|---|
| `date` | date | Calendar day |
| `kwh` | kWh | **Target.** Daily active energy. Minute kW readings summed, divided by 60. |
| `reactive_kvarh` | kVArh | Daily reactive energy, same conversion |
| `voltage` | V | Mean voltage across the day |
| `intensity` | A | Mean current intensity across the day |
| `sub1_wh` | Wh | Sub-meter 1: kitchen (dishwasher, oven, microwave) |
| `sub2_wh` | Wh | Sub-meter 2: laundry (washing machine, dryer, fridge, light) |
| `sub3_wh` | Wh | Sub-meter 3: water heater and air conditioner |
| `other_wh` | Wh | Active energy not captured by the three sub-meters (residual, not a metered circuit) |

Coverage: 2006-12-17 to 2010-11-25, 1418 days.
Note: 7 interior gaps remain where whole days were dropped, so lag features
computed by row position can reach back more than the stated number of calendar days.

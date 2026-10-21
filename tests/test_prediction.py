"""
Tests for the forecasting path - the part that produces numbers for days that
have not happened yet.

The rest of the suite checks that the models are measured honestly. These check
that the thing a client would actually run does what it claims.
"""
import numpy as np
import pandas as pd
import pytest

import predict as pr
import train_model as tm


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Train both models once on the synthetic daily table and reuse them."""
    import forecast_pipeline as fp

    rng = np.random.default_rng(2)
    n = 500
    date = pd.date_range("2007-01-01", periods=n, freq="D")
    kwh = np.clip(26 + 6 * np.sin(2 * np.pi * date.dayofyear / 365.25)
                  + rng.normal(0, 3, n), 1, None)
    df = pd.DataFrame({
        "date": date, "kwh": kwh, "reactive_kvarh": kwh * 0.08,
        "voltage": 240 + rng.normal(0, 2, n), "intensity": kwh / 5,
        "sub1_wh": rng.uniform(0, 2000, n),
        "sub2_wh": rng.uniform(0, 3000, n),
        "sub3_wh": rng.uniform(0, 9000, n),
    })
    df["other_wh"] = df.kwh * 1000 - df.sub1_wh - df.sub2_wh - df.sub3_wh

    d = tmp_path_factory.mktemp("trained")
    data = d / "daily.csv"
    df.to_csv(data, index=False)
    tm.main(str(data), str(d / "models"))
    return df, str(data), str(d / "models")


# ------------------------------------------------- feature building by hand

def test_lag_1_is_the_most_recent_value():
    series = [10.0, 20.0, 30.0, 40.0, 50.0] * 10
    row = pr.history_features(series, "2010-06-15")
    assert row["lag_1"] == series[-1]
    assert row["lag_7"] == series[-7]


def test_rolling_mean_uses_only_past_values():
    series = list(range(1, 41))              # 1..40
    row = pr.history_features(series, "2010-06-15")
    assert row["roll_mean_7"] == pytest.approx(np.mean(series[-7:]))
    assert row["roll_mean_28"] == pytest.approx(np.mean(series[-28:]))


def test_calendar_features_for_a_known_date():
    # 2010-06-15 was a Tuesday, day 166 of the year.
    f = pr.calendar_features("2010-06-15")
    assert f["dow"] == 1
    assert f["month"] == 6
    assert f["doy"] == 166
    assert f["is_weekend"] == 0


def test_weekend_flag():
    assert pr.calendar_features("2010-06-19")["is_weekend"] == 1   # Saturday
    assert pr.calendar_features("2010-06-20")["is_weekend"] == 1   # Sunday
    assert pr.calendar_features("2010-06-21")["is_weekend"] == 0   # Monday


def test_cyclical_encodings_stay_on_the_unit_circle():
    for date in ("2010-01-01", "2010-06-15", "2010-12-31"):
        f = pr.calendar_features(date)
        assert f["dow_sin"] ** 2 + f["dow_cos"] ** 2 == pytest.approx(1.0)
        assert f["doy_sin"] ** 2 + f["doy_cos"] ** 2 == pytest.approx(1.0)


# ------------------------------------------------------------- the forecast

def test_forecast_starts_the_day_after_the_last_observation(trained):
    df, _, models = trained
    fc, _, _ = pr.forecast(df, 7, models)

    expected = pd.Timestamp(df.date.iloc[-1]) + pd.Timedelta(days=1)
    assert fc.date.iloc[0] == expected


def test_forecast_dates_are_consecutive(trained):
    df, _, models = trained
    fc, _, _ = pr.forecast(df, 14, models)
    assert (fc.date.diff().dt.days.dropna() == 1).all()
    assert len(fc) == 14


def test_first_step_uses_the_accurate_model_and_the_rest_recursive(trained):
    """Only tomorrow can use yesterday's end-of-day readings. Everything
    further out must fall back to the reduced feature set."""
    df, _, models = trained
    fc, _, _ = pr.forecast(df, 5, models)

    assert fc.model.iloc[0] == "next_day"
    assert set(fc.model.iloc[1:]) == {"horizon"}


def test_predictions_are_physically_plausible(trained):
    df, _, models = trained
    fc, _, _ = pr.forecast(df, 30, models)

    assert (fc.predicted_kwh > 0).all()
    # Nothing should wander outside the range the house has ever recorded by
    # more than a wide margin - a runaway recursive forecast would.
    assert fc.predicted_kwh.max() < df.kwh.max() * 1.5
    assert fc.predicted_kwh.min() > df.kwh.min() * 0.5


def test_uncertainty_band_widens_with_distance(trained):
    """Day 30 must not be presented with the same confidence as day 1."""
    df, _, models = trained
    fc, _, _ = pr.forecast(df, 20, models)

    width = fc.upper_kwh - fc.lower_kwh
    assert width.iloc[-1] > width.iloc[0]
    assert fc.expected_MAE.is_monotonic_increasing


def test_band_contains_the_point_forecast(trained):
    df, _, models = trained
    fc, _, _ = pr.forecast(df, 10, models)
    assert (fc.lower_kwh <= fc.predicted_kwh).all()
    assert (fc.predicted_kwh <= fc.upper_kwh).all()


def test_lower_bound_never_goes_negative(trained):
    """A house cannot consume negative energy, so the band is clipped."""
    df, _, models = trained
    fc, _, _ = pr.forecast(df, 60, models)
    assert (fc.lower_kwh >= 0).all()


def test_forecast_is_deterministic(trained):
    """Same data, same models, same answer - the seed is fixed and nothing
    samples at prediction time."""
    df, _, models = trained
    a, _, _ = pr.forecast(df, 10, models)
    b, _, _ = pr.forecast(df, 10, models)
    pd.testing.assert_frame_equal(a, b)


def test_horizon_model_never_sees_end_of_day_features(trained):
    """The whole reason two models exist. If a sub-meter lag reached the
    horizon feature list, recursive forecasting would be impossible."""
    _, _, models = trained
    _, meta = pr.load_model(f"{models}/horizon.joblib")

    for col in ("other_wh", "sub1_wh", "sub2_wh", "sub3_wh",
                "voltage", "intensity", "reactive_kvarh"):
        assert not any(f.startswith(col) for f in meta["features"]), \
            f"{col} leaked into the horizon model"


def test_next_day_model_does_use_them(trained):
    """And the one-step model should, since they are genuinely available."""
    _, _, models = trained
    _, meta = pr.load_model(f"{models}/next_day.joblib")
    assert "other_wh_lag1" in meta["features"]


# ------------------------------------------------------------ monthly view

def test_monthly_flags_months_the_forecast_only_partly_covers(trained):
    df, _, models = trained
    fc, _, _ = pr.forecast(df, 20, models)
    monthly = pr.to_monthly(fc, df.date.iloc[-1])

    # 20 days cannot fill a calendar month, so nothing may be marked complete.
    assert not monthly.complete.any()
    assert monthly.days_forecast.sum() == 20


def test_monthly_totals_match_the_daily_sum(trained):
    df, _, models = trained
    fc, _, _ = pr.forecast(df, 45, models)
    monthly = pr.to_monthly(fc, df.date.iloc[-1])

    assert monthly.predicted_kwh.sum() == pytest.approx(fc.predicted_kwh.sum())


# --------------------------------------------------------- saved model card

def test_model_metadata_records_provenance(trained):
    _, _, models = trained
    for name in ("next_day", "horizon"):
        _, meta = pr.load_model(f"{models}/{name}.joblib")
        for key in ("features", "n_estimators", "trained_rows",
                    "trained_from", "trained_to", "data_sha256_12", "seed"):
            assert key in meta, f"{name} metadata missing {key}"
        assert meta["seed"] == 42


def test_model_card_was_written(trained):
    from pathlib import Path
    _, _, models = trained
    card = Path(models) / "model_card.md"
    assert card.exists()
    text = card.read_text()
    assert "Limits" in text and "One household" in text

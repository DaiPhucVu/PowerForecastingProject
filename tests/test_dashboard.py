"""
Tests for the dashboard's logic (no browser): the household view's explanations must add up to the forecast
it shows, and the researcher export must carry a stable run ID, the data version and every file.
"""
import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
import dashboard as db


def explanations(days: pd.DatetimeIndex, base=20.0, seed=0) -> pd.DataFrame:
    """Per-day contributions in the shape of results/lstm_shap_daily.csv, already regrouped."""
    rng = np.random.default_rng(seed)
    out = pd.DataFrame({"date": days, "base_kwh": base, "unexplained_kwh": rng.normal(0, 0.05, len(days))})
    for name in db.HOUSEHOLD_DRIVERS:
        out[name] = rng.normal(0, 2, len(days))
    out["predicted_kwh"] = out["base_kwh"] + out[list(db.HOUSEHOLD_DRIVERS)].sum(axis=1) + out["unexplained_kwh"]
    return out


def test_month_drivers_add_up_to_the_month_forecast():
    expl = explanations(pd.date_range("2010-03-01", "2010-04-30"))
    typical, pushes = db.month_drivers(expl, pd.Timestamp("2010-03-01"))
    march = expl[expl["date"].dt.month == 3]
    assert typical == pytest.approx(20.0 * 31)
    assert typical + pushes.sum() + march["unexplained_kwh"].sum() == pytest.approx(march["predicted_kwh"].sum())
    assert list(pushes.abs()) == sorted(pushes.abs(), reverse=True)     # biggest reason first


def test_month_drivers_refuses_an_incomplete_month():
    expl = explanations(pd.date_range("2010-03-01", "2010-03-20"))
    assert db.month_drivers(expl, pd.Timestamp("2010-03-01")) is None


def test_household_drivers_cover_every_lstm_input_group_once():
    lstm_groups = ["kwh", "intensity", "reactive_kvarh", "voltage", "sub1_wh", "sub2_wh", "sub3_wh", "other_wh",
                   "day_of_week", "time_of_year"]
    members = [c for cols in db.HOUSEHOLD_DRIVERS.values() for c in cols]
    assert sorted(members) == sorted(lstm_groups)


def prediction_table():
    days = pd.date_range("2010-02-01", periods=60, name="date")
    actual = 20 + np.sin(np.arange(60))
    return pd.DataFrame({"date": days, "actual": actual, "naive": actual + 3, "lstm": actual + 0.5}), ["naive", "lstm"]


def test_best_model_ignores_the_baseline():
    table, models = prediction_table()
    assert db.best_model(table, models) == "lstm"


def test_export_bundle_has_manifest_files_and_a_stable_run_id(tmp_path, monkeypatch):
    table, models = prediction_table()
    for m in models:
        table[["date", "actual", m]].rename(columns={"actual": "actual_kwh", m: f"{m}_kwh"}) \
            .to_csv(tmp_path / f"{m}_test_predictions.csv", index=False)
    (tmp_path / "run_metadata.json").write_text(json.dumps({"data_file": "household_daily.csv",
                                                            "data_sha256_12": "abc123def456"}))
    monkeypatch.setattr(db, "RESULTS_DIR", tmp_path)

    bundle, manifest = db.export_bundle(table, models)
    names = zipfile.ZipFile(io.BytesIO(bundle)).namelist()
    assert {"manifest.json", "comparison_scores.csv", "all_test_predictions.csv",
            "predictions/lstm_test_predictions.csv", "predictions/naive_test_predictions.csv"} <= set(names)
    assert manifest["data_sha256_12"] == "abc123def456"
    assert set(manifest["files"]) == set(names) - {"manifest.json"}

    _, again = db.export_bundle(table, models)
    assert again["run_id"] == manifest["run_id"]                    # same results, same ID

    (tmp_path / "lstm_test_predictions.csv").write_text("date,actual_kwh,lstm_kwh\n2010-02-01,1,2\n")
    _, changed = db.export_bundle(table, models)
    assert changed["run_id"] != manifest["run_id"]                  # changed results, new ID

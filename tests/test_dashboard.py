"""
Tests for the dashboard generator.

The Week 8 QA plan sets one acceptance criterion for the dashboard (feature F3):
the scores it displays must match `comparison_table.csv` exactly. That is checked
here rather than by eye, so it stays true after the next rebuild.
"""
import csv
import json
import re
from pathlib import Path

import pytest

import build_dashboard as bd


@pytest.fixture
def project(tmp_path):
    """A minimal project tree with known numbers in it."""
    (tmp_path / "results").mkdir()
    (tmp_path / "data").mkdir()

    def write(rel, rows):
        with (tmp_path / rel).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    write("results/comparison_table.csv", [
        {"model": "Seasonal naive (lag 7)", "MAE": 6.164, "RMSE": 7.91,
         "MAPE_%": 27.84, "MAE_vs_baseline_%": 0.0},
        {"model": "XGBoost", "MAE": 4.727, "RMSE": 6.02,
         "MAPE_%": 27.82, "MAE_vs_baseline_%": 23.3},
        {"model": "Ridge regression", "MAE": 4.734, "RMSE": 6.05,
         "MAPE_%": 25.06, "MAE_vs_baseline_%": 23.2},
    ])
    write("results/rolling_folds.csv", [
        {"fold": f, "model": m, "MAE": v}
        for f in (1, 2) for m, v in (("XGBoost", 4.7), ("Ridge regression", 4.8))
    ])
    write("results/rolling_summary.csv", [
        {"model": "XGBoost", "mean_MAE": 4.727, "std_MAE": 0.974,
         "worst_MAE": 6.354, "mean_MAPE": 27.8, "mean_vs_baseline": 23.6,
         "beat_baseline_%": 100.0},
        {"model": "Ridge regression", "mean_MAE": 4.734, "std_MAE": 0.894,
         "worst_MAE": 5.995, "mean_MAPE": 25.1, "mean_vs_baseline": 23.5,
         "beat_baseline_%": 100.0},
    ])
    write("results/ablation.csv", [
        {"feature_set": "all features", "n_features": 33, "dropped": 0,
         "mean_MAE": 4.727, "std_MAE": 0.974, "mean_MAPE": 27.8,
         "MAE_change": 0.0, "MAE_change_%": 0.0},
        {"feature_set": "no residual lags", "n_features": 31, "dropped": 2,
         "mean_MAE": 4.810, "std_MAE": 1.095, "mean_MAPE": 28.5,
         "MAE_change": 0.082, "MAE_change_%": 1.74},
    ])
    (tmp_path / "data" / "manifest.json").write_text(json.dumps({
        "daily_rows": 1418, "raw_rows": 2075259,
        "date_from": "2006-12-16", "date_to": "2010-11-02",
        "min_coverage_rule": 0.9, "raw_sha256": "a" * 64,
    }))
    return tmp_path


def build(root):
    out = Path(root) / "results" / "dashboard.html"
    bd.main(str(root), str(out))
    return out.read_text()


# ------------------------------------------- the QA plan's F3 criterion

def test_displayed_scores_match_the_comparison_table(project):
    """F3: 'MAE, RMSE, MAPE match the table exactly.' Every number in
    comparison_table.csv must appear in the rendered page."""
    page = build(project)

    with (project / "results" / "comparison_table.csv").open() as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        assert r["model"] in page, f"{r['model']} missing from the dashboard"
        for col in ("MAE", "RMSE", "MAPE_%"):
            shown = f"{float(r[col]):,.3f}"
            assert shown in page, \
                f"{r['model']} {col}={shown} not displayed on the dashboard"


def test_best_model_tile_is_not_the_baseline(project):
    """The headline tile must report the best LEARNED model, never the
    seasonal-naive baseline, even if the baseline sorted first."""
    page = build(project)
    tile = page.split('Best single-split MAE')[0][-400:]
    assert "4.73" in tile          # tiles round to 2dp; the table keeps 3
    assert "6.16" not in tile


def test_improvement_against_baseline_is_computed_correctly(project):
    page = build(project)
    expected = (6.164 - 4.727) / 6.164 * 100      # 23.3%
    assert f"{expected:,.1f}" in page


# ------------------------------------------------------- graceful degradation

def test_builds_with_nothing_but_an_empty_tree(tmp_path):
    """A dashboard built before any script has run must still open, and say
    which command produces each missing piece."""
    page = build(tmp_path)
    assert "<html" in page and "</html>" in page
    assert "not yet generated" in page
    assert "src/forecast_pipeline.py" in page


def test_missing_forecast_does_not_break_the_page(project):
    page = build(project)                   # no forecast.csv in the fixture
    assert "src/predict.py" in page
    assert "Model comparison" in page       # the rest still rendered


# ------------------------------------------------------------- correctness

def test_every_chart_has_a_table_beneath_it(project):
    """Three light-mode series sit below 3:1 contrast, so the palette's relief
    rule applies: the values must also be readable as text."""
    page = build(project)
    assert page.count("<table>") >= 3


def test_page_is_self_contained(project):
    """No external scripts, stylesheets or images - it has to open from a file
    on a machine with no internet."""
    page = build(project)
    assert "<script" not in page.lower()
    assert "http://" not in page
    assert not re.search(r'<link[^>]+stylesheet', page, re.I)
    assert not re.search(r'<img\b', page, re.I)


def test_dark_mode_is_defined_under_both_scopes(project):
    page = build(project)
    assert "prefers-color-scheme:dark" in page.replace(" ", "")
    assert '[data-theme=dark]' in page.replace('"', "")


def test_marks_carry_hover_titles(project):
    page = build(project)
    assert page.count("<title>") >= 5


def test_text_is_escaped(tmp_path):
    (tmp_path / "results").mkdir()
    with (tmp_path / "results" / "comparison_table.csv").open("w",
                                                             newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "MAE", "RMSE", "MAPE_%"])
        w.writeheader()
        w.writerow({"model": "<script>x</script>", "MAE": 1.0,
                    "RMSE": 1.0, "MAPE_%": 1.0})
    page = build(tmp_path)
    assert "<script>" not in page
    assert "&lt;script&gt;" in page


def test_rounded_bar_only_rounds_the_data_end():
    """A bar rounded at both ends detaches visually from its baseline."""
    d = bd.rounded_bar(0, 0, 100, 20, r=4, horizontal=True)
    assert d.startswith("M0,0")        # square at the baseline
    assert d.count("A4,4") == 2        # two arcs, both at the far end


def test_zero_width_bar_renders_nothing():
    assert bd.rounded_bar(0, 0, 0, 20) == ""

"""Forecast dashboard with two views, one per persona in the Week 5 worklog.

Run from the repo root:
    python -m streamlit run src/dashboard.py
Open a view directly with ?view=household or ?view=researcher.

Household view (Household Resident: no ML background, wants quick answers):
    US1  the forecast for a month, in kWh and dollars, so the bill is no surprise
    US2  the few things that pushed that forecast up or down, in plain sentences
    US3  the forecast next to what was actually used
    It shows only the most accurate model, and no metric names.

Researcher view (Researcher/Analyst: wants detail, raw numbers and exports):
    US4  every model on the same test days and metrics, daily or monthly
    US5  explainability output in technical detail, and everything as files, with a run ID and the
         data version (FR6)

It reads every results/<model>_test_predictions.csv. To add a model, write that file in the
shared format (see README, "Dashboard"):
    date, actual_kwh, <model>_kwh
One row per test day. All models must cover the same test days with the same actual values,
otherwise the comparison isn't fair and the page says so.

The household view's explanations come from results/<model>_shap_daily.csv (written by
src/explain_lstm.py for the LSTM). Without that file for the best model, that section is hidden.
"""

import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
PATTERN = "*_test_predictions.csv"
NAIVE = "naive"           # the baseline the project must beat
TARGET_GAIN = 15.0        # % lower MAE than the naive baseline (Innovation Report)

HOUSEHOLD, RESEARCHER = "Household", "Researcher"
DEFAULT_PRICE = 0.30      # $ per kWh; the household view lets the user type their own

# Input groups from src/explain_lstm.py, regrouped into things a resident recognises. kWh, current,
# reactive energy, voltage and the residual all describe the home's overall recent use.
HOUSEHOLD_DRIVERS = {
    "The time of year": ["time_of_year"],
    "Your overall use in the last few days": ["kwh", "intensity", "reactive_kvarh", "voltage", "other_wh"],
    "Water heater and air-con use": ["sub3_wh"],
    "Kitchen appliance use": ["sub1_wh"],
    "Laundry use": ["sub2_wh"],
    "The mix of weekdays and weekends": ["day_of_week"],
}
UP_COLOR, DOWN_COLOR = "#d95f02", "#1b9e77"

# Explainability outputs shown and exported in the researcher view, if they exist.
ATTRIBUTION_FILES = [
    "lstm_shap_daily.csv", "lstm_shap_summary.csv", "lstm_permutation_importance.csv",
    "lstm_lookback_importance.csv", "shap_summary.csv", "rf_feature_importance.csv", "xgb_feature_importance.csv",
]

DISPLAY_NAMES = {
    "naive": "Seasonal naive",
    "lstm": "LSTM",
    "gru": "GRU",
    "xgboost": "XGBoost",
    "random_forest": "Random forest",
    "rf": "Random forest",
    "ridge": "Ridge",
}
ACTUAL = "Actual"
# Each model keeps its colour on every chart and in both views, however many models are loaded.
MODEL_COLOR = {
    "lstm": "#1f77b4",            # blue
    "random_forest": "#2ca02c",   # green
    "rf": "#2ca02c",
    "xgboost": "#d62728",         # red
    "gru": "#ff7f0e",             # orange
    "ridge": "#9467bd",           # purple
    "naive": "#8c564b",           # brown
}
EXTRA_COLORS = ["#e377c2", "#17becf", "#bcbd22"]   # for models not listed above


def model_color(model: str, all_models: list[str] = ()) -> str:
    if model in MODEL_COLOR:
        return MODEL_COLOR[model]
    unlisted = [m for m in all_models if m not in MODEL_COLOR]
    i = unlisted.index(model) if model in unlisted else 0
    return EXTRA_COLORS[i % len(EXTRA_COLORS)]


def display_name(model: str) -> str:
    return DISPLAY_NAMES.get(model, model.replace("_", " ").title())


@st.cache_data
def load_predictions(files_and_mtimes: tuple) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Merge every prediction file into one table: date, actual, one column per model.

    Returns the table, the model keys, and a list of problems found (shown as warnings).
    The mtimes are only in the arguments so the cache refreshes when a file changes.
    """
    problems = []
    merged, models = None, []
    for path, _ in files_and_mtimes:
        path = Path(path)
        df = pd.read_csv(path, parse_dates=["date"])
        pred_cols = [c for c in df.columns if c.endswith("_kwh") and c != "actual_kwh"]
        if "actual_kwh" not in df.columns or len(pred_cols) != 1:
            problems.append(f"`{path.name}` skipped: needs columns date, actual_kwh and one <model>_kwh.")
            continue
        model = pred_cols[0].removesuffix("_kwh")
        df = df[["date", "actual_kwh", pred_cols[0]]].rename(columns={pred_cols[0]: model})
        models.append(model)

        if merged is None:
            merged = df
            continue
        if not df["date"].reset_index(drop=True).equals(merged["date"].reset_index(drop=True)):
            problems.append(f"**{display_name(model)}** was tested on different days from the other models, "
                            "so its scores aren't directly comparable.")
        merged = merged.merge(df, on="date", how="outer", suffixes=("", f"_{model}"))
        other = merged.pop(f"actual_kwh_{model}")
        both = merged["actual_kwh"].notna() & other.notna()
        if (merged.loc[both, "actual_kwh"] - other[both]).abs().max() > 1e-6:
            problems.append(f"**{display_name(model)}** has different actual values for the same days. "
                            "It was probably built from a different daily table.")
        merged["actual_kwh"] = merged["actual_kwh"].fillna(other)

    if merged is None:
        return pd.DataFrame(), [], problems
    merged = merged.rename(columns={"actual_kwh": "actual"}).sort_values("date")
    # Put missing days back as empty rows, so chart lines break at gaps instead of
    # drawing a straight line across days that have no data.
    all_days = pd.date_range(merged["date"].min(), merged["date"].max(), freq="D", name="date")
    merged = merged.set_index("date").reindex(all_days).reset_index()
    # Baseline first, then the rest alphabetically, so colours stay stable as models are added.
    models = sorted(models, key=lambda m: (m != NAIVE, m))
    return merged, models, problems


def scores(actual: pd.Series, predicted: pd.Series) -> dict:
    ok = actual.notna() & predicted.notna()
    y, p = actual[ok].to_numpy(float), predicted[ok].to_numpy(float)
    nonzero = y != 0   # MAPE is undefined at 0, same rule as the team notebook
    return {
        "MAE (kWh)": np.abs(y - p).mean(),
        "RMSE (kWh)": np.sqrt(((y - p) ** 2).mean()),
        "MAPE (%)": np.abs((y[nonzero] - p[nonzero]) / y[nonzero]).mean() * 100,
        "Days": int(ok.sum()),
    }


def monthly(table: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Daily values added up per calendar month. Only months with every day present count,
    because a month with missing days would look artificially low."""
    cols = ["actual"] + models
    m = table.set_index("date")[cols].resample("MS")
    sums, counts = m.sum(min_count=1), m.count()
    complete = (counts == counts.index.days_in_month.to_numpy()[:, None]).all(axis=1)
    return sums[complete].reset_index()


def actual_color() -> str:
    """Near-black on a light page, near-white on a dark one, so the actual line always stands out."""
    theme = getattr(getattr(st, "context", None), "theme", None)
    return "#f0f0f0" if getattr(theme, "type", "light") == "dark" else "#222222"


def color_scale(models: list[str]) -> alt.Scale:
    names = [ACTUAL] + [display_name(m) for m in models]
    colors = [actual_color()] + [model_color(m, models) for m in models]
    return alt.Scale(domain=names, range=colors)


def monthly_bar_chart(months: pd.DataFrame, shown: list[str], all_models: list[str]):
    """Bars, not lines: months with missing days are left out, and a line would join
    the months either side of a gap as if it knew what happened in between."""
    long = months.rename(columns={"actual": ACTUAL, **{m: display_name(m) for m in all_models}})
    series = [ACTUAL] + [display_name(m) for m in shown]
    long = long.melt(id_vars="date", value_vars=series, var_name="Series", value_name="kWh")
    long["Month"] = long["date"].dt.strftime("%b %Y")
    month_order = list(dict.fromkeys(long.sort_values("date")["Month"]))
    return alt.Chart(long).mark_bar().encode(
        x=alt.X("Month:N", sort=month_order, title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("Series:N", sort=series),
        y=alt.Y("kWh:Q", title="kWh per month"),
        color=alt.Color("Series:N", scale=color_scale(all_models), sort=series,
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["Month", "Series", alt.Tooltip("kWh:Q", format=".1f")],
    ).properties(height=380)


MUTED_ACTUAL = "#9a9a9a"   # mid grey reads on light and dark pages, and lets the model's colour lead


def model_panel(table: pd.DataFrame, model: str, all_models: list[str], y_max: float):
    """One small chart: a single model against the actual. Every panel shares the same y-axis
    so they can be compared at a glance."""
    name = display_name(model)
    color = model_color(model, all_models)
    long = (table[["date", "actual", model]]
            .rename(columns={"actual": ACTUAL, model: name})
            .melt(id_vars="date", var_name="Series", value_name="kWh"))   # gaps stay in, lines break

    hover = alt.selection_point(fields=["date"], nearest=True, on="pointerover", empty=False)
    x = alt.X("date:T", title=None, axis=alt.Axis(format="%b %Y", grid=False, tickCount="month"))
    lines = alt.Chart(long).mark_line(strokeWidth=1.4).encode(
        x=x,
        y=alt.Y("kWh:Q", title="kWh/day", scale=alt.Scale(domain=[0, y_max])),
        color=alt.Color("Series:N", scale=alt.Scale(domain=[ACTUAL, name], range=[MUTED_ACTUAL, color]),
                        legend=None),
    )
    rule = alt.Chart(long.dropna()).mark_rule(color="#999").encode(
        x=x,
        opacity=alt.condition(hover, alt.value(0.6), alt.value(0)),
        tooltip=[alt.Tooltip("date:T", title="Date", format="%a %d %b %Y"),
                 alt.Tooltip(f"{ACTUAL}:Q", format=".1f"), alt.Tooltip(f"{name}:Q", format=".1f")],
    ).transform_pivot("Series", value="kWh", groupby=["date"]).add_params(hover)
    return (lines + rule).properties(height=230, title=alt.Title(name, anchor="start", color=color))


def line_chart(table: pd.DataFrame, shown: list[str], all_models: list[str]):
    long = table.rename(columns={"actual": ACTUAL, **{m: display_name(m) for m in all_models}})
    series = [ACTUAL] + [display_name(m) for m in shown]
    # Missing days stay in as empty values, which breaks the line at each gap.
    long = long.melt(id_vars="date", value_vars=series, var_name="Series", value_name="kWh")

    hover = alt.selection_point(fields=["date"], nearest=True, on="pointerover", empty=False)
    x = alt.X("date:T", title=None, axis=alt.Axis(format="%d %b %Y"))
    is_actual = alt.datum.Series == ACTUAL
    lines = alt.Chart(long).mark_line().encode(
        x=x,
        y=alt.Y("kWh:Q", title="kWh per day"),
        color=alt.Color("Series:N", scale=color_scale(all_models), legend=alt.Legend(orient="top", title=None)),
        strokeWidth=alt.condition(is_actual, alt.value(2.2), alt.value(1.3)),
        opacity=alt.condition(is_actual, alt.value(1.0), alt.value(0.85)),
    )
    # Invisible vertical rule per day; hovering shows every series' value for that day.
    rule = alt.Chart(long.dropna()).mark_rule(color="#999").encode(
        x=x,
        opacity=alt.condition(hover, alt.value(0.6), alt.value(0)),
        tooltip=[alt.Tooltip("date:T", title="Date", format="%a %d %b %Y")]
        + [alt.Tooltip(f"{s}:Q", format=".1f") for s in series],
    ).transform_pivot("Series", value="kWh", groupby=["date"]).add_params(hover)
    # No wheel zoom: it hijacks page scrolling. The date range slider does the zooming.
    return (lines + rule).properties(height=380)


# ======================================================================================
# Household view
# ======================================================================================

def best_model(table: pd.DataFrame, models: list[str]) -> str:
    """The model with the lowest daily MAE over the whole test period, ignoring the baseline."""
    candidates = [m for m in models if m != NAIVE] or models
    return min(candidates, key=lambda m: scores(table["actual"], table[m])["MAE (kWh)"])


@st.cache_data
def load_daily_explanations(path: str, _mtime: float) -> pd.DataFrame:
    """Per-day SHAP contributions (kWh) regrouped into HOUSEHOLD_DRIVERS. Contributions for groups the
    file doesn't have count as zero; the additivity gap stays in "unexplained_kwh"."""
    df = pd.read_csv(path, parse_dates=["date"])
    out = df[["date", "predicted_kwh", "base_kwh", "unexplained_kwh"]].copy()
    for name, cols in HOUSEHOLD_DRIVERS.items():
        out[name] = df[[c for c in cols if c in df.columns]].sum(axis=1)
    return out


def month_drivers(expl: pd.DataFrame, month_start: pd.Timestamp) -> tuple[float, pd.Series] | None:
    """For one calendar month: the typical-month total (base x days) and each driver's push in kWh.
    Additive, so typical + every push + the small unexplained rest = the month's forecast."""
    in_month = expl[expl["date"].dt.to_period("M") == month_start.to_period("M")]
    if in_month.empty or len(in_month) != month_start.days_in_month:
        return None
    typical = float(in_month["base_kwh"].sum())
    pushes = in_month[list(HOUSEHOLD_DRIVERS)].sum()
    return typical, pushes.reindex(pushes.abs().sort_values(ascending=False).index)


def month_overview_chart(months: pd.DataFrame, model: str, selected: str):
    long = months.assign(Month=months["date"].dt.strftime("%b %Y"))
    long = long.rename(columns={"actual": "What you used", model: "Our forecast"})
    long = long.melt(id_vars=["date", "Month"], value_vars=["What you used", "Our forecast"],
                     var_name="Series", value_name="kWh")
    order = list(dict.fromkeys(long.sort_values("date")["Month"]))
    return alt.Chart(long).mark_bar().encode(
        x=alt.X("Month:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("Series:N"),
        y=alt.Y("kWh:Q", title="kWh per month"),
        color=alt.Color("Series:N", scale=alt.Scale(domain=["What you used", "Our forecast"],
                                                    range=[MUTED_ACTUAL, model_color(model)]),
                        legend=alt.Legend(orient="top", title=None)),
        opacity=alt.condition(alt.datum.Month == selected, alt.value(1.0), alt.value(0.35)),
        tooltip=["Month", "Series", alt.Tooltip("kWh:Q", format=",.0f")],
    ).properties(height=260)


def month_daily_chart(days: pd.DataFrame, model: str):
    long = days.rename(columns={"actual": "What you used", model: "Our forecast"})
    long = long.melt(id_vars="date", value_vars=["What you used", "Our forecast"], var_name="Series", value_name="kWh")
    return alt.Chart(long).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%a %d")),
        y=alt.Y("kWh:Q", title="kWh per day"),
        color=alt.Color("Series:N", scale=alt.Scale(domain=["What you used", "Our forecast"],
                                                    range=[MUTED_ACTUAL, model_color(model)]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("date:T", title="Day", format="%a %d %b"), "Series",
                 alt.Tooltip("kWh:Q", format=".1f")],
    ).properties(height=260)


def drivers_chart(pushes: pd.Series):
    df = pd.DataFrame({"Reason": pushes.index, "kWh": pushes.to_numpy()})
    df["Effect"] = np.where(df["kWh"] >= 0, "Pushed it up", "Pulled it down")
    return alt.Chart(df).mark_bar().encode(
        y=alt.Y("Reason:N", sort=list(df["Reason"]), title=None, axis=alt.Axis(labelLimit=320)),
        x=alt.X("kWh:Q", title="kWh added to or taken off the month"),
        color=alt.Color("Effect:N", scale=alt.Scale(domain=["Pushed it up", "Pulled it down"],
                                                    range=[UP_COLOR, DOWN_COLOR]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["Reason", alt.Tooltip("kWh:Q", format="+,.0f")],
    ).properties(height=40 * len(df) + 40)


def household_view(table: pd.DataFrame, models: list[str]) -> None:
    st.title("Your energy forecast")
    st.caption("How much electricity we expect your home to use, and why.")

    model = best_model(table, models)
    months = monthly(table, models)
    if months.empty:
        st.info("There isn't a complete month of forecasts to show yet.")
        return

    price = st.sidebar.number_input("Your electricity price ($ per kWh)", min_value=0.0, value=DEFAULT_PRICE,
                                    step=0.01, format="%.2f",
                                    help="Find it on your bill. Used only to turn kWh into dollars.")
    labels = list(months["date"].dt.strftime("%B %Y"))
    label = st.sidebar.selectbox("Month", labels, index=len(labels) - 1)
    row = months.iloc[labels.index(label)]
    forecast, used = float(row[model]), float(row["actual"])

    # ---- US1 and US3: the forecast, the bill, and what actually happened
    a, b, c = st.columns(3)
    a.metric(f"Forecast for {label}", f"{forecast:,.0f} kWh", f"about ${forecast * price:,.0f}", delta_color="off", delta_arrow="off", border=True)
    b.metric("What you actually used", f"{used:,.0f} kWh", f"about ${used * price:,.0f}", delta_color="off", delta_arrow="off", border=True)
    diff = (forecast - used) / used * 100
    c.metric("How close the forecast was", f"{abs(diff):.0f}% {'over' if diff > 0 else 'under'}",
             f"{forecast - used:+,.0f} kWh", delta_color="off", delta_arrow="off", border=True)
    typical_error = float(((months[model] - months["actual"]).abs() / months["actual"]).mean() * 100)
    st.caption(f"Across the {len(months)} months we can check, the monthly forecast was off by about "
               f"{typical_error:.0f}% on average.")

    left, right = st.columns(2)
    with left:
        st.markdown(f"**Day by day in {label}**")
        in_month = table[table["date"].dt.to_period("M") == row["date"].to_period("M")]
        st.altair_chart(month_daily_chart(in_month, model), use_container_width=True)
    with right:
        st.markdown("**Month by month**")
        st.altair_chart(month_overview_chart(months, model, row["date"].strftime("%b %Y")),
                        use_container_width=True)
        st.caption("Months with days of missing meter data are left out, because their totals would look too low.")

    # ---- US2: what pushed the forecast up or down
    st.subheader(f"Why {label} looks like this")
    path = RESULTS_DIR / f"{model}_shap_daily.csv"
    found = month_drivers(load_daily_explanations(str(path), path.stat().st_mtime), row["date"]) \
        if path.exists() else None
    if found is None:
        st.info("Explanations for this forecast aren't available yet.")
    else:
        typical, pushes = found
        st.markdown(f"A typical {row['date'].days_in_month}-day month in this home uses about "
                    f"**{typical:,.0f} kWh**. The forecast for {label} is **{forecast:,.0f} kWh**. "
                    "The biggest reasons for the difference:")
        for name, kwh in pushes.head(3).items():
            verb = "added" if kwh > 0 else "took off"
            st.markdown(f"- **{name}** {verb} about **{abs(kwh):,.0f} kWh** (about ${abs(kwh) * price:,.0f}).")
        st.altair_chart(drivers_chart(pushes), use_container_width=True)
        st.caption("Each bar is how much one reason moved this month's forecast away from a typical month.")

    st.caption(f"Forecasts come from the {display_name(model)} model, the most accurate of the "
               f"{len([m for m in models if m != NAIVE])} we tested.")


# ======================================================================================
# Researcher view
# ======================================================================================

def full_period_scores(table: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Daily scores over the whole test period: the table every comparison and export uses."""
    out = pd.DataFrame({display_name(m): scores(table["actual"], table[m]) for m in models}).T
    if NAIVE in models:
        naive_mae = out.loc[display_name(NAIVE), "MAE (kWh)"]
        out["vs naive (%)"] = (1 - out["MAE (kWh)"] / naive_mae) * 100
    out.index.name = "model"
    return out.sort_values("MAE (kWh)")


def lookback_chart(lookback: pd.DataFrame):
    return alt.Chart(lookback).mark_bar(color=model_color("lstm")).encode(
        x=alt.X("day:N", sort=list(lookback["day"]), title="Day of the input window (t-1 = the day before)",
                axis=alt.Axis(labelAngle=0)),
        y=alt.Y("perm_mae_increase_kwh:Q", title="MAE increase when shuffled (kWh)"),
        tooltip=["day", alt.Tooltip("perm_mae_increase_kwh:Q", format=".3f", title="MAE increase"),
                 alt.Tooltip("mean_abs_shap_kwh:Q", format=".3f", title="mean |SHAP|")],
    ).properties(height=280)


def read_csv_if(name: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / name
    return pd.read_csv(path) if path.exists() else None


def show_images(names: list[str]) -> None:
    images = [RESULTS_DIR / n for n in names if (RESULTS_DIR / n).exists()]
    for col, img in zip(st.columns(len(images)) if images else [], images):
        col.image(str(img), caption=img.name)


def explainability_tab() -> None:
    perm, lookback = read_csv_if("lstm_permutation_importance.csv"), read_csv_if("lstm_lookback_importance.csv")
    lstm_shap, xgb_shap = read_csv_if("lstm_shap_summary.csv"), read_csv_if("shap_summary.csv")
    if all(x is None for x in (perm, lookback, lstm_shap, xgb_shap)):
        st.info("No explainability output yet. Run `python src/explain.py --data data/household_daily.csv` "
                "and `python -m src.explain_lstm`.")
        return

    if perm is not None or lookback is not None:
        st.subheader("LSTM")
        left, right = st.columns(2)
        if perm is not None:
            with left:
                st.markdown("**Permutation importance**: test MAE increase when an input's history is shuffled")
                st.dataframe(perm.drop(columns="input").set_index("plain_name").style.format("{:.3f}"),
                             use_container_width=True)
        if lookback is not None:
            with right:
                st.markdown("**Lookback**: meter readings shuffled one window day at a time")
                st.altair_chart(lookback_chart(lookback), use_container_width=True)
    if lstm_shap is not None:
        st.markdown("**SHAP (expected gradients)**, mean |contribution| per input group over the test days")
        st.dataframe(lstm_shap.drop(columns="input").set_index("plain_name"), use_container_width=True)
        show_images(["lstm_permutation_importance.png", "lstm_shap_waterfall_highest_day.png"])
        drivers = RESULTS_DIR / "lstm_drivers.md"
        if drivers.exists():
            with st.expander("Written summary and comparison with XGBoost (lstm_drivers.md)"):
                st.markdown(drivers.read_text(encoding="utf-8"))
    if xgb_shap is not None:
        st.subheader("XGBoost")
        st.markdown("**TreeSHAP**, mean |contribution| per feature over the test days")
        st.dataframe(xgb_shap.set_index("feature"), use_container_width=True)
        show_images(["shap_beeswarm.png", "shap_waterfall_highest_day.png"])

    rolling = read_csv_if("rolling_summary.csv")
    if rolling is not None:
        st.subheader("Robustness: rolling-origin validation")
        st.dataframe(rolling.set_index("model"), use_container_width=True)


def sha256_12(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def export_bundle(table: pd.DataFrame, models: list[str]) -> tuple[bytes, dict]:
    """Every result as one zip, with a manifest naming the run and the data version (FR6).

    The run ID is a hash of every exported file, so the same results always get the same ID and any
    change to a prediction or attribution gives a new one."""
    files: dict[str, bytes] = {}
    files["comparison_scores.csv"] = full_period_scores(table, models).round(4).to_csv().encode()
    merged = table.dropna(subset=["actual"]).rename(columns={"actual": "actual_kwh",
                                                             **{m: f"{m}_kwh" for m in models}})
    files["all_test_predictions.csv"] = merged.to_csv(index=False, date_format="%Y-%m-%d").encode()
    for path in sorted(RESULTS_DIR.glob(PATTERN)):
        files[f"predictions/{path.name}"] = path.read_bytes()
    for name in ATTRIBUTION_FILES:
        if (RESULTS_DIR / name).exists():
            files[f"attributions/{name}"] = (RESULTS_DIR / name).read_bytes()

    run_meta = json.loads((RESULTS_DIR / "run_metadata.json").read_text()) \
        if (RESULTS_DIR / "run_metadata.json").exists() else {}
    manifest = {
        "run_id": sha256_12(b"".join(name.encode() + sha256_12(data).encode() for name, data in sorted(files.items()))),
        "data_file": run_meta.get("data_file"),
        "data_sha256_12": run_meta.get("data_sha256_12"),
        "test_start": str(table.loc[table["actual"].notna(), "date"].min().date()),
        "test_end": str(table.loc[table["actual"].notna(), "date"].max().date()),
        "models": models,
        "files": {name: sha256_12(data) for name, data in sorted(files.items())},
        "exported_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
    return buf.getvalue(), manifest


def export_tab(table: pd.DataFrame, models: list[str]) -> None:
    bundle, manifest = export_bundle(table, models)
    a, b, c = st.columns(3)
    a.metric("Run ID", manifest["run_id"])
    b.metric("Data version (SHA-256)", manifest["data_sha256_12"] or "unknown")
    c.metric("Test period", f"{manifest['test_start']} to {manifest['test_end']}")
    st.download_button("Download everything (.zip)", bundle, file_name=f"energy_forecast_{manifest['run_id']}.zip",
                       mime="application/zip", type="primary")
    st.caption("Predictions for every model, the comparison scores, feature attributions, and manifest.json "
               "with the run ID, data checksum and a checksum for every file.")

    st.markdown("**Single files**")
    scores_csv = full_period_scores(table, models).round(4).to_csv().encode()
    preds = table.dropna(subset=["actual"]).rename(columns={"actual": "actual_kwh", **{m: f"{m}_kwh" for m in models}})
    cols = st.columns(3)
    cols[0].download_button("Comparison scores (.csv)", scores_csv, "comparison_scores.csv", "text/csv")
    cols[1].download_button("All test predictions (.csv)", preds.to_csv(index=False, date_format="%Y-%m-%d").encode(),
                            "all_test_predictions.csv", "text/csv")
    if (RESULTS_DIR / "lstm_shap_daily.csv").exists():
        cols[2].download_button("LSTM SHAP per day (.csv)", (RESULTS_DIR / "lstm_shap_daily.csv").read_bytes(),
                                "lstm_shap_daily.csv", "text/csv")
    with st.expander("manifest.json"):
        st.json(manifest)


def researcher_view(table: pd.DataFrame, models: list[str], problems: list[str], files: list[Path]) -> None:
    st.title("Household energy forecast: model comparison")
    st.caption("Next-day forecasts vs what the meter recorded, over the test period.")
    for p in problems:
        st.warning(p)
    compare, explain, export = st.tabs(["Model comparison", "Explainability", "Export"])
    with explain:
        explainability_tab()
    with export:
        export_tab(table, models)
    with compare:
        model_comparison(table, models, files)


def model_comparison(table: pd.DataFrame, models: list[str], files: list[Path]) -> None:
    # ---- Sidebar controls
    st.sidebar.header("Models")
    shown = [m for m in models if st.sidebar.checkbox(display_name(m), value=True, key=f"show_{m}")]
    st.sidebar.header("View")
    view = st.sidebar.radio("Resolution", ["Daily", "Monthly"], label_visibility="collapsed",
                            help="Monthly adds up each day's forecast, the way a household sees its bill.")
    overlay = view == "Daily" and st.sidebar.toggle(
        "All models on one chart", value=False,
        help="Off: one small chart per model, which is easier to read. On: every line on one chart.")
    first, last = table["date"].min().date(), table["date"].max().date()
    start, end = st.sidebar.slider("Date range", min_value=first, max_value=last, value=(first, last),
                                   format="DD MMM YYYY")
    in_range = table[(table["date"].dt.date >= start) & (table["date"].dt.date <= end)]

    with st.sidebar.expander("Data files"):
        for f in files:
            st.caption(f"`{f.name}`")
        st.caption("Add a model: save `results/<model>_test_predictions.csv` with columns "
                   "`date, actual_kwh, <model>_kwh`, then refresh.")

    # ---- Chart
    if view == "Daily":
        if overlay or not shown:
            st.altair_chart(line_chart(in_range, shown, models), use_container_width=True)
        else:
            y_max = float(np.nanmax(in_range[["actual"] + shown].to_numpy())) * 1.05
            for m in shown:
                st.altair_chart(model_panel(in_range, m, models, y_max), use_container_width=True)
        st.caption("Grey: actual. Hover for values. Gaps are days without enough meter data.")
        shown_table = in_range
    else:
        months = monthly(in_range, models)
        if months.empty:
            st.info("No complete calendar month in this range. Widen the date range.")
            return
        st.altair_chart(monthly_bar_chart(months, shown, models), use_container_width=True)
        st.caption("Complete months only; months with missing meter days would look too low.")
        shown_table = months

    # ---- Scores
    st.subheader("Scores")
    rows = {display_name(m): scores(shown_table["actual"], shown_table[m]) for m in models}
    score_table = pd.DataFrame(rows).T
    n_days = int(score_table["Days"].max())
    score_table = score_table.drop(columns="Days")
    formats = {"MAE (kWh)": "{:.2f}", "RMSE (kWh)": "{:.2f}", "MAPE (%)": "{:.1f}"}
    if NAIVE in models:
        naive_mae = score_table.loc[display_name(NAIVE), "MAE (kWh)"]
        score_table["vs naive"] = (1 - score_table["MAE (kWh)"] / naive_mae) * 100
        # The 15% target only counts on the full daily test period, not a zoomed-in slice.
        if view == "Daily" and (start, end) == (first, last):
            score_table[f"{TARGET_GAIN:.0f}% target"] = [
                "—" if n == display_name(NAIVE) else ("✓ met" if g >= TARGET_GAIN else "✗ not yet")
                for n, g in score_table["vs naive"].items()]
        formats["vs naive"] = "{:+.1f}%"
    st.dataframe(score_table.style.format(formats), use_container_width=True)
    unit = "days" if view == "Daily" else "months"
    st.caption(f"{view} values, {start:%d %b %Y} to {end:%d %b %Y}, {n_days} {unit}. "
               "\"vs naive\" = how much lower the MAE is than the seasonal naive baseline.")

    if view == "Monthly":
        with st.expander("Monthly totals"):
            out = shown_table.set_index("date").rename(columns={"actual": ACTUAL, **{m: display_name(m) for m in models}})
            for m in models:
                out[f"{display_name(m)} error"] = (out[display_name(m)] / out[ACTUAL] - 1) * 100
            out.index = out.index.strftime("%b %Y")
            st.dataframe(out.style.format("{:.1f}"), use_container_width=True)


# ======================================================================================

def persona_selector() -> str:
    """Sidebar switch between the two views, kept in the URL (?view=...) so each can be linked to."""
    options = [HOUSEHOLD, RESEARCHER]
    requested = str(st.query_params.get("view", HOUSEHOLD)).capitalize()
    choice = st.sidebar.radio("Who is this for?", options,
                              index=options.index(requested) if requested in options else 0,
                              captions=["Your forecast and bill", "Models, explanations, exports"])
    st.query_params["view"] = choice.lower()
    st.sidebar.divider()
    return choice


def main() -> None:
    st.set_page_config(page_title="Household energy forecast", page_icon="⚡", layout="wide")
    persona = persona_selector()

    files = sorted(RESULTS_DIR.glob(PATTERN))
    table, models, problems = load_predictions(tuple((str(f), f.stat().st_mtime) for f in files))
    if not models:
        st.error(f"No prediction files found in `{RESULTS_DIR}`. "
                 "Run a model first, for example `python -m src.models.lstm`.")
        return

    if persona == HOUSEHOLD:
        household_view(table, models)
    else:
        researcher_view(table, models, problems, files)


if __name__ == "__main__":   # streamlit runs this file as __main__
    main()

"""Forecast dashboard: every model's test predictions against the actual kWh, on one chart.

Run from the repo root:
    python -m streamlit run src/dashboard.py

It reads every results/<model>_test_predictions.csv. To add a model, write that file in the
shared format (see README, "Dashboard"):
    date, actual_kwh, <model>_kwh
One row per test day. All models must cover the same test days with the same actual values,
otherwise the comparison isn't fair and the page says so.
"""

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
PATTERN = "*_test_predictions.csv"
NAIVE = "naive"           # the baseline the project must beat
TARGET_GAIN = 15.0        # % lower MAE than the naive baseline (Innovation Report)

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
MODEL_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]


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
    colors = [actual_color()] + [MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(len(models))]
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
    color = MODEL_COLORS[all_models.index(model) % len(MODEL_COLORS)]
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


def main() -> None:
    st.set_page_config(page_title="Household energy forecast", page_icon="⚡", layout="wide")
    st.title("Household energy forecast")
    st.caption("Next-day forecasts vs what the meter recorded, over the test period.")

    files = sorted(RESULTS_DIR.glob(PATTERN))
    table, models, problems = load_predictions(tuple((str(f), f.stat().st_mtime) for f in files))
    if not models:
        st.error(f"No prediction files found in `{RESULTS_DIR}`. "
                 "Run a model first, for example `python -m src.models.lstm`.")
        return
    for p in problems:
        st.warning(p)

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


if __name__ == "__main__":   # streamlit runs this file as __main__
    main()

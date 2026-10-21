"""
Build a single self-contained HTML dashboard from the project's results.

No server, no install, no internet. It reads whatever is in results/, data/ and
models/, draws the charts as inline SVG and writes one .html file that opens in
any browser, commits to the repository, and can be emailed to the client.

Sections render only if their input file exists, so this is safe to run at any
point - a missing forecast shows as "not yet generated" rather than crashing.

Usage:  python build_dashboard.py

Writes results/dashboard.html
"""
import argparse
import csv
import html
import json
import math
from pathlib import Path

# ---------------------------------------------------------------- palette
# Validated with the data-viz palette validator: 4 categorical slots pass the
# lightness band, chroma floor, adjacent CVD separation and normal-vision floor
# in both light and dark mode. Light mode warns on contrast for slots 3 and 4,
# so every chart carries direct labels and a table view (the relief rule).
LIGHT = {
    "s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "s4": "#eda100",
    "pos": "#e34948", "neg": "#2a78d6", "mid": "#f0efec",
}
DARK = {
    "s1": "#3987e5", "s2": "#d95926", "s3": "#199e70", "s4": "#c98500",
    "pos": "#e66767", "neg": "#3987e5", "mid": "#383835",
}


def esc(s):
    return html.escape(str(s), quote=True)


def read_rows(path):
    p = Path(path)
    if not p.exists():
        return None
    with p.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows or None


def num(x, nd=2):
    return f"{x:,.{nd}f}" if isinstance(x, (int, float)) else "-"


def missing(what, how):
    return (f'<div class="missing"><strong>{esc(what)} not yet generated.</strong> '
            f'Run <code>{esc(how)}</code> and rebuild this page.</div>')


# ------------------------------------------------------------ svg helpers

def rounded_bar(x, y, w, h, r=4, horizontal=True):
    """A bar with only its DATA end rounded; the baseline end stays square."""
    r = max(0, min(r, (h if horizontal else w) / 2, abs(w if horizontal else h)))
    if horizontal:
        if w <= 0:
            return ""
        return (f"M{x},{y} H{x + w - r} A{r},{r} 0 0 1 {x + w},{y + r} "
                f"V{y + h - r} A{r},{r} 0 0 1 {x + w - r},{y + h} H{x} Z")
    if h <= 0:
        return ""
    return (f"M{x},{y + h} V{y + r} A{r},{r} 0 0 1 {x + r},{y} "
            f"H{x + w - r} A{r},{r} 0 0 1 {x + w},{y + r} V{y + h} Z")


def hbar_chart(items, unit="", label_w=190, bar_h=22, gap=12, width=760,
               highlight=None, note=None):
    """items: list of (label, value, series_role). Values must be >= 0."""
    if not items:
        return ""
    pad_r = 86
    plot_w = width - label_w - pad_r
    vmax = max(v for _, v, _ in items) or 1
    height = len(items) * (bar_h + gap) + 26

    out = [f'<svg viewBox="0 0 {width} {height}" role="img" '
           f'class="chart" preserveAspectRatio="xMinYMin meet">']

    # recessive gridlines at 0, 50, 100%
    for frac in (0, 0.5, 1.0):
        gx = label_w + plot_w * frac
        out.append(f'<line x1="{gx:.1f}" y1="6" x2="{gx:.1f}" '
                   f'y2="{height - 26}" class="grid"/>')
        out.append(f'<text x="{gx:.1f}" y="{height - 10}" class="tick" '
                   f'text-anchor="middle">{num(vmax * frac)}</text>')

    for i, (label, value, role) in enumerate(items):
        y = 6 + i * (bar_h + gap)
        w = plot_w * (value / vmax)
        strong = highlight is not None and label == highlight
        out.append(
            f'<path d="{rounded_bar(label_w, y, w, bar_h)}" '
            f'fill="var(--{role})" class="{"mark strong" if strong else "mark"}">'
            f'<title>{esc(label)}: {num(value)} {esc(unit)}</title></path>')
        out.append(f'<text x="{label_w - 10}" y="{y + bar_h * 0.72}" '
                   f'class="cat" text-anchor="end">{esc(label)}</text>')
        out.append(f'<text x="{label_w + w + 8}" y="{y + bar_h * 0.72}" '
                   f'class="val{" strongtext" if strong else ""}">'
                   f'{num(value)}</text>')

    out.append(f'<line x1="{label_w}" y1="6" x2="{label_w}" '
               f'y2="{height - 26}" class="baseline"/>')
    out.append("</svg>")
    body = f'<div class="chartwrap">{"".join(out)}</div>'
    return body + (f'<p class="note">{note}</p>' if note else "")


def line_chart(series, x_labels, y_label="", width=760, height=300,
               pad=(14, 118, 44, 54)):
    """series: list of dicts {name, values, role, dashed}. One y axis only."""
    if not series:
        return ""
    pt, pr, pb, pl = pad
    pw, ph = width - pl - pr, height - pt - pb

    vals = [v for s in series for v in s["values"] if v is not None]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1
    lo, hi = lo - span * 0.12, hi + span * 0.12
    n = len(x_labels)

    def X(i):
        return pl + (pw * i / (n - 1) if n > 1 else pw / 2)

    def Y(v):
        return pt + ph * (1 - (v - lo) / (hi - lo))

    out = [f'<svg viewBox="0 0 {width} {height}" role="img" class="chart" '
           f'preserveAspectRatio="xMinYMin meet">']

    for k in range(5):
        v = lo + (hi - lo) * k / 4
        y = Y(v)
        out.append(f'<line x1="{pl}" y1="{y:.1f}" x2="{pl + pw}" '
                   f'y2="{y:.1f}" class="grid"/>')
        out.append(f'<text x="{pl - 8}" y="{y + 4:.1f}" class="tick" '
                   f'text-anchor="end">{num(v)}</text>')

    for i, lab in enumerate(x_labels):
        out.append(f'<text x="{X(i):.1f}" y="{pt + ph + 22}" class="tick" '
                   f'text-anchor="middle">{esc(lab)}</text>')

    ends = []
    for s in series:
        pts = [(X(i), Y(v)) for i, v in enumerate(s["values"]) if v is not None]
        if not pts:
            continue
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        dash = ' stroke-dasharray="6 4"' if s.get("dashed") else ""
        out.append(f'<path d="{d}" fill="none" stroke="var(--{s["role"]})" '
                   f'stroke-width="2" stroke-linejoin="round"{dash}/>')
        for (x, y), v in zip(pts, [v for v in s["values"] if v is not None]):
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" '
                       f'fill="var(--{s["role"]})" class="ring">'
                       f'<title>{esc(s["name"])} - {num(v)}</title></circle>')
        ends.append([pts[-1][0], pts[-1][1], s["name"]])

    # Direct labels at the series ends - identity is never colour alone.
    # Nudge them apart so two close lines do not print on top of each other.
    ends.sort(key=lambda e: e[1])
    MINGAP = 15
    for i in range(1, len(ends)):
        if ends[i][1] - ends[i - 1][1] < MINGAP:
            ends[i][1] = ends[i - 1][1] + MINGAP
    overflow = ends[-1][1] - (pt + ph) if ends else 0
    if overflow > 0:
        for e in ends:
            e[1] -= overflow
    for lx, ly, name in ends:
        out.append(f'<text x="{lx + 10:.1f}" y="{ly + 4:.1f}" '
                   f'class="dlabel">{esc(name)}</text>')

    out.append(f'<line x1="{pl}" y1="{pt + ph}" x2="{pl + pw}" '
               f'y2="{pt + ph}" class="baseline"/>')
    if y_label:
        out.append(f'<text x="{pl}" y="10" class="axtitle">{esc(y_label)}</text>')
    out.append("</svg>")
    return f'<div class="chartwrap">{"".join(out)}</div>' 


def diverging_chart(items, noise=None, unit="kWh", width=760, label_w=190,
                    bar_h=22, gap=12):
    """items: list of (label, signed_value). Blue = fell, red = rose, gray zero."""
    if not items:
        return ""
    pad_r = 96
    plot_w = width - label_w - pad_r
    amax = max(abs(v) for _, v in items) or 1
    amax = max(amax, noise or 0) * 1.15
    # If every change has the same sign, half the chart would be empty, so the
    # zero line moves to that edge instead of sitting uselessly in the middle.
    has_pos = any(v > 0 for _, v in items)
    has_neg = any(v < 0 for _, v in items)
    if has_pos and has_neg:
        zero, half = label_w + plot_w / 2, plot_w / 2
    elif has_neg:
        zero, half = label_w + plot_w, plot_w
    else:
        zero, half = label_w, plot_w
    height = len(items) * (bar_h + gap) + 30

    out = [f'<svg viewBox="0 0 {width} {height}" role="img" class="chart" '
           f'preserveAspectRatio="xMinYMin meet">']

    if noise:
        signs = [s for s, on in ((1, has_pos), (-1, has_neg)) if on]
        for sign in signs:
            nx = zero + half * (noise / amax) * sign
            out.append(f'<line x1="{nx:.1f}" y1="4" x2="{nx:.1f}" '
                       f'y2="{height - 30}" class="noise"/>')
        lx = zero + half * (noise / amax) * signs[0]
        anchor = "start" if signs[0] > 0 else "end"
        out.append(f'<text x="{lx + 6 * signs[0]:.1f}" y="{height - 14}" '
                   f'class="tick" text-anchor="{anchor}">fold-to-fold '
                   f'noise</text>')

    for i, (label, value) in enumerate(items):
        y = 4 + i * (bar_h + gap)
        w = half * (abs(value) / amax)
        if value >= 0:
            d = rounded_bar(zero, y, w, bar_h)
            role, tx, anchor = "pos", zero + w + 8, "start"
        else:
            # mirror: round the left (data) end
            r = min(4, bar_h / 2, w)
            d = (f"M{zero},{y} H{zero - w + r} A{r},{r} 0 0 0 {zero - w},{y + r} "
                 f"V{y + bar_h - r} A{r},{r} 0 0 0 {zero - w + r},{y + bar_h} "
                 f"H{zero} Z") if w > 0 else ""
            role, tx, anchor = "neg", zero - w - 8, "end"

        verdict = ("error rose" if value > 0 else "error fell")
        out.append(f'<path d="{d}" fill="var(--{role})" class="mark">'
                   f'<title>{esc(label)}: {verdict} {num(abs(value))} '
                   f'{esc(unit)}</title></path>')
        out.append(f'<text x="{label_w - 10}" y="{y + bar_h * 0.72}" '
                   f'class="cat" text-anchor="end">{esc(label)}</text>')
        out.append(f'<text x="{tx:.1f}" y="{y + bar_h * 0.72}" class="val" '
                   f'text-anchor="{anchor}">{value:+.3f}</text>')

    out.append(f'<line x1="{zero:.1f}" y1="4" x2="{zero:.1f}" '
               f'y2="{height - 30}" class="baseline"/>')
    out.append("</svg>")
    return f'<div class="chartwrap">{"".join(out)}</div>' 


def forecast_chart(obs, fc, width=760, height=300, pad=(16, 110, 40, 54)):
    """obs: [(date, kwh)] tail of history. fc: [(date, pred, lo, hi)]."""
    if not fc:
        return ""
    pt, pr, pb, pl = pad
    pw, ph = width - pl - pr, height - pt - pb

    vals = ([v for _, v in obs] + [p for _, p, _, _ in fc]
            + [lo for _, _, lo, _ in fc] + [hi for _, _, _, hi in fc])
    lo_v, hi_v = min(vals), max(vals)
    span = (hi_v - lo_v) or 1
    lo_v, hi_v = lo_v - span * 0.08, hi_v + span * 0.08

    n = len(obs) + len(fc)

    def X(i):
        return pl + pw * i / max(1, n - 1)

    def Y(v):
        return pt + ph * (1 - (v - lo_v) / (hi_v - lo_v))

    out = [f'<svg viewBox="0 0 {width} {height}" role="img" class="chart" '
           f'preserveAspectRatio="xMinYMin meet">']

    for k in range(5):
        v = lo_v + (hi_v - lo_v) * k / 4
        y = Y(v)
        out.append(f'<line x1="{pl}" y1="{y:.1f}" x2="{pl + pw}" y2="{y:.1f}" '
                   f'class="grid"/>')
        out.append(f'<text x="{pl - 8}" y="{y + 4:.1f}" class="tick" '
                   f'text-anchor="end">{num(v, 0)}</text>')

    # uncertainty band
    off = len(obs)
    up = [(X(off + i), Y(hi)) for i, (_, _, _, hi) in enumerate(fc)]
    dn = [(X(off + i), Y(lo)) for i, (_, _, lo, _) in enumerate(fc)][::-1]
    band = ("M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in up + dn) + " Z")
    out.append(f'<path d="{band}" fill="var(--s2)" opacity="0.16"/>')

    d_obs = "M" + " L".join(f"{X(i):.1f},{Y(v):.1f}"
                            for i, (_, v) in enumerate(obs))
    out.append(f'<path d="{d_obs}" fill="none" stroke="var(--s1)" '
               f'stroke-width="2" stroke-linejoin="round"/>')

    d_fc = "M" + " L".join(f"{X(off + i):.1f},{Y(p):.1f}"
                           for i, (_, p, _, _) in enumerate(fc))
    out.append(f'<path d="{d_fc}" fill="none" stroke="var(--s2)" '
               f'stroke-width="2" stroke-linejoin="round"/>')

    for i, (dt, p, lo, hi) in enumerate(fc):
        out.append(f'<circle cx="{X(off + i):.1f}" cy="{Y(p):.1f}" r="3.5" '
                   f'fill="var(--s2)" class="ring"><title>{esc(dt)}: '
                   f'{num(p)} kWh ({num(lo)}-{num(hi)})</title></circle>')

    sx = X(off - 1) if obs else pl
    out.append(f'<line x1="{sx:.1f}" y1="{pt}" x2="{sx:.1f}" '
               f'y2="{pt + ph}" class="noise"/>')
    out.append(f'<text x="{X(max(0, off // 2)):.1f}" y="{pt + ph + 20}" '
               f'class="tick" text-anchor="middle">observed</text>')
    out.append(f'<text x="{X(off + len(fc) // 2):.1f}" y="{pt + ph + 20}" '
               f'class="tick" text-anchor="middle">forecast</text>')

    # The two regions are already captioned under the axis, so no end labels -
    # one of them would land on the divider and collide with the other series.
    out.append(f'<line x1="{pl}" y1="{pt + ph}" x2="{pl + pw}" '
               f'y2="{pt + ph}" class="baseline"/>')
    out.append(f'<text x="{pl}" y="11" class="axtitle">kWh per day</text>')
    out.append("</svg>")
    return f'<div class="chartwrap">{"".join(out)}</div>' 


def table(rows, cols=None, nd=3):
    if not rows:
        return ""
    cols = cols or list(rows[0].keys())
    head = "".join(f"<th>{esc(c)}</th>" for c in cols)
    body = []
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append(f"<td>{num(v, nd) if isinstance(v, float) else esc(v)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def tile(value, label, sub=""):
    return (f'<div class="tile"><div class="tval">{value}</div>'
            f'<div class="tlab">{esc(label)}</div>'
            f'{f"<div class=tsub>{esc(sub)}</div>" if sub else ""}</div>')


# ------------------------------------------------------------------- build

def main(root, outfile):
    root = Path(root)
    R, D, M = root / "results", root / "data", root / "models"

    comparison = read_rows(R / "comparison_table.csv")
    rolling_s = read_rows(R / "rolling_summary.csv")
    rolling_f = read_rows(R / "rolling_folds.csv")
    ablation = read_rows(R / "ablation.csv")
    mvd = read_rows(R / "monthly_vs_daily.csv")
    monthly = read_rows(R / "monthly_report.csv")
    shap = read_rows(R / "shap_summary.csv")
    forecast = read_rows(R / "forecast.csv")
    fc_month = read_rows(R / "forecast_monthly.csv")
    daily = read_rows(D / "household_daily.csv")

    man = {}
    if (D / "manifest.json").exists():
        man = json.loads((D / "manifest.json").read_text())

    parts = []

    # ---- stat tiles
    tiles = []
    if comparison:
        base = next((r for r in comparison
                     if "naive" in str(r.get("model", "")).lower()), None)
        best = min((r for r in comparison if r is not base),
                   key=lambda r: r["MAE"], default=None)
        if best:
            tiles.append(tile(f'{num(best["MAE"])}<span class="u">kWh</span>',
                              "Best single-split MAE", str(best["model"])))
        if base and best:
            imp = (base["MAE"] - best["MAE"]) / base["MAE"] * 100
            tiles.append(tile(f'{num(imp,1)}<span class="u">%</span>',
                              "Better than baseline",
                              f'vs seasonal naive {num(base["MAE"])} kWh'))
    if mvd:
        dly = next((r for r in mvd if r["level"] == "daily"), None)
        mth = next((r for r in mvd if r["level"] == "monthly"), None)
        if dly and mth:
            tiles.append(tile(f'{num(mth["MAPE_%"],1)}<span class="u">%</span>',
                              "Monthly error (MAPE)",
                              f'daily is {num(dly["MAPE_%"],1)}% - errors cancel'))
    if forecast:
        f0 = forecast[0]
        tiles.append(tile(f'{num(f0["predicted_kwh"])}<span class="u">kWh</span>',
                          "Next-day forecast",
                          f'{f0.get("date","")}  +/- {num(f0["expected_MAE"])}'))
    if man:
        tiles.append(tile(f'{int(man.get("daily_rows", 0)):,}',
                          "Usable days", f'{man.get("date_from","")} to '
                                         f'{man.get("date_to","")}'))
    if tiles:
        parts.append(f'<div class="tiles">{"".join(tiles)}</div>')

    # ---- model comparison
    parts.append('<section><h2>Model comparison</h2>'
                 '<p class="lede">Mean absolute error on the held-out test '
                 'period, in kWh per day. Lower is better. The seasonal-naive '
                 'baseline is what any model has to beat to be worth '
                 'anything.</p>')
    if comparison:
        items, best_name = [], None
        srt = sorted(comparison, key=lambda r: r["MAE"])
        for r in srt:
            name = str(r["model"])
            is_base = "naive" in name.lower()
            items.append((name, r["MAE"], "mut" if is_base else "s1"))
            if not is_base and best_name is None:
                best_name = name
        parts.append(hbar_chart(items, unit="kWh/day", highlight=best_name))
        parts.append(table(comparison))
    else:
        parts.append(missing("Model comparison",
                             "python src/forecast_pipeline.py --data data/household_daily.csv"))
    parts.append("</section>")

    # ---- rolling origin
    parts.append('<section><h2>Rolling-origin validation</h2>'
                 '<p class="lede">The comparison above uses one test window. '
                 'Here the window walks forward through time: each fold trains '
                 'only on days before its test block. Five independent reads '
                 'instead of one, and the spread between them is the honest '
                 'statement of reliability.</p>')
    if rolling_f:
        folds = sorted({int(r["fold"]) for r in rolling_f})
        names = []
        for r in rolling_f:
            if r["model"] not in names:
                names.append(r["model"])
        roles = ["s1", "s2", "s3", "s4"]
        series = []
        for i, nm in enumerate(names):
            vals = []
            for f in folds:
                hit = next((r for r in rolling_f
                            if r["model"] == nm and int(r["fold"]) == f), None)
                vals.append(hit["MAE"] if hit else None)
            series.append({
                "name": nm, "values": vals,
                "role": "mut" if "naive" in str(nm).lower() else roles[i % 4],
                "dashed": "naive" in str(nm).lower(),
            })
        parts.append(line_chart(series, [f"fold {f}" for f in folds],
                                y_label="MAE (kWh/day)"))
        if rolling_s:
            ml = [r for r in rolling_s
                  if "naive" not in str(r["model"]).lower()]
            if ml:
                spread = max(r["mean_MAE"] for r in ml) - min(r["mean_MAE"] for r in ml)
                noise = sum(r["std_MAE"] for r in ml) / len(ml)
                verdict = ("The models are statistically indistinguishable: the "
                           f"best-to-worst gap is {num(spread,3)} kWh against a "
                           f"fold-to-fold standard deviation of {num(noise,3)}. "
                           "The finding is the improvement over the baseline, "
                           "not a winning model."
                           if spread < 0.25 * noise else
                           f"The gap between models ({num(spread,3)} kWh) is "
                           f"large relative to the fold-to-fold noise "
                           f"({num(noise,3)}), so naming a best model is "
                           f"defensible.")
                parts.append(f'<p class="callout">{verdict}</p>')
            parts.append(table(rolling_s,
                               ["model", "mean_MAE", "std_MAE", "worst_MAE",
                                "mean_MAPE", "beat_baseline_%"]))
    else:
        parts.append(missing("Rolling-origin validation",
                             "python src/rolling_validation.py --data data/household_daily.csv"))
    parts.append("</section>")

    # ---- ablation
    parts.append('<section><h2>Which features actually earn their place</h2>'
                 '<p class="lede">Each group of features is removed and the '
                 'model retrained across all five folds. Bars to the right mean '
                 'the error rose without that group, so it carried signal. Bars '
                 'inside the dotted lines moved less than the model\'s own '
                 'fold-to-fold variation, which means the group was '
                 'redundant.</p>')
    if ablation:
        base = next((r for r in ablation
                     if r["feature_set"] == "all features"), None)
        items = [(str(r["feature_set"]).replace("no ", "without "),
                  r["MAE_change"])
                 for r in sorted(ablation, key=lambda r: r["MAE_change"])
                 if r is not base]
        parts.append(diverging_chart(items,
                                     noise=base["std_MAE"] if base else None))
        parts.append(table(ablation,
                           ["feature_set", "n_features", "mean_MAE",
                            "std_MAE", "MAE_change", "MAE_change_%"]))
    else:
        parts.append(missing("Ablation",
                             "python src/ablation.py --data data/household_daily.csv"))
    parts.append("</section>")

    # ---- drivers
    parts.append('<section><h2>What drives a day\'s forecast</h2>'
                 '<p class="lede">Mean absolute SHAP value: how far each '
                 'feature moves a single day\'s prediction away from the average '
                 'prediction, in kWh. Read alongside the ablation above - a '
                 'large SHAP value on a feature the ablation calls redundant '
                 'means the credit is shared with a correlated feature, not that '
                 'the feature is irreplaceable.</p>')
    if shap:
        top = sorted(shap, key=lambda r: -r["mean_abs_shap_kwh"])[:8]
        items = [(str(r.get("plain_name") or r["feature"]),
                  r["mean_abs_shap_kwh"], "s3") for r in top]
        parts.append(hbar_chart(items, unit="kWh", label_w=260))
        parts.append(table(top, ["plain_name", "mean_abs_shap_kwh",
                                 "max_abs_shap_kwh", "direction"], nd=3))
    else:
        parts.append(missing("SHAP drivers",
                             "python src/explain.py --data data/household_daily.csv"))
    parts.append("</section>")

    # ---- forecast
    parts.append('<section><h2>Forecast</h2>'
                 '<p class="lede">Tomorrow comes from the full-feature model, '
                 'which can use yesterday\'s end-of-day meter readings. Every '
                 'day after that is produced recursively by the reduced model, '
                 'feeding each prediction back in as the next day\'s lag. The '
                 'band widens with the square root of the step count: it is an '
                 'estimate of accumulating error, not a measured prediction '
                 'interval.</p>')
    if forecast:
        obs = []
        if daily:
            obs = [(str(r["date"])[:10], r["kwh"]) for r in daily[-60:]]
        fc = [(str(r["date"])[:10], r["predicted_kwh"],
               r["lower_kwh"], r["upper_kwh"]) for r in forecast]
        parts.append(forecast_chart(obs, fc))
        if fc_month:
            parts.append('<h3>Monthly projection</h3>'
                         '<p class="lede">A month the forecast only partly '
                         'covers is flagged - a stub month is not a monthly '
                         'total.</p>')
            parts.append(table(fc_month, nd=1))
        parts.append('<h3>Daily detail</h3>')
        parts.append(table(forecast[:31],
                           ["date", "step", "predicted_kwh", "lower_kwh",
                            "upper_kwh", "model"]))
    else:
        parts.append(missing("Forecast",
                             "python src/train_model.py --data data/household_daily.csv "
                             "&& python src/predict.py --data data/household_daily.csv"))
    parts.append("</section>")

    # ---- monthly reporting
    if monthly:
        parts.append('<section><h2>Monthly reporting on the test period</h2>'
                     '<p class="lede">The model is fitted on days because there '
                     'are about 1,400 of them and only 47 months. The client '
                     'plans at month scale. Daily errors partly cancel when '
                     'summed over a month, which is why the monthly percentage '
                     'error is the lower of the two.</p>')
        parts.append(table(monthly, nd=1))
        if mvd:
            parts.append(table(mvd))
        parts.append("</section>")

    # ---- provenance
    prov = []
    if man:
        prov.append(
            f'<li><strong>{int(man.get("daily_rows",0)):,} usable days</strong> '
            f'from {esc(man.get("date_from",""))} to '
            f'{esc(man.get("date_to",""))}, built from '
            f'{int(man.get("raw_rows",0)):,} raw minute-level readings</li>')
        if man.get("raw_sha256"):
            prov.append(f'<li>Raw file SHA-256 '
                        f'<code>{esc(str(man["raw_sha256"])[:24])}...</code> - '
                        f'the exact source these numbers came from</li>')
        prov.append(f'<li>A calendar day is kept only if at least '
                    f'{float(man.get("min_coverage_rule", man.get("coverage_rule", 0.9)))*100:.0f}% '
                    f'of its 1,440 minutes were recorded. Partial days are '
                    f'dropped, never interpolated</li>')
    prov.append('<li>Source: UCI <em>Individual household electric power '
                'consumption</em> (Hebrail &amp; Berard 2012), CC BY 4.0. '
                'Modified: minute readings aggregated to daily totals under the '
                'coverage rule above</li>')
    prov.append('<li>All splits chronological; no random k-fold. Leakage '
                'assertions run on every build</li>')
    parts.append('<section><h2>Provenance</h2><ul class="prov">'
                 + "".join(prov) + '</ul></section>')

    card = ""
    if (M / "model_card.md").exists():
        card = (M / "model_card.md").read_text()
        lim = card.split("## Limits")
        if len(lim) > 1:
            body = lim[1].split("##")[0].strip()
            # Bullets wrap across lines in the markdown source, so join each
            # one back together before stripping the ** and ` markers.
            bullets, cur = [], None
            for raw in body.splitlines():
                t = raw.strip()
                if t.startswith("- "):
                    if cur:
                        bullets.append(cur)
                    cur = t[2:]
                elif t and cur is not None:
                    cur += " " + t
                elif not t and cur:
                    bullets.append(cur); cur = None
            if cur:
                bullets.append(cur)

            def demark(t):
                return (t.replace("**", "").replace("`", "")
                         .replace("*", "").strip())

            bullets = [demark(b) for b in bullets if demark(b)]
            if bullets:
                parts.append('<section><h2>Limits</h2><ul class="prov">'
                             + "".join(f"<li>{esc(b)}</li>" for b in bullets)
                             + "</ul></section>")

    css = """
:root{color-scheme:light;
 --page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--mut:#898781;
 --grid:#e1e0d9;--base:#c3c2b7;--ring:rgba(11,11,11,.10);
 --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;
 --pos:#e34948;--neg:#2a78d6}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
 color-scheme:dark;
 --page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--mut:#898781;
 --grid:#2c2c2a;--base:#383835;--ring:rgba(255,255,255,.10);
 --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;
 --pos:#e66767;--neg:#3987e5}}
:root[data-theme=dark]{color-scheme:dark;
 --page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--mut:#898781;
 --grid:#2c2c2a;--base:#383835;--ring:rgba(255,255,255,.10);
 --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;
 --pos:#e66767;--neg:#3987e5}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);
 font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:860px;margin:0 auto;padding:40px 16px 72px}
header{margin-bottom:28px}
h1{font-size:27px;line-height:1.25;margin:0 0 6px;letter-spacing:-.01em}
.sub{color:var(--ink2);margin:0 0 2px}
.meta{color:var(--mut);font-size:13px;margin:4px 0 0}
section{background:var(--surface);border:1px solid var(--ring);border-radius:10px;
 padding:22px 22px 18px;margin:18px 0}
h2{font-size:18px;margin:0 0 6px;letter-spacing:-.005em}
h3{font-size:14px;margin:22px 0 4px;color:var(--ink2);
 text-transform:uppercase;letter-spacing:.06em}
.lede{color:var(--ink2);margin:0 0 16px;font-size:14px;max-width:62ch}
.callout{margin:14px 0 0;padding:12px 14px;border-left:3px solid var(--s1);
 background:var(--page);border-radius:0 6px 6px 0;font-size:14px;color:var(--ink)}
.missing{padding:14px;background:var(--page);border:1px dashed var(--base);
 border-radius:8px;color:var(--ink2);font-size:14px}
.tiles{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(152px,1fr));
 margin-bottom:6px}
.tile{background:var(--surface);border:1px solid var(--ring);border-radius:10px;
 padding:16px 16px 14px}
.tval{font-size:30px;line-height:1.1;letter-spacing:-.02em}
.tval .u{font-size:14px;color:var(--mut);margin-left:3px;letter-spacing:0}
.tlab{font-size:13px;color:var(--ink2);margin-top:5px}
.tsub{font-size:12px;color:var(--mut);margin-top:2px}
.chartwrap{overflow-x:auto;overflow-y:hidden;-webkit-overflow-scrolling:touch;
 padding-bottom:2px}
.chart{width:100%;height:auto;display:block;overflow:visible}
@media (max-width:720px){.chart{min-width:640px}
 .chartwrap{margin:0 -6px;padding:0 6px}}
.grid{stroke:var(--grid);stroke-width:1}
.baseline{stroke:var(--base);stroke-width:1}
.noise{stroke:var(--mut);stroke-width:1;stroke-dasharray:3 3}
.tick{fill:var(--mut);font-size:11px}
.cat{fill:var(--ink2);font-size:12.5px}
.val{fill:var(--ink2);font-size:12px;font-variant-numeric:tabular-nums}
.strongtext{fill:var(--ink);font-weight:600}
.dlabel{fill:var(--ink2);font-size:11.5px}
.axtitle{fill:var(--mut);font-size:11px}
.mark{fill-opacity:.9}.mark.strong{fill-opacity:1}
.mark:hover{fill-opacity:1}
.ring{stroke:var(--surface);stroke-width:2}
svg [fill="var(--mut)"]{fill:var(--mut)}
.tablewrap{overflow-x:auto;margin-top:14px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--grid);
 white-space:nowrap}
th{color:var(--mut);font-weight:600;font-size:11px;text-transform:uppercase;
 letter-spacing:.05em}
td{font-variant-numeric:tabular-nums;color:var(--ink2)}
ul.prov{margin:0;padding-left:20px;color:var(--ink2);font-size:14px}
ul.prov li{margin-bottom:7px}
code{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;
 background:var(--page);padding:1px 5px;border-radius:4px}
.note{color:var(--mut);font-size:12px;margin:8px 0 0}
footer{color:var(--mut);font-size:12.5px;margin-top:26px;text-align:center}
@media (max-width:640px){.wrap{padding:24px 16px 56px}h1{font-size:22px}
 .tval{font-size:25px}}
"""

    mut_note = ""  # --mut is already defined; charts reference it by role

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Household Energy Forecasting</title>
<style>{css}</style></head>
<body>
<div class="wrap">
<header>
<h1>Household Energy Usage Prediction</h1>
<p class="sub">Daily electricity forecasting for the ARC Research Hub for
Integrated Energy Storage Solutions</p>
<p class="meta">Forecasting 5 &middot; ICT30016 ICT Innovation Project &middot;
Swinburne University of Technology</p>
</header>
{''.join(parts)}
<footer>Generated by <code>src/build_dashboard.py</code> from the files in
<code>results/</code>, <code>data/</code> and <code>models/</code>.
Hover any mark for its exact value. Every chart is backed by the table beneath
it.</footer>
</div></body></html>"""

    out = Path(outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)

    have = sum(x is not None for x in (comparison, rolling_f, ablation, shap,
                                       forecast, monthly))
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB, "
          f"{have}/6 sections populated)")
    for name, data in (("comparison_table", comparison),
                       ("rolling_folds", rolling_f),
                       ("ablation", ablation),
                       ("shap_summary", shap),
                       ("forecast", forecast),
                       ("monthly_report", monthly)):
        print(f"  {'ok     ' if data else 'missing'} {name}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="results/dashboard.html")
    a = ap.parse_args()
    main(a.root, a.out)

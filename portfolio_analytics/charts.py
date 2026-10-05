"""Plotly figures for the dashboard. Colours follow one palette so the portfolio is always blue,
the benchmark always orange, and losses or breaches always red."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a86"
RED, AMBER, GOOD = "#d03b3b", "#fab219", "#0ca30c"
DIVERGING = [[0, RED], [0.5, "#f4f4f2"], [1, GOOD]]
CORR_SCALE = [[0, BLUE], [0.5, "#f4f4f2"], [1, RED]]


def _layout(fig: go.Figure, title: str | None = None, y_pct: bool = True, height: int = 380) -> go.Figure:
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=50 if title else 20, b=10),
                      hovermode="x unified", legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
    if y_pct:
        fig.update_yaxes(tickformat=".0%")
    return fig


def growth(port: pd.Series, bench: pd.Series | None, bench_name: str = "Nifty 50") -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=port.index, y=100 * (1 + port).cumprod(), name="Portfolio", line=dict(color=BLUE, width=2))
    if bench is not None:
        fig.add_scatter(x=bench.index, y=100 * (1 + bench).cumprod(), name=bench_name,
                        line=dict(color=ORANGE, width=1.5))
    fig.update_yaxes(tickprefix="₹")
    return _layout(fig, "Growth of ₹100", y_pct=False)


def drawdown(port_dd: pd.Series, bench_dd: pd.Series | None, bench_name: str = "Nifty 50") -> go.Figure:
    fig = go.Figure()
    if bench_dd is not None:
        fig.add_scatter(x=bench_dd.index, y=bench_dd, name=bench_name, line=dict(color=ORANGE, width=1))
    fig.add_scatter(x=port_dd.index, y=port_dd, name="Portfolio", fill="tozeroy",
                    line=dict(color=BLUE, width=1.5), fillcolor="rgba(42,120,214,0.15)")
    return _layout(fig, "Drawdown from previous peak", height=300).update_layout(legend_traceorder="reversed")


def monthly_heatmap(table: pd.DataFrame) -> go.Figure:
    z = table.to_numpy(dtype=float)
    lim = np.nanmax(np.abs(z)) if np.isfinite(z).any() else 0.1
    text = [[f"{v:.1%}" if np.isfinite(v) else "" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(z=z, x=list(table.columns), y=[str(i) for i in table.index], text=text,
                               texttemplate="%{text}", colorscale=DIVERGING, zmin=-lim, zmax=lim,
                               showscale=False, hovertemplate="%{y} %{x}: %{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_xaxes(side="top")
    return _layout(fig, None, y_pct=False, height=60 + 32 * len(table)).update_layout(hovermode="closest")


def return_distribution(r: pd.Series, var_lines: dict[str, float]) -> go.Figure:
    """Histogram of daily returns with VaR / ES thresholds (given as positive losses)."""
    fig = go.Figure(go.Histogram(x=r, nbinsx=120, marker_color=BLUE, opacity=0.75, name="Daily returns",
                                 hovertemplate="%{x:.2%}: %{y} days<extra></extra>"))
    colours = [AMBER, RED, "#7a1f1f", GREY]
    for (name, v), c in zip(var_lines.items(), colours):
        fig.add_vline(x=-v, line=dict(color=c, width=2, dash="dash"))
        # Empty trace so each threshold gets a legend entry instead of a label over the bars
        fig.add_scatter(x=[None], y=[None], mode="lines", name=f"{name}: {v:.2%}",
                        line=dict(color=c, width=2, dash="dash"))
    fig.update_xaxes(tickformat=".1%", range=[r.quantile(0.0005) * 1.3, r.quantile(0.9995) * 1.3])
    fig.data[0].showlegend = False
    return _layout(fig, "Distribution of daily portfolio returns", y_pct=False).update_layout(
        hovermode="closest", bargap=0.02)


def risk_vs_weight(contrib: pd.DataFrame) -> go.Figure:
    df = contrib.sort_values("Risk contribution")
    fig = go.Figure()
    fig.add_bar(y=df.index, x=df["Weight"], name="Weight", orientation="h", marker_color=GREY)
    fig.add_bar(y=df.index, x=df["Risk contribution"], name="Share of VaR", orientation="h", marker_color=BLUE)
    fig.update_xaxes(tickformat=".0%")
    return _layout(fig, "Weight vs share of portfolio risk", y_pct=False, height=60 + 34 * len(df)).update_layout(
        barmode="group", hovermode="y unified")


def correlation(corr: pd.DataFrame) -> go.Figure:
    text = [[f"{v:.2f}" for v in row] for row in corr.to_numpy()]
    fig = go.Figure(go.Heatmap(z=corr.to_numpy(), x=list(corr.columns), y=list(corr.index), text=text,
                               texttemplate="%{text}", colorscale=CORR_SCALE, zmin=-1, zmax=1,
                               hovertemplate="%{y} / %{x}: %{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "Correlation of daily returns", y_pct=False, height=120 + 40 * len(corr)).update_layout(
        hovermode="closest")


def backtest(series: pd.DataFrame, title: str) -> go.Figure:
    hits = series[series["Exception"]]
    fig = go.Figure()
    fig.add_bar(x=series.index, y=series["Return"], name="Daily return", marker_color=GREY, opacity=0.6)
    fig.add_scatter(x=series.index, y=series["VaR"], name="VaR forecast (loss threshold)",
                    line=dict(color=BLUE, width=1.5))
    fig.add_scatter(x=hits.index, y=hits["Return"], mode="markers", name="Exception",
                    marker=dict(color=RED, size=7, symbol="x"))
    return _layout(fig, title, height=420).update_layout(bargap=0)


def rolling(df: pd.DataFrame, col: str, title: str, pct: bool, ref: float | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=df.index, y=df[col], name=col, line=dict(color=BLUE, width=1.5))
    if ref is not None:
        fig.add_hline(y=ref, line=dict(color=GREY, dash="dot"))
    fig = _layout(fig, title, y_pct=pct, height=280).update_layout(showlegend=False)
    if not pct:
        fig.update_yaxes(tickformat=".2f")
    return fig


def frontier(front: pd.DataFrame, points: dict[str, dict], assets: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=assets["Volatility"], y=assets["Expected return"], mode="markers+text", text=assets.index,
                    textposition="top center", name="Single holdings", textfont=dict(size=10, color=GREY),
                    marker=dict(color=GREY, size=7))
    fig.add_scatter(x=front["Volatility"], y=front["Expected return"], mode="lines", name="Efficient frontier",
                    line=dict(color=BLUE, width=2.5))
    colours = {"Your portfolio": ORANGE, "Min variance": GREEN, "Max Sharpe": RED, "Risk parity": AMBER,
               "Equal weight": GREY}
    for name, s in points.items():
        fig.add_scatter(x=[s["Volatility"]], y=[s["Expected return"]], mode="markers", name=name,
                        marker=dict(size=14, color=colours.get(name, BLUE), line=dict(color="white", width=1.5),
                                    symbol="star" if name == "Your portfolio" else "circle"))
    ys = list(assets["Expected return"]) + list(front["Expected return"]) + [s["Expected return"] for s in points.values()]
    pad = (max(ys) - min(ys)) * 0.08
    fig.update_yaxes(range=[min(ys) - pad, max(ys) + pad])
    fig.update_xaxes(tickformat=".0%", title="Annual volatility")
    fig.update_yaxes(title="Expected annual return")
    return _layout(fig, "Efficient frontier (in-sample)", height=460).update_layout(hovermode="closest")


def weights_bars(weights: pd.DataFrame) -> go.Figure:
    colours = [ORANGE, GREEN, RED, AMBER, GREY]
    fig = go.Figure()
    for col, c in zip(weights.columns, colours):
        fig.add_bar(x=weights.index, y=weights[col], name=col, marker_color=c)
    return _layout(fig, "Weights by strategy", height=360).update_layout(barmode="group")


def stress_bars(df: pd.DataFrame, bench_name: str = "Benchmark") -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=df["Scenario"], y=df["Portfolio"], name="Portfolio", marker_color=BLUE,
                text=[f"{v:.1%}" for v in df["Portfolio"]], textposition="outside")
    fig.add_bar(x=df["Scenario"], y=df["Benchmark"], name=bench_name, marker_color=ORANGE,
                text=[f"{v:.1%}" for v in df["Benchmark"]], textposition="outside")
    return _layout(fig, "Return through each stress episode", height=400).update_layout(barmode="group")

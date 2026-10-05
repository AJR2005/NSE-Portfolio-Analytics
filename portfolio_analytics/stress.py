"""Stress tests: how the portfolio behaved in real Indian market shocks, and hypothetical market moves."""
from __future__ import annotations

import pandas as pd

from .metrics import capm

# Peak-to-trough windows of real NSE stress episodes (dates are trading days on the Nifty 50).
SCENARIOS = [
    {"name": "COVID-19 crash", "start": "2020-02-19", "end": "2020-03-23",
     "note": "Nifty 50 fell about 38% in five weeks as lockdowns began."},
    {"name": "2022 rate-hike sell-off", "start": "2022-01-17", "end": "2022-06-17",
     "note": "Global inflation, Fed rate hikes and the Russia-Ukraine war; heavy FPI selling."},
    {"name": "Adani-Hindenburg report", "start": "2023-01-24", "end": "2023-02-28",
     "note": "Short-seller report hit Adani group stocks and banking sentiment."},
    {"name": "2024 election-result day", "start": "2024-06-03", "end": "2024-06-04",
     "note": "Nifty 50 fell about 6% in one day when results missed exit polls."},
    {"name": "Late-2024 FPI outflow", "start": "2024-09-26", "end": "2024-11-21",
     "note": "Record foreign outflows after the September 2024 peak."},
]


def historical_scenarios(asset_returns: pd.DataFrame, port_r: pd.Series, bench_r: pd.Series) -> pd.DataFrame:
    """Compounded return over each scenario window, for the portfolio, the benchmark and each holding.
    Scenarios outside the loaded date range are skipped."""
    rows = []
    for s in SCENARIOS:
        # Returns dated after the start close up to and including the end close.
        start, end = pd.Timestamp(s["start"]), pd.Timestamp(s["end"])
        window = port_r.loc[(port_r.index > start) & (port_r.index <= end)]
        # Skip scenarios not fully covered by the loaded history.
        if window.empty or port_r.index[0] > start or port_r.index[-1] < end:
            continue
        idx = window.index
        row = {"Scenario": s["name"], "From": start.date(), "To": end.date(), "Note": s["note"],
               "Portfolio": float((1 + window).prod() - 1),
               "Benchmark": float((1 + bench_r.reindex(idx).fillna(0)).prod() - 1)}
        assets = (1 + asset_returns.reindex(idx).fillna(0)).prod() - 1
        row["Worst holding"] = assets.idxmin()
        row["Worst holding return"] = float(assets.min())
        rows.append(row)
    return pd.DataFrame(rows)


def beta_shock(asset_returns: pd.DataFrame, weights: pd.Series, bench_r: pd.Series,
               shocks=(-0.05, -0.10, -0.20, 0.10)) -> tuple[pd.DataFrame, pd.Series]:
    """Expected portfolio move if the Nifty 50 moves by each shock, using each holding's beta."""
    betas = pd.Series({c: capm(asset_returns[c], bench_r)["beta"] for c in asset_returns.columns})
    w = weights.reindex(asset_returns.columns).fillna(0)
    port_beta = float((w * betas).sum())
    rows = [{"Nifty 50 move": s, "Expected portfolio move": port_beta * s} for s in shocks]
    return pd.DataFrame(rows), betas

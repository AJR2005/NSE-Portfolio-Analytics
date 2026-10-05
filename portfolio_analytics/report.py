"""Excel risk report: one workbook with every table the dashboard shows."""
from __future__ import annotations

import io

import pandas as pd

from . import metrics, risk, stress
from .data import label


def build_report(prices: pd.DataFrame, weights: pd.Series, bench_r: pd.Series | None, rf: float,
                 rebalance: str | None = "M", confidence: float = 0.99) -> bytes:
    asset_r = metrics.returns(prices[weights.index])
    port_r = metrics.portfolio_returns(asset_r, weights, rebalance)
    if bench_r is not None:
        bench_r = bench_r.reindex(port_r.index).fillna(0)

    summary = pd.Series(metrics.summary(port_r, rf, bench_r), name="Portfolio").to_frame()
    if bench_r is not None:
        summary["Nifty 50"] = pd.Series(metrics.summary(bench_r, rf))
    holdings = pd.DataFrame({t: metrics.summary(asset_r[t], rf) for t in asset_r.columns}).T
    holdings.insert(0, "Weight", weights)
    holdings.index = [label(t) for t in holdings.index]
    contrib = risk.risk_contributions(asset_r, weights, confidence)
    contrib.index = [label(t) for t in contrib.index]
    bt = pd.DataFrame([{k: v for k, v in vars(risk.backtest_var(port_r, m, confidence)).items() if k != "series"}
                       for m in ("historical", "parametric", "cornish-fisher")])

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        summary.to_excel(xl, sheet_name="Summary")
        holdings.to_excel(xl, sheet_name="Holdings")
        risk.var_table(port_r, asset_r, weights).to_excel(xl, sheet_name="VaR", index=False)
        contrib.to_excel(xl, sheet_name="Risk contributions")
        bt.to_excel(xl, sheet_name="VaR backtest", index=False)
        metrics.calendar_returns(port_r).to_excel(xl, sheet_name="Monthly returns")
        if bench_r is not None:
            stress.historical_scenarios(asset_r, port_r, bench_r).to_excel(xl, sheet_name="Stress tests", index=False)
        pd.DataFrame({"Portfolio": port_r, "Drawdown": metrics.drawdown(port_r)}).to_excel(xl, sheet_name="Daily")
    return buf.getvalue()

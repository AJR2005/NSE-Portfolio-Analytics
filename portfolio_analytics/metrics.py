"""Return and performance metrics.

Conventions: daily simple returns, 252 trading days a year, risk-free rate given as an annual
decimal (e.g. 0.065) and converted to a daily rate where needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")


def portfolio_returns(asset_returns: pd.DataFrame, weights: pd.Series, rebalance: str | None = "M") -> pd.Series:
    """Daily portfolio returns.

    rebalance=None  -> buy and hold: weights drift with prices after day one.
    rebalance="M"   -> reset to target weights at the start of each month ("Q" quarterly, "D" daily).
    """
    weights = weights.reindex(asset_returns.columns).fillna(0.0)
    weights = weights / weights.sum()
    if rebalance == "D":
        return (asset_returns * weights).sum(axis=1).rename("Portfolio")

    out = []
    if rebalance is None:
        periods = [asset_returns]
    else:
        freq = {"M": "M", "Q": "Q", "Y": "Y"}[rebalance]
        periods = [g for _, g in asset_returns.groupby(asset_returns.index.to_period(freq))]
    for block in periods:
        # Within a holding period each position grows with its own price; the portfolio's
        # daily return is the change in total value.
        growth = (1 + block).cumprod()
        value = (growth * weights).sum(axis=1)
        prev = pd.concat([pd.Series([1.0], index=[block.index[0] - pd.Timedelta(days=1)]), value.iloc[:-1]])
        out.append(pd.Series(value.values / prev.values - 1, index=block.index))
    return pd.concat(out).rename("Portfolio")


def growth_of(r: pd.Series | pd.DataFrame, start: float = 100.0):
    return start * (1 + r).cumprod()


def cagr(r: pd.Series) -> float:
    years = len(r) / TRADING_DAYS
    return float((1 + r).prod() ** (1 / years) - 1) if years > 0 else np.nan


def annual_volatility(r: pd.Series) -> float:
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe(r: pd.Series, rf: float = 0.0) -> float:
    """Annualised Sharpe ratio: mean daily excess return / its standard deviation, scaled by sqrt(252)."""
    excess = r - rf / TRADING_DAYS
    sd = excess.std(ddof=1)
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan


def sortino(r: pd.Series, rf: float = 0.0) -> float:
    """Like Sharpe, but only downside deviation counts as risk."""
    excess = r - rf / TRADING_DAYS
    downside = np.sqrt((np.minimum(excess, 0) ** 2).mean())
    return float(excess.mean() / downside * np.sqrt(TRADING_DAYS)) if downside > 0 else np.nan


def drawdown(r: pd.Series) -> pd.Series:
    """Percentage below the running peak of the growth curve, at every date (0 = at a new high)."""
    wealth = (1 + r).cumprod()
    return wealth / wealth.cummax() - 1


def max_drawdown_details(r: pd.Series) -> dict:
    dd = drawdown(r)
    trough = dd.idxmin()
    wealth = (1 + r).cumprod()
    peak = wealth.loc[:trough].idxmax()
    after = dd.loc[trough:]
    recovered = after[after >= 0]
    recovery = recovered.index[0] if len(recovered) else None
    return {
        "max_drawdown": float(dd.min()),
        "peak": peak,
        "trough": trough,
        "recovery": recovery,
        "days_to_trough": int(len(dd.loc[peak:trough]) - 1),
        "days_to_recover": int(len(dd.loc[trough:recovery]) - 1) if recovery is not None else None,
    }


def calmar(r: pd.Series) -> float:
    mdd = drawdown(r).min()
    return float(cagr(r) / abs(mdd)) if mdd < 0 else np.nan


def capm(r: pd.Series, bench: pd.Series, rf: float = 0.0) -> dict:
    """Beta and Jensen's alpha from an OLS regression of excess returns on the benchmark's excess returns."""
    df = pd.concat([r, bench], axis=1, join="inner").dropna()
    y = df.iloc[:, 0] - rf / TRADING_DAYS
    x = df.iloc[:, 1] - rf / TRADING_DAYS
    beta = float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))
    alpha_daily = float(y.mean() - beta * x.mean())
    resid = y - (alpha_daily + beta * x)
    r2 = float(1 - resid.var() / y.var())
    active = df.iloc[:, 0] - df.iloc[:, 1]
    te = float(active.std(ddof=1) * np.sqrt(TRADING_DAYS))
    return {
        "beta": beta,
        "alpha": alpha_daily * TRADING_DAYS,
        "r_squared": r2,
        "correlation": float(df.iloc[:, 0].corr(df.iloc[:, 1])),
        "tracking_error": te,
        "information_ratio": float(active.mean() * TRADING_DAYS / te) if te > 0 else np.nan,
    }


def summary(r: pd.Series, rf: float = 0.0, bench: pd.Series | None = None) -> dict:
    dd = max_drawdown_details(r)
    out = {
        "Total return": float((1 + r).prod() - 1),
        "CAGR": cagr(r),
        "Annual volatility": annual_volatility(r),
        "Sharpe ratio": sharpe(r, rf),
        "Sortino ratio": sortino(r, rf),
        "Max drawdown": dd["max_drawdown"],
        "Calmar ratio": calmar(r),
        "Skewness": float(r.skew()),
        "Excess kurtosis": float(r.kurt()),
        "Best day": float(r.max()),
        "Worst day": float(r.min()),
        "Positive days": float((r > 0).mean()),
    }
    if bench is not None:
        c = capm(r, bench, rf)
        out.update({"Beta": c["beta"], "Alpha (annual)": c["alpha"], "Tracking error": c["tracking_error"],
                    "Information ratio": c["information_ratio"]})
    return out


def rolling_metrics(r: pd.Series, bench: pd.Series | None, rf: float = 0.0, window: int = 126) -> pd.DataFrame:
    excess = r - rf / TRADING_DAYS
    out = pd.DataFrame({
        "Volatility": r.rolling(window).std() * np.sqrt(TRADING_DAYS),
        "Sharpe": excess.rolling(window).mean() / excess.rolling(window).std() * np.sqrt(TRADING_DAYS),
    })
    if bench is not None:
        b = bench.reindex(r.index)
        out["Beta"] = r.rolling(window).cov(b) / b.rolling(window).var()
    return out.dropna()


MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def calendar_returns(r: pd.Series) -> pd.DataFrame:
    """Monthly returns as a year x month table, plus the full-year return."""
    monthly = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
    table = monthly.unstack().reindex(columns=range(1, 13))
    table.columns = MONTHS
    table["Year"] = (1 + r).groupby(r.index.year).prod() - 1
    table.index.name = None
    return table

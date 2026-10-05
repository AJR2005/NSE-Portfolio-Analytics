"""Price data: live from Yahoo Finance via yfinance, or the bundled NSE snapshot for offline use."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = Path(os.environ.get("NSE_SNAPSHOT", ROOT / "data" / "nse_prices.csv"))
CACHE_DIR = ROOT / "data" / "cache"

BENCHMARK = "^NSEI"  # Nifty 50 index
DEFAULT_BASKET: dict[str, float] = {
    "RELIANCE.NS": 0.08,
    "HDFCBANK.NS": 0.08,
    "ICICIBANK.NS": 0.08,
    "INFY.NS": 0.08,
    "TCS.NS": 0.08,
    "ITC.NS": 0.08,
    "LT.NS": 0.08,
    "BHARTIARTL.NS": 0.08,
    "NIFTYBEES.NS": 0.20,  # Nifty 50 ETF
    "GOLDBEES.NS": 0.16,  # Gold ETF
}
NAMES = {
    "RELIANCE.NS": "Reliance Industries", "HDFCBANK.NS": "HDFC Bank", "ICICIBANK.NS": "ICICI Bank",
    "INFY.NS": "Infosys", "TCS.NS": "TCS", "ITC.NS": "ITC", "LT.NS": "Larsen & Toubro",
    "BHARTIARTL.NS": "Bharti Airtel", "NIFTYBEES.NS": "Nippon Nifty 50 ETF", "GOLDBEES.NS": "Nippon Gold ETF",
    "BANKBEES.NS": "Nippon Bank ETF", "JUNIORBEES.NS": "Nippon Nifty Next 50 ETF", "^NSEI": "Nifty 50",
    "^NSEBANK": "Nifty Bank", "SBIN.NS": "State Bank of India", "HINDUNILVR.NS": "Hindustan Unilever",
    "KOTAKBANK.NS": "Kotak Mahindra Bank", "AXISBANK.NS": "Axis Bank", "MARUTI.NS": "Maruti Suzuki",
    "SUNPHARMA.NS": "Sun Pharma", "ASIANPAINT.NS": "Asian Paints", "BAJFINANCE.NS": "Bajaj Finance",
    "TATAMOTORS.NS": "Tata Motors", "WIPRO.NS": "Wipro", "HCLTECH.NS": "HCL Technologies",
}
# Extra tickers kept in the snapshot so the dashboard can swap holdings offline.
SNAPSHOT_TICKERS = sorted(set(DEFAULT_BASKET) | {BENCHMARK, "^NSEBANK", "BANKBEES.NS", "JUNIORBEES.NS",
                                                  "SBIN.NS", "HINDUNILVR.NS", "KOTAKBANK.NS", "AXISBANK.NS",
                                                  "MARUTI.NS", "SUNPHARMA.NS", "ASIANPAINT.NS", "BAJFINANCE.NS",
                                                  "WIPRO.NS", "HCLTECH.NS"})


def label(ticker: str) -> str:
    return NAMES.get(ticker, ticker.replace(".NS", ""))


@dataclass
class PriceData:
    prices: pd.DataFrame  # adjusted close, one column per ticker, trading days as index
    source: str
    dropped: list[str] = field(default_factory=list)  # tickers with no usable data
    repaired: dict[str, int] = field(default_factory=dict)  # bad ticks removed, per ticker


def download(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Adjusted closes from Yahoo Finance. NSE tickers end in .NS; indices start with ^."""
    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, threads=True)
    if raw.empty:
        return pd.DataFrame()
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": tickers[0]})
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.sort_index()


def load_snapshot() -> pd.DataFrame:
    df = pd.read_csv(SNAPSHOT, index_col=0, parse_dates=True)
    return df.sort_index()


def remove_bad_ticks(prices: pd.DataFrame, tolerance: float = 0.4) -> tuple[pd.DataFrame, dict[str, int]]:
    """Blanks out isolated bad quotes, e.g. Yahoo showing an ETF at 1/10th or 1/100th of its price for a
    couple of days around a split. A price more than 40% away from the median of the surrounding
    11 days cannot be real for a large-cap or ETF (NSE price bands and index circuit breakers stop that),
    while a genuine crash moves the median along with it."""
    median = prices.rolling(11, center=True, min_periods=3).median()
    bad = ((prices / median - 1).abs() > tolerance) & prices.notna()
    counts = {c: int(n) for c, n in bad.sum().items() if n}
    return prices.mask(bad), counts


def clean(prices: pd.DataFrame, min_coverage: float = 0.9) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    """Aligns tickers on common trading days.

    Stock and index holidays differ slightly, so small gaps are forward-filled (a missing day = no
    price change). Tickers with too little history for the window are dropped rather than silently
    shortening every other series.
    """
    prices = prices.dropna(how="all")
    # Rows where most tickers have no price are exchange holidays with a stray quote for one or two
    # instruments; keeping them would insert fake zero-return days for everything else.
    prices = prices.loc[prices.notna().mean(axis=1) >= 0.5]
    prices, repaired = remove_bad_ticks(prices)
    coverage = prices.notna().mean()
    dropped = coverage[coverage < min_coverage].index.tolist()
    prices = prices.drop(columns=dropped).ffill(limit=5).dropna()
    return prices, dropped, repaired


def load_prices(tickers: list[str], start: str, end: str | None = None, live: bool = True) -> PriceData:
    """Live download first; falls back to the bundled snapshot (e.g. offline or rate-limited)."""
    tickers = list(dict.fromkeys(tickers))
    prices, source = pd.DataFrame(), "snapshot"
    if live:
        try:
            prices = download(tickers, start, end)
            source = "live"
        except Exception as exc:  # network error, Yahoo rate limit, API change
            log.warning("Live download failed (%s); using bundled snapshot", exc)
    missing = [t for t in tickers if t not in prices.columns or prices[t].dropna().empty]
    if missing and SNAPSHOT.exists():
        snap = load_snapshot()
        have = [t for t in missing if t in snap.columns]
        if have:
            fill = snap.loc[snap.index >= pd.Timestamp(start), have]
            if end:
                fill = fill.loc[fill.index <= pd.Timestamp(end)]
            prices = fill if prices.empty else prices.join(fill, how="outer")
            source = "snapshot" if source == "snapshot" or len(have) == len(tickers) else "live + snapshot"
    cols = [t for t in tickers if t in prices.columns]
    cleaned, dropped, repaired = clean(prices[cols]) if cols else (pd.DataFrame(), [], {})
    dropped += [t for t in tickers if t not in cols]
    return PriceData(cleaned, source, dropped, repaired)

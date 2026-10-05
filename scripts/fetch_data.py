"""Refreshes data/nse_prices.csv from Yahoo Finance. Run monthly by .github/workflows/update-data.yml.

    python scripts/fetch_data.py [--start 2019-01-01]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from portfolio_analytics.data import SNAPSHOT, SNAPSHOT_TICKERS, download  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01-01", help="covers the 2020 COVID crash for stress tests")
    args = ap.parse_args()

    prices = download(SNAPSHOT_TICKERS, args.start)
    if prices.empty:
        print("Download returned no data", file=sys.stderr)
        return 1
    prices = prices.dropna(how="all")
    empty = [t for t in SNAPSHOT_TICKERS if t not in prices.columns or prices[t].dropna().empty]
    if empty:
        print(f"No data for: {', '.join(empty)}", file=sys.stderr)
    if "^NSEI" in empty or len(empty) > len(SNAPSHOT_TICKERS) // 2:
        print("Too many tickers failed; keeping the existing snapshot", file=sys.stderr)
        return 1
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    prices.round(4).to_csv(SNAPSHOT, index_label="Date")
    print(f"Wrote {SNAPSHOT.name}: {len(prices)} days x {prices.shape[1]} tickers, "
          f"{prices.index[0]:%Y-%m-%d} to {prices.index[-1]:%Y-%m-%d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

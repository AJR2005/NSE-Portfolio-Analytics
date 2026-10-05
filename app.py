"""NSE Portfolio Analytics: Streamlit dashboard.  Run with:  streamlit run app.py"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from portfolio_analytics import charts, metrics, optimize, risk, stress
from portfolio_analytics.data import DEFAULT_BASKET, PriceData, label, load_prices
from portfolio_analytics.report import build_report

st.set_page_config(page_title="NSE Portfolio Analytics", page_icon="📈", layout="wide")

FETCH_START = "2019-01-01"
PERIODS = {"1Y": 1, "3Y": 3, "5Y": 5, "Max": None}
REBALANCE = {"Monthly": "M", "Quarterly": "Q", "Yearly": "Y", "Buy and hold": None}
BENCHMARKS = {"Nifty 50": "^NSEI", "Nifty Bank": "^NSEBANK"}
PCT_KEYS = {"Total return", "CAGR", "Annual volatility", "Max drawdown", "Best day", "Worst day",
            "Positive days", "Alpha (annual)", "Tracking error"}
PLOT = {"width": "stretch", "config": {"displayModeBar": False}}


# ---------- helpers ----------
def normalise_ticker(t: str) -> str:
    """'infy' -> 'INFY.NS'; indices (^NSEI) and tickers with an exchange suffix are kept as typed."""
    t = str(t).strip().upper()
    if not t or t == "NONE":
        return ""
    return t if t.startswith("^") or "." in t else f"{t}.NS"


def inr(x: float) -> str:
    """Indian digit grouping: 1234567 -> ₹12,34,567."""
    sign, x = ("-" if x < 0 else ""), abs(round(x))
    s = str(int(x))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        head = ",".join([head[max(0, i - 2):i] for i in range(len(head), 0, -2)][::-1])
        s = f"{head},{tail}"
    return f"{sign}₹{s}"


def fmt_metric(key: str, v: float) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{v:.2%}" if key in PCT_KEYS else f"{v:.2f}"


def fmt_table(df: pd.DataFrame, pct_cols=(), num_cols=()) -> pd.DataFrame:
    out = df.copy()
    for c in pct_cols:
        out[c] = out[c].map(lambda v: f"{v:.2%}" if pd.notna(v) else "–")
    for c in num_cols:
        out[c] = out[c].map(lambda v: f"{v:.2f}" if pd.notna(v) else "–")
    return out


@st.cache_data(ttl=6 * 3600, show_spinner="Loading NSE prices…")
def get_prices(tickers: tuple[str, ...], live: bool) -> PriceData:
    return load_prices(list(tickers), FETCH_START, live=live)


# ---------- sidebar ----------
with st.sidebar:
    st.header("Portfolio")
    default = pd.DataFrame({"Ticker": [t.replace(".NS", "") for t in DEFAULT_BASKET],
                            "Weight %": [w * 100 for w in DEFAULT_BASKET.values()]})
    edited = st.data_editor(
        default, num_rows="dynamic", hide_index=True, width="stretch", key="holdings",
        column_config={"Ticker": st.column_config.TextColumn(help="NSE symbol, e.g. INFY or NIFTYBEES"),
                       "Weight %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=1.0,
                                                                format="%.1f")})
    bench_name = st.selectbox("Benchmark", list(BENCHMARKS))
    period = st.segmented_control("Period", list(PERIODS), default="5Y") or "5Y"
    rebalance_name = st.selectbox("Rebalancing", list(REBALANCE), help="How often weights are reset to target.")
    rf = st.number_input("Risk-free rate (% a year)", 0.0, 15.0, 6.5, 0.25,
                         help="91-day T-bill yield is the usual proxy in India.") / 100
    st.header("Risk settings")
    confidence = st.select_slider("VaR confidence", [0.90, 0.95, 0.975, 0.99], value=0.99,
                                  format_func=lambda c: f"{c:.1%}".replace(".0%", "%"))
    horizon = st.select_slider("VaR horizon (trading days)", [1, 5, 10, 21], value=1)
    notional = st.number_input("Portfolio value (₹)", 10_000, 1_000_000_000, 1_000_000, 100_000)
    st.header("Data")
    live = st.toggle("Live prices from Yahoo Finance", value=False,
                     help="Off: bundled NSE snapshot, refreshed monthly by GitHub Actions. "
                          "On: fresh download, falling back to the snapshot if Yahoo is unreachable.")

# ---------- data ----------
holdings = edited.assign(Ticker=edited["Ticker"].map(normalise_ticker)).dropna()
holdings = holdings[(holdings["Ticker"] != "") & (holdings["Weight %"] > 0)].groupby("Ticker", sort=False).sum()
if holdings.empty:
    st.warning("Add at least one holding with a positive weight.")
    st.stop()
target = holdings["Weight %"] / holdings["Weight %"].sum()
bench_ticker = BENCHMARKS[bench_name]

data = get_prices(tuple(sorted(set(target.index) | {bench_ticker})), live)
prices = data.prices
if PERIODS[period] and not prices.empty:
    prices = prices.loc[prices.index >= prices.index[-1] - pd.DateOffset(years=PERIODS[period])]
missing = [t for t in target.index if t not in prices.columns]
if missing:
    hint = "" if live else " Turn on live prices to fetch tickers that aren't in the snapshot."
    st.warning(f"No price data for {', '.join(missing)}; left out and the other weights rescaled.{hint}")
    target = target.drop(missing)
    target = target / target.sum()
if target.empty or bench_ticker not in prices.columns or len(prices) < 300:
    st.error("Not enough price history to analyse. Check the tickers or choose a longer period.")
    st.stop()

asset_r = metrics.returns(prices[target.index])
bench_r = metrics.returns(prices[[bench_ticker]])[bench_ticker].reindex(asset_r.index)
port_r = metrics.portfolio_returns(asset_r, target, REBALANCE[rebalance_name])
names = {t: label(t) for t in prices.columns}

# ---------- header ----------
st.title("NSE Portfolio Analytics")
src = {"live": "live Yahoo Finance prices", "snapshot": "bundled NSE snapshot",
       "live + snapshot": "live prices plus the bundled snapshot"}[data.source]
st.caption(f"{len(target)} holdings · {asset_r.index[0]:%d %b %Y} to {asset_r.index[-1]:%d %b %Y} "
           f"({len(asset_r):,} trading days) · {rebalance_name.lower()} rebalancing · {src} · "
           f"risk-free {rf:.2%}")

s_port = metrics.summary(port_r, rf, bench_r)
s_bench = metrics.summary(bench_r, rf)
var_h, es_h = risk.historical_var(port_r, confidence, horizon)
conf_txt = f"{confidence:.1%}".replace(".0%", "%")

k = st.columns(6)
k[0].metric("CAGR", f"{s_port['CAGR']:.1%}", f"{s_port['CAGR'] - s_bench['CAGR']:+.1%} vs {bench_name}")
k[1].metric("Volatility", f"{s_port['Annual volatility']:.1%}",
            f"{s_port['Annual volatility'] - s_bench['Annual volatility']:+.1%}", delta_color="inverse")
k[2].metric("Sharpe ratio", f"{s_port['Sharpe ratio']:.2f}", f"{s_port['Sharpe ratio'] - s_bench['Sharpe ratio']:+.2f}")
k[3].metric("Max drawdown", f"{s_port['Max drawdown']:.1%}",
            f"{s_port['Max drawdown'] - s_bench['Max drawdown']:+.1%}")
k[4].metric("Beta", f"{s_port['Beta']:.2f}", help=f"Sensitivity to the {bench_name}")
k[5].metric(f"{horizon}-day VaR ({conf_txt})", inr(var_h * notional), f"ES {inr(es_h * notional)}",
            delta_color="off", delta_arrow="off", help="Historical VaR on the portfolio value set in the sidebar.")

tabs = st.tabs(["Overview", "Risk & VaR", "VaR backtest", "Rolling", "Optimiser", "Stress tests", "Download"])

# ---------- Overview ----------
with tabs[0]:
    st.plotly_chart(charts.growth(port_r, bench_r, bench_name), **PLOT)
    st.plotly_chart(charts.drawdown(metrics.drawdown(port_r), metrics.drawdown(bench_r), bench_name), **PLOT)
    dd = metrics.max_drawdown_details(port_r)
    rec = (f"recovered on {dd['recovery']:%d %b %Y} after {dd['days_to_recover']} trading days"
           if dd["recovery"] is not None else "and has not yet recovered to the old peak")
    st.markdown(f"**Worst fall:** {dd['max_drawdown']:.1%} from the peak on {dd['peak']:%d %b %Y} to the trough on "
                f"{dd['trough']:%d %b %Y} ({dd['days_to_trough']} trading days), {rec}.")

    c1, c2 = st.columns([2, 3])
    with c1:
        st.subheader("Summary")
        rows = [{"Metric": key, "Portfolio": fmt_metric(key, v), bench_name: fmt_metric(key, s_bench.get(key))}
                for key, v in s_port.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=600)
    with c2:
        st.subheader("Monthly returns")
        st.plotly_chart(charts.monthly_heatmap(metrics.calendar_returns(port_r)), **PLOT)
    st.subheader("Holdings")
    h = pd.DataFrame({t: metrics.summary(asset_r[t], rf, bench_r) for t in asset_r.columns}).T
    h = h[["Total return", "CAGR", "Annual volatility", "Sharpe ratio", "Max drawdown", "Beta"]]
    h.insert(0, "Weight", target)
    h.insert(0, "Ticker", [t.replace(".NS", "") for t in h.index])
    h.index = [names[t] for t in h.index]
    st.dataframe(fmt_table(h, ["Weight", "Total return", "CAGR", "Annual volatility", "Max drawdown"],
                           ["Sharpe ratio", "Beta"]), width="stretch")

# ---------- Risk & VaR ----------
with tabs[1]:
    st.subheader(f"{horizon}-day Value at Risk on {inr(notional)}")
    vt = risk.var_table(port_r, asset_r, target, confidences=(0.95, 0.99) if confidence in (0.95, 0.99)
                        else (0.95, confidence), horizon=horizon)
    vt["Confidence"] = vt["Confidence"].map(lambda c: f"{c:.1%}".replace(".0%", "%"))
    vt["VaR (₹)"] = (vt["VaR"] * notional).map(inr)
    vt["ES (₹)"] = (vt["Expected Shortfall"] * notional).map(inr)
    st.dataframe(fmt_table(vt, ["VaR", "Expected Shortfall"]), hide_index=True, width="stretch")
    hist99 = risk.historical_var(port_r, 0.99)[0]
    norm99 = risk.parametric_var(port_r, 0.99)[0]
    st.caption(
        f"VaR is the loss not exceeded on {conf_txt} of {horizon}-day periods; Expected Shortfall is the average "
        f"loss on the days that do exceed it. Daily returns here have skew {s_port['Skewness']:.2f} and excess "
        f"kurtosis {s_port['Excess kurtosis']:.1f}, so the 99% historical VaR is "
        f"{'above' if hist99 > norm99 else 'below'} the normal-distribution estimate "
        f"({hist99:.2%} vs {norm99:.2%}). Multi-day parametric and Monte Carlo VaR use square-root-of-time scaling; "
        f"historical VaR uses overlapping {horizon}-day returns.")
    if s_port["Excess kurtosis"] > 6:
        st.warning(f"Excess kurtosis is {s_port['Excess kurtosis']:.1f}. The Cornish-Fisher expansion is only "
                   "reliable for moderate skew and kurtosis (roughly below 6), so treat its numbers as an "
                   "upper bound here; the historical estimate is the better guide.")
    lines = {"VaR 95%": risk.historical_var(port_r, 0.95)[0], "VaR 99%": hist99,
             "ES 99%": risk.historical_var(port_r, 0.99)[1]}
    st.plotly_chart(charts.return_distribution(port_r, lines), **PLOT)

    c1, c2 = st.columns(2)
    with c1:
        contrib = risk.risk_contributions(asset_r, target, confidence).rename(index=names)
        st.plotly_chart(charts.risk_vs_weight(contrib), **PLOT)
        top = contrib.index[0]
        st.caption(f"Euler decomposition of {conf_txt} parametric VaR: components add up to the total. "
                   f"{top} is {contrib.loc[top, 'Weight']:.0%} of the money but "
                   f"{contrib.loc[top, 'Risk contribution']:.0%} of the risk.")
    with c2:
        st.plotly_chart(charts.correlation(asset_r.rename(columns=names).corr()), **PLOT)

# ---------- VaR backtest ----------
with tabs[2]:
    st.subheader("Does the VaR model hold up out of sample?")
    b1, b2 = st.columns([1, 1])
    method = b1.radio("Model", ["historical", "parametric", "cornish-fisher"], horizontal=True,
                      format_func=lambda m: {"historical": "Historical", "parametric": "Parametric (normal)",
                                             "cornish-fisher": "Cornish-Fisher"}[m])
    window = b2.select_slider("Estimation window (trading days)", [125, 250, 500], value=250)
    if len(port_r) <= window + 50:
        st.info("Choose a longer period to backtest with this window.")
    else:
        bt = risk.backtest_var(port_r, method, confidence, window)
        m = st.columns(5)
        m[0].metric("Days tested", f"{bt.observations:,}")
        m[1].metric("Exceptions", bt.exceptions, f"{bt.exceptions - bt.expected:+.1f} vs {bt.expected:.1f} expected",
                    delta_color="inverse")
        m[2].metric("Kupiec p-value", f"{bt.kupiec_p:.3f}", "pass" if bt.kupiec_p > 0.05 else "reject",
                    delta_color="normal" if bt.kupiec_p > 0.05 else "inverse")
        m[3].metric("Christoffersen p-value", f"{bt.christoffersen_p:.3f}",
                    "independent" if bt.christoffersen_p > 0.05 else "clustered",
                    delta_color="normal" if bt.christoffersen_p > 0.05 else "inverse")
        light = {"green": "🟢 Green", "yellow": "🟡 Yellow", "red": "🔴 Red"}.get(bt.traffic_light, "n/a")
        last = int(bt.series["Exception"].iloc[-250:].sum())
        m[4].metric("Basel traffic light", light, f"{last} in last 250 days" if bt.traffic_light else "99% only",
                    delta_color="off", delta_arrow="off")
        st.plotly_chart(charts.backtest(bt.series, f"Daily return vs {conf_txt} VaR forecast"), **PLOT)

        rows = []
        for mth in ["historical", "parametric", "cornish-fisher"]:
            b = risk.backtest_var(port_r, mth, confidence, window)
            rows.append({"Model": mth.title(), "Exceptions": b.exceptions, "Expected": round(b.expected, 1),
                         "Exception rate": b.exception_rate, "Kupiec p": b.kupiec_p,
                         "Christoffersen p": b.christoffersen_p, "Traffic light": (b.traffic_light or "n/a").title()})
        st.dataframe(fmt_table(pd.DataFrame(rows), ["Exception rate"], ["Kupiec p", "Christoffersen p"]),
                     hide_index=True, width="stretch")
        with st.expander("How to read this"):
            st.markdown(
                f"- Each day's VaR is estimated from only the previous **{window}** days, then compared with what "
                f"actually happened. A loss bigger than the forecast is an **exception**.\n"
                f"- At {conf_txt} confidence about {1 - confidence:.1%} of days should be exceptions.\n"
                "- **Kupiec (proportion of failures):** is the number of exceptions consistent with the confidence "
                "level? p < 0.05 means the model under- or over-states risk.\n"
                "- **Christoffersen (independence):** do exceptions cluster? Clustering means the model reacts too "
                "slowly when volatility jumps.\n"
                "- **Basel traffic light** (99% VaR, last 250 days): 0–4 exceptions green, 5–9 yellow, 10+ red. "
                "Banks in the yellow or red zone get a higher capital multiplier.")

# ---------- Rolling ----------
with tabs[3]:
    win = st.select_slider("Rolling window", [63, 126, 252], value=126,
                           format_func=lambda w: {63: "3 months", 126: "6 months", 252: "1 year"}[w])
    roll = metrics.rolling_metrics(port_r, bench_r, rf, win)
    st.plotly_chart(charts.rolling(roll, "Volatility", "Rolling annualised volatility", True), **PLOT)
    st.plotly_chart(charts.rolling(roll, "Sharpe", "Rolling Sharpe ratio", False, ref=0), **PLOT)
    st.plotly_chart(charts.rolling(roll, "Beta", f"Rolling beta to the {bench_name}", False, ref=1), **PLOT)

# ---------- Optimiser ----------
with tabs[4]:
    o1, o2 = st.columns(2)
    cap = o1.slider("Maximum weight per holding", 10, 100, 35, 5, format="%d%%",
                    help="Caps concentration; also stops the optimiser piling into the best past performer.") / 100
    shrink = o2.toggle("Ledoit-Wolf covariance shrinkage", value=True)
    if cap * len(target) < 1:
        st.warning(f"With {len(target)} holdings the cap must be at least {1 / len(target):.0%}.")
    else:
        cov = optimize.covariance(asset_r, shrink)
        mu = optimize.expected_returns(asset_r)
        strategies = {
            "Your portfolio": target,
            "Min variance": optimize.min_variance(cov, cap),
            "Max Sharpe": optimize.max_sharpe(mu, cov, rf, cap),
            "Risk parity": optimize.risk_parity(cov),
            "Equal weight": pd.Series(1 / len(target), index=target.index),
        }
        pts = {k: optimize.portfolio_stats(w, mu, cov, rf) for k, w in strategies.items()}
        single = pd.DataFrame({"Expected return": mu, "Volatility": np.sqrt(np.diag(cov))}, index=cov.index)
        single.index = [t.replace(".NS", "") for t in single.index]
        st.plotly_chart(charts.frontier(optimize.efficient_frontier(mu, cov, cap), pts, single), **PLOT)

        rows = []
        for k, w in strategies.items():
            r = metrics.portfolio_returns(asset_r, w, REBALANCE[rebalance_name])
            rows.append({"Strategy": k, "Expected return": pts[k]["Expected return"],
                         "Volatility": pts[k]["Volatility"], "Sharpe": pts[k]["Sharpe"],
                         "Max drawdown": metrics.drawdown(r).min(),
                         f"{conf_txt} 1-day VaR": risk.historical_var(r, confidence)[0]})
        st.dataframe(fmt_table(pd.DataFrame(rows), ["Expected return", "Volatility", "Max drawdown",
                                                   f"{conf_txt} 1-day VaR"], ["Sharpe"]),
                     hide_index=True, width="stretch")
        wdf = pd.DataFrame(strategies).rename(index=names)
        st.plotly_chart(charts.weights_bars(wdf), **PLOT)
        st.download_button("Download weights (CSV)", wdf.to_csv().encode(), "optimised_weights.csv", "text/csv")
        st.caption("In-sample: weights are fitted to the same history they are scored on, so the frontier flatters "
                   "every strategy. Expected returns from past means are the noisiest input, which is why weights "
                   "are capped and the covariance is shrunk. Risk parity and minimum variance do not use expected "
                   "returns at all.")

# ---------- Stress tests ----------
with tabs[5]:
    st.subheader("Historical scenarios")
    sc = stress.historical_scenarios(asset_r, port_r, bench_r)
    if sc.empty:
        st.info("None of the stress episodes fall inside the selected period. Choose 'Max'.")
    else:
        st.plotly_chart(charts.stress_bars(sc, bench_name), **PLOT)
        sc_show = sc.assign(**{"Window": sc["From"].map(lambda d: f"{d:%d %b %Y}") + " – "
                               + sc["To"].map(lambda d: f"{d:%d %b %Y}"),
                               "Worst holding": sc["Worst holding"].map(names),
                               "P&L (₹)": (sc["Portfolio"] * notional).map(inr)})
        cols = ["Scenario", "Window", "Portfolio", "Benchmark", "P&L (₹)", "Worst holding", "Worst holding return"]
        st.dataframe(fmt_table(sc_show[cols].rename(columns={"Benchmark": bench_name}),
                               ["Portfolio", bench_name, "Worst holding return"]),
                     hide_index=True, width="stretch")
        st.markdown("\n".join(f"- **{r.Scenario}:** {r.Note}" for r in sc.itertuples()))
    st.subheader(f"Hypothetical {bench_name} shock")
    shock = st.slider(f"{bench_name} move", -40, 20, -10, 1, format="%d%%") / 100
    table, betas = stress.beta_shock(asset_r, target, bench_r, shocks=(shock,))
    move = table["Expected portfolio move"].iloc[0]
    st.metric("Expected portfolio move", f"{move:.1%}", f"{inr(move * notional)} on {inr(notional)}",
              delta_color="off", delta_arrow="off")
    st.caption(f"Portfolio beta {float((target * betas).sum()):.2f}: the sum of each holding's weight × beta. "
               "This captures only market risk; stock-specific news and changing correlations in a crisis "
               "usually make real losses worse.")
    bt_df = pd.DataFrame({"Name": [names[t] for t in betas.index], "Weight": target.reindex(betas.index),
                          "Beta": betas, "Expected move": betas * shock})
    st.dataframe(fmt_table(bt_df, ["Weight", "Expected move"], ["Beta"]), hide_index=True,
                 width="stretch")

# ---------- Download ----------
with tabs[6]:
    st.subheader("Export")
    st.download_button("Excel risk report", build_report(prices, target, bench_r, rf, REBALANCE[rebalance_name],
                                                         confidence),
                       "nse_portfolio_risk_report.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
    st.caption("Workbook sheets: Summary, Holdings, VaR (four methods × two confidence levels), Risk contributions, "
               "VaR backtest, Monthly returns, Stress tests and Daily returns with drawdown.")
    st.download_button("Prices (CSV)", prices.to_csv().encode(), "prices.csv", "text/csv")
    st.download_button("Daily returns (CSV)", pd.concat([asset_r, port_r, bench_r.rename(bench_name)], axis=1)
                       .to_csv().encode(), "daily_returns.csv", "text/csv")
    if data.dropped:
        st.caption(f"Dropped for insufficient history: {', '.join(data.dropped)}")

st.divider()
st.caption("For education and analysis only, not investment advice. Prices are dividend- and split-adjusted closes.")

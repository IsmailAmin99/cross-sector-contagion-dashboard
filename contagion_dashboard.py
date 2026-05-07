"""
Cross-Sector Contagion Dashboard
Interactive Streamlit dashboard for the contagion prediction project.

Shows $1,000 investment returns vs SPX, Nasdaq, and other benchmarks,
event-driven sector stress heatmaps, and full model analytics.

Authors: Isma'il Amin & Danielson Azumah
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict
import streamlit as st
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
import datetime as dt
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------


class EventCat(Enum):
    # any new categories get added here, DO NOT TRY TO APPEND LATER ON; won't work

    GEOPOL = "geopolitical"
    RATE_DEC = "rate_decision"
    INFLATION = "inflation_surprise"
    SUPPLY_CHAIN = "supply_chain"
    COMMODITY = "energy_shock"
    RECESSION = "recession_scare"


@dataclass(frozen=True)
class MarketEvent:
    date: str
    cat: EventCat    # category
    sev: int    # severity
    name: str


@dataclass(frozen=True)
class Security:
    ticker: str
    name: str
    asset_class: str


EVENTS: List[MarketEvent] = [
    MarketEvent("2021-03-23", EventCat.SUPPLY_CHAIN, 3, "Suez Canal blockage"),
    MarketEvent("2021-06-16", EventCat.RATE_DEC,
                2, "Fed hawkish dot-plot surprise"),
    MarketEvent("2021-09-20", EventCat.RECESSION,
                2, "Evergrande default fears"),
    MarketEvent("2021-11-26", EventCat.SUPPLY_CHAIN,
                3, "Omicron Variant Discovery"),
    MarketEvent("2022-02-24", EventCat.GEOPOL,
                5, "Russia invades Ukraine"),
    MarketEvent("2022-06-15", EventCat.RATE_DEC,
                4, "75bps Fed Hike (Inflation Peak)"),
    MarketEvent("2022-08-26", EventCat.RATE_DEC, 3,
                "Powell's Jackson Hole 'Pain' Speech"),
    MarketEvent("2022-09-28", EventCat.GEOPOL,
                3, "Nord Stream pipeline leaks"),
    MarketEvent("2023-03-10", EventCat.RECESSION, 4, "SVB / Banking Crisis"),
    MarketEvent("2023-10-07", EventCat.GEOPOL,
                3, "Israel-Hamas Conflict Start"),
    MarketEvent("2024-04-10", EventCat.INFLATION, 3,
                "Hot CPI / Higher-for-Longer Pivot"),
    MarketEvent("2024-08-05", EventCat.RECESSION, 4,
                "Yen Carry Trade / Recession Panic"),
    MarketEvent("2025-01-20", EventCat.GEOPOL,
                5, "Liberation Day Tariffs")
]


SECTOR_ETFS = [
    Security("XLK", "Technology", "Equity"),
    Security("XLE", "Energy", "Equity"),
    Security("XLV", "Healthcare", "Equity"),
    Security("XLF", "Financials", "Equity"),
    Security("XLI", "Industrials", "Equiity"),
    Security("XLC", "Comm. Services", "Equity"),
    Security("XLY", "Consumer Disc.", "Equity"),
    Security("XLP", "Consumer Staples", "Equity"),
    Security("XLRE", "Real Estate", "Equity"),
    Security("XLU", "Utilities", "Equity"),
    Security("XLB", "Materials", "Equity")
]

BENCHMARKS = [
    Security("SPY", "S&P 500", "Equity"),
    Security("QQQ", "Nasdaq 100", "Equity"),
    Security("IWM", "Russell 2000", "Equity"),
    Security("AGG", "US Agg. Bond", "Equity"),
    Security("DIA", "Dow Jones", "Equity"),
    Security("EFA", "Intl Developed", "Equity")
]


EVENT_COLORS = {
    EventCat.GEOPOL: "#ef4444",
    EventCat.RATE_DEC: "#3b82f6",
    EventCat.INFLATION: "#f59e0b",
    EventCat.SUPPLY_CHAIN: "#8b5cf6",
    EventCat.COMMODITY: "#f97316",
    EventCat.RECESSION: "#6b7280",
}
"""
SECTOR_COLORS_OLD = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8",
]
"""
SECTOR_COLORS = {
    "XLK": "#1f77b4", "XLE": "#2ca02c", "XLV": "#ff7f0e",
    "XLF": "#d62728", "XLI": "#8c564b", "XLC": "#e377c2",
    "XLY": "#7f7f7f", "XLP": "#bcbd22", "XLRE": "#9467bd",
    "XLU": "#17becf", "XLB": "#aec7e8"
}


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Downloading market data...")
def load_all_data(start, end):
    all_tickers = [s.ticker for s in SECTOR_ETFS] + \
        [s.ticker for s in BENCHMARKS] + ["^VIX"]
    raw = yf.download(all_tickers, start=start, end=end,
                      progress=False, auto_adjust=True)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
        volumes = raw["Volume"].copy()
    else:
        prices = raw.copy()
        volumes = pd.DataFrame(index=raw.index)

    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.droplevel(0)

    prices = prices.ffill().dropna(how="all")
    volumes = volumes.ffill().dropna(how="all")
    return prices, volumes


def compute_strategy_returns(prices, events, sector_tickers, benchmark="SPY",
                             hedge_window=10, stress_threshold=1.0):
    """
    Simulate a contagion-aware strategy:
    - After each event, compute which sectors are likely stressed
      (based on event-study abnormal return logic).
    - Underweight stressed sectors, overweight safe sectors.
    - Compare vs equal-weight and buy-and-hold benchmarks.
    """
    returns = prices.pct_change().fillna(0)
    sector_list = [t for t in sector_tickers if t in returns.columns]
    n = len(sector_list)
    trading_dates = prices.index

    equal_weights = np.ones(n) / n
    active_weights = equal_weights.copy()

    strat_values = [1.0]
    equal_values = [1.0]

    spy_ret = returns[benchmark] if benchmark in returns.columns else returns[sector_list].mean(
        axis=1)

    cooldown_until = None

    for i in range(1, len(trading_dates)):
        date = trading_dates[i]
        prev_date = trading_dates[i - 1]

        for ev in events:
            ev_date = pd.Timestamp(ev.date)
            if prev_date <= ev_date <= date or (ev_date == date):
                stressed = []
                for j, sector in enumerate(sector_list):
                    if sector not in returns.columns:
                        continue
                    lookback = max(0, i - 20)
                    sect_ret = returns[sector].iloc[lookback:i]
                    spx_ret = spy_ret.iloc[lookback:i]
                    if len(sect_ret) < 5:
                        continue

                    cov = sect_ret.cov(spx_ret)
                    var = spx_ret.var()
                    beta = cov / var if var > 1e-8 else 1.0

                    recent_abnormal = sect_ret.iloc[-5:].mean() - \
                        beta * spx_ret.iloc[-5:].mean()
                    hist_std = (sect_ret - beta * spx_ret).std()

                    if hist_std > 1e-8 and recent_abnormal < -stress_threshold * hist_std:
                        stressed.append(j)

                severity_mult = ev.sev / 5.0

                if stressed:
                    new_w = equal_weights.copy()
                    cut = 0.5 * severity_mult
                    for idx in stressed:
                        new_w[idx] *= (1 - cut)
                    new_w /= new_w.sum()
                    active_weights = new_w
                    cooldown_until = date + \
                        pd.Timedelta(days=hedge_window * 1.5)
                break

        if cooldown_until is not None and date > cooldown_until:
            blend = 0.1
            active_weights = (1 - blend) * active_weights + \
                blend * equal_weights
            if np.allclose(active_weights, equal_weights, atol=0.005):
                active_weights = equal_weights.copy()
                cooldown_until = None

        day_ret = returns.loc[date, sector_list].values.copy().astype(float)
        np.nan_to_num(day_ret, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

        strat_ret = np.dot(active_weights, day_ret)
        equal_ret = np.dot(equal_weights, day_ret)

        strat_values.append(strat_values[-1] * (1 + strat_ret))
        equal_values.append(equal_values[-1] * (1 + equal_ret))

    dates = trading_dates[:len(strat_values)]
    return pd.DataFrame({
        "Contagion Strategy": strat_values,
        "Equal-Weight Sectors": equal_values,
    }, index=dates)


def compute_risk_metrics(series, rf_annual=0.04):
    """Compute Sharpe, max drawdown, CAGR, volatility for a price series."""
    rets = series.pct_change().dropna()
    n_years = len(rets) / 252
    if n_years < 0.01:
        return {}

    cagr = (series.iloc[-1] / series.iloc[0]) ** (1 / n_years) - 1
    vol = rets.std() * np.sqrt(252)
    sharpe = (cagr - rf_annual) / vol if vol > 0 else 0

    rolling_max = series.cummax()
    drawdown = (series - rolling_max) / rolling_max
    max_dd = drawdown.min()

    total_ret = series.iloc[-1] / series.iloc[0] - 1

    return {
        "Total Return": f"{total_ret:.1%}",
        "CAGR": f"{cagr:.1%}",
        "Volatility": f"{vol:.1%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_dd:.1%}",
    }


# ---------------------------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Cross-Sector Contagion Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stMetric {border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px;}
    div[data-testid="stMetricValue"] {font-size: 1.4rem;}
    .block-container {padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)
# -- Sidebar --
with st.sidebar:
    st.title("Settings")

    initial_investment = st.slider(
        "Initial Investment ($)", 100, 100_000, 1000, step=100,
        help="How much money to simulate investing at the start date."
    )

    date_range = st.date_input(
        "Date Range",
        value=(dt.date(2021, 1, 1), dt.date(2025, 4, 3)),
        min_value=dt.date(2020, 1, 1),
        max_value=dt.date(2025, 12, 31),
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = dt.date(2021, 1, 1), dt.date(2025, 4, 3)

    show_events = st.checkbox("Show event markers on charts", value=True)
    event_filter = st.multiselect(
        "Filter event categories",
        options=list(EVENT_COLORS.keys()),
        default=list(EVENT_COLORS.keys()),
        format_func=lambda x: x.name.replace("_", " ").title(),
    )

    selected_benchmarks = st.multiselect(
        "Benchmarks to compare",
        options=[s.ticker for s in BENCHMARKS],
        default=["SPY", "QQQ", "IWM"],
        format_func=lambda ticker: next(
            s.name for s in BENCHMARKS if s.ticker == ticker),
    )

    selected_sectors = st.sidebar.multiselect(
        "Select Sectors",
        options=[s.ticker for s in SECTOR_ETFS],
        default=[s.ticker for s in SECTOR_ETFS],
        format_func=lambda ticker: next(
            s.name for s in SECTOR_ETFS if s.ticker == ticker),
    )

    st.divider()
    st.caption("Isma'il Amin & Danielson Azumah")
    st.caption("CSCI-3412 Machine Learning")

# -- Load Data --
prices, volumes = load_all_data(str(start_date), str(end_date))

# =========================================================================
# HEADER
# =========================================================================
st.title("Cross-Sector Contagion Dashboard")
st.markdown(
    "**Predicting contagion effects of geopolitical & macro events for market alpha**")

# =========================================================================
# TAB LAYOUT
# =========================================================================
tab_perf, tab_sectors, tab_events, tab_risk, tab_model = st.tabs([
    "Investment Performance",
    "Sector Analysis",
    "Event Impact Explorer",
    "Risk Metrics",
    "Model Results",
])

# Filter events
filtered_events = [e for e in EVENTS if e.cat in event_filter
                   and pd.Timestamp(e.date) >= pd.Timestamp(start_date)
                   and pd.Timestamp(e.date) <= pd.Timestamp(end_date)]

# Pre-compute strategy returns (needed across multiple tabs)
strat_df = compute_strategy_returns(
    prices, filtered_events, [s.ticker for s in SECTOR_ETFS]
)
strat_norm = strat_df["Contagion Strategy"] * initial_investment
eq_norm = strat_df["Equal-Weight Sectors"] * initial_investment


def add_event_markers(fig, events_list, y_range=None):
    """Add vertical event lines and annotations to a plotly figure."""
    for ev in events_list:
        ev_date = pd.Timestamp(ev.date)
        color = EVENT_COLORS.get(ev.cat, "#94a3b8")
        fig.add_shape(
            type="line", x0=ev_date, x1=ev_date,
            y0=0, y1=1, yref="paper",
            line=dict(color=color, width=1, dash="dot"),
            opacity=0.5,
        )
        fig.add_annotation(
            x=ev_date, y=1.02, yref="paper",
            text=ev.name[:30], showarrow=False,
            font=dict(size=8, color=color),
            textangle=-45, xanchor="left",
        )


def calculate_safe_norm(series, initial_val):
    """ 
    prevents the dashboard from crashing/ showing blank charts if the data is messy
    """
    valid_prices = series.dropna()
    valid_prices = valid_prices[valid_prices > 0]

    if valid_prices.empty:
        return series * 0  # return 0's if no valid data exists

    base_price = valid_prices.iloc[0]
    return (series / base_price) * initial_val


# filter events to ensure they only appear if they exist in the trading data
valid_events = [e for e in filtered_events if e.date in prices.index]

# =========================================================================
# TAB 1: INVESTMENT PERFORMANCE
# =========================================================================

with tab_perf:
    st.subheader(f"${initial_investment:,} Invested - Growth Comparison")

    fig_perf = go.Figure()

    bench_colors = {
        "SPY": "#2563eb", "QQQ": "#7c3aed", "IWM": "#0891b2",
        "AGG": "#64748b", "DIA": "#d97706", "EFA": "#059669",
    }

    for bm in selected_benchmarks:
        if bm in prices.columns:
            norm = calculate_safe_norm(prices[bm], initial_investment)
            metrics = compute_risk_metrics(norm)
            total = metrics.get("Total Return", "N/A")
            benchmark_name = next(s.name for s in BENCHMARKS if s.ticker == bm)
            fig_perf.add_trace(go.Scatter(
                x=norm.index, y=norm.values,
                name=f"{benchmark_name} ({total})",
                line=dict(color=bench_colors.get(bm, "#94a3b8"), width=2),
                hovertemplate=f"{benchmark_name}<br>Date: %{{x}}<br>Value: $%{{y:,.0f}}<extra></extra>",
            ))

    strat_metrics = compute_risk_metrics(strat_norm)
    eq_metrics = compute_risk_metrics(eq_norm)

    fig_perf.add_trace(go.Scatter(
        x=strat_norm.index, y=strat_norm.values,
        name=f"Contagion Strategy ({strat_metrics.get('Total Return', 'N/A')})",
        line=dict(color="#dc2626", width=3),
        hovertemplate="Contagion Strategy<br>Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>",
    ))
    fig_perf.add_trace(go.Scatter(
        x=eq_norm.index, y=eq_norm.values,
        name=f"Equal-Weight Sectors ({eq_metrics.get('Total Return', 'N/A')})",
        line=dict(color="#f97316", width=2, dash="dash"),
        hovertemplate="Equal-Weight<br>Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>",
    ))

    if show_events:
        # implemented filter: only show events that fall within price date's timeline
        if valid_events:
            add_event_markers(fig_perf, valid_events)


# Chart Layout
    fig_perf.update_layout(
        title=dict(text="Cumulatve Strategy Performance vs Benchmarks", x=0.5),
        height=550,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1),
        yaxis_title="Portfolio Value ($)",
        xaxis_title="Date",
        yaxis_tickprefix="$",
        yaxis_tickformat=",",
        margin=dict(t=80, b=40),
    )
    fig_perf.add_hline(y=initial_investment, line_dash="dash", line_color="#94a3b8",
                       opacity=0.5)
    fig_perf.add_annotation(
        x=0.01, xref="paper", y=initial_investment,
        text=f"${initial_investment:,} invested", showarrow=False,
        font=dict(size=10, color="#94a3b8"), xanchor="left", yshift=10,
    )

    st.plotly_chart(fig_perf, width="stretch")

    # Final values summary
    st.subheader("Final Portfolio Values")
    cols = st.columns(min(len(selected_benchmarks) + 2, 6))
    col_idx = 0

    for bm in selected_benchmarks:
        if bm in prices.columns and col_idx < len(cols):
            final = prices[bm].iloc[-1] / \
                prices[bm].iloc[0] * initial_investment
            ret = (final / initial_investment - 1) * 100
            cols[col_idx].metric(
                next(s.name for s in BENCHMARKS if s.ticker == bm),
                f"${final:,.0f}",
                f"{ret:+.1f}%",
            )
            col_idx += 1

    if col_idx < len(cols):
        final_strat = strat_norm.iloc[-1]
        ret_strat = (final_strat / initial_investment - 1) * 100
        cols[col_idx].metric(
            "Contagion Strategy",
            f"${final_strat:,.0f}",
            f"{ret_strat:+.1f}%",
        )
        col_idx += 1

    if col_idx < len(cols):
        final_eq = eq_norm.iloc[-1]
        ret_eq = (final_eq / initial_investment - 1) * 100
        cols[col_idx].metric(
            "Equal-Weight",
            f"${final_eq:,.0f}",
            f"{ret_eq:+.1f}%",
        )

    # Drawdown chart
    st.subheader("Drawdown Analysis")
    fig_dd = go.Figure()

    for bm in selected_benchmarks:
        if bm in prices.columns:
            p = prices[bm]
            dd = (p - p.cummax()) / p.cummax() * 100
            fig_dd.add_trace(go.Scatter(
                x=dd.index, y=dd.values,
                name=next(s.name for s in BENCHMARKS if s.ticker == bm),
                line=dict(color=bench_colors.get(bm, "#94a3b8"), width=1.5),
                fill="tozeroy", opacity=0.3,
                hovertemplate=f"{benchmark_name}<br>Drawdown: %{{y:.1f}}%<extra></extra>",
            ))

    strat_dd = (strat_norm - strat_norm.cummax()) / strat_norm.cummax() * 100
    fig_dd.add_trace(go.Scatter(
        x=strat_dd.index, y=strat_dd.values,
        name="Contagion Strategy",
        line=dict(color="#dc2626", width=2),
        hovertemplate="Contagion Strategy<br>Drawdown: %{y:.1f}%<extra></extra>",
    ))

    if show_events:
        add_event_markers(fig_dd, filtered_events)

    fig_dd.update_layout(
        height=350, template="plotly_white",
        yaxis_title="Drawdown (%)", xaxis_title="Date",
        yaxis_ticksuffix="%",
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig_dd, width="stretch")


# =========================================================================
# TAB 2: SECTOR ANALYSIS
# =========================================================================
with tab_sectors:
    st.subheader("Sector Performance Comparison")

    col1, col2 = st.columns([3, 1])

    with col1:
        fig_sect = go.Figure()
        for i, ticker in enumerate(selected_sectors):
            sec_name = next(s.name for s in SECTOR_ETFS if s.ticker == ticker)
            if ticker in prices.columns:
                norm = prices[ticker] / \
                    prices[ticker].iloc[0] * initial_investment
                fig_sect.add_trace(go.Scatter(
                    x=norm.index, y=norm.values,
                    name=f"{ticker} ({sec_name})",
                    line=dict(color=SECTOR_COLORS.get(
                        ticker, "#94a3b"), width=1.8),
                    hovertemplate=f"Sector: {sec_name}<br>Price: %{{y:.2f}}<extra></extra>",
                ))

        if show_events:
            add_event_markers(fig_sect, filtered_events)

        fig_sect.update_layout(
            height=500, template="plotly_white",
            yaxis_title=f"Value of ${initial_investment:,}",
            yaxis_tickprefix="$", yaxis_tickformat=",",
            legend=dict(font=dict(size=10)),
            margin=dict(t=40),
        )
        fig_sect.add_hline(y=initial_investment, line_dash="dash",
                           line_color="#94a3b8", opacity=0.4)
        st.plotly_chart(fig_sect, width="stretch")

    with col2:
        st.markdown("**Total Returns**")
        ret_data = []
        for ticker in selected_sectors:
            if ticker in prices.columns:
                total = (prices[ticker].iloc[-1] /
                         prices[ticker].iloc[0] - 1) * 100
                ret_data.append(
                    {"Sector": f"{ticker}", "Name": sec_name, "Return": total})
        ret_df = pd.DataFrame(ret_data).sort_values("Return", ascending=False)
        for _, row in ret_df.iterrows():
            color = "#16a34a" if row["Return"] > 0 else "#dc2626"
            st.markdown(f"**{row['Sector']}** {row['Name']}<br>"
                        f"<span style='color:{color};font-size:1.1em'>{row['Return']:+.1f}%</span>",
                        unsafe_allow_html=True)

    # Correlation heatmap
    st.subheader("Rolling Sector Correlation Matrix")
    corr_window = st.slider("Correlation window (days)", 10, 120, 30)

    available_sectors = [t for t in selected_sectors if t in prices.columns]
    if available_sectors:
        sector_returns = prices[available_sectors].pct_change().dropna()
        corr = sector_returns.tail(corr_window).corr()

        fig_corr = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=[next((s.name for s in SECTOR_ETFS if s.ticker == t), t)
               for t in corr.columns],
            y=[next((s.name for s in SECTOR_ETFS if s.ticker == t), t)
               for t in corr.index],
            colorscale="RdBu_r", zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="%{x} vs %{y}<br>Corr: %{z:.3f}<extra></extra>",
        ))
        fig_corr.update_layout(
            height=500, template="plotly_white",
            title=f"Pairwise Correlation (last {corr_window} days)",
            margin=dict(t=60),
        )
        st.plotly_chart(fig_corr, width="stretch")


# =========================================================================
# TAB 3: EVENT IMPACT EXPLORER
# =========================================================================
with tab_events:
    st.subheader("Event Impact Analysis")

    if not filtered_events:
        st.warning(
            "No events match your current filters. Adjust the sidebar settings.")
    else:
        selected_event = st.selectbox(
            "Select an event to analyze:",
            options=range(len(filtered_events)),
            format_func=lambda i: f"{filtered_events[i].date} - {filtered_events[i].name}",
        )

        ev = filtered_events[selected_event]
        ev_date = pd.Timestamp(ev.date)
        trading_dates = prices.index

        idx = trading_dates.searchsorted(ev_date)
        if idx >= len(trading_dates):
            idx = len(trading_dates) - 1
        ev_actual = trading_dates[idx]

        pre_start = max(0, idx - 20)
        post_end = min(len(trading_dates) - 1, idx + 20)

        col_info, col_sev = st.columns([3, 1])
        with col_info:
            st.markdown(f"### {ev.name}")
            st.markdown(f"**Date:** {ev.date} | "
                        f"**Category:** {ev.cat.name.replace('_', ' ').title()} | "
                        f"**Severity:** {'*' * ev.sev} ({ev.sev}/5)")
        with col_sev:
            st.metric("Severity", f"{ev.sev}/5")

        st.subheader("Sector Returns Around Event")

        window_returns = {}
        for ticker in selected_sectors:
            if ticker not in prices.columns:
                continue
            window = prices[ticker].iloc[pre_start:post_end + 1]
            base_price = prices[ticker].iloc[idx]
            if base_price > 0:
                normalized = window / base_price * 100
                window_returns[ticker] = normalized

        fig_event = go.Figure()
        for i, (ticker, vals) in enumerate(window_returns.items()):
            fig_event.add_trace(go.Scatter(
                x=vals.index, y=vals.values - 100,
                name=f"{ticker} ({sec_name})",
                line=dict(
                    width=2, color=SECTOR_COLORS.get(ticker, "#94a3b8")),
                hovertemplate=f"{ticker}<br>Return: %{{y:+.2f}}%<extra></extra>",
            ))

        fig_event.add_shape(
            type="line", x0=ev_actual, x1=ev_actual,
            y0=0, y1=1, yref="paper",
            line=dict(color="#dc2626", width=2),
        )
        fig_event.add_annotation(
            x=ev_actual, y=1.05, yref="paper",
            text="EVENT", showarrow=False,
            font=dict(size=11, color="#dc2626"),
        )
        fig_event.add_hline(y=0, line_dash="dash", line_color="#94a3b8")

        fig_event.update_layout(
            height=450, template="plotly_white",
            yaxis_title="Return from Event Date (%)",
            yaxis_ticksuffix="%",
            legend=dict(font=dict(size=9)),
            margin=dict(t=40),
        )
        st.plotly_chart(fig_event, width="stretch")

        st.subheader("Post-Event Cumulative Returns (10 trading days)")
        post_returns = {}
        for ticker in selected_sectors:
            if ticker not in prices.columns:
                continue
            post_idx = min(idx + 10, len(prices) - 1)
            base_price = prices[ticker].iloc[idx]
            if base_price > 0:
                ret = (prices[ticker].iloc[post_idx] / base_price - 1) * 100
                post_returns[ticker] = ret

        if post_returns:
            sorted_sectors_ev = sorted(
                post_returns.items(), key=lambda x: x[1])
            tickers_sorted = [s[0] for s in sorted_sectors_ev]
            returns_sorted = [s[1] for s in sorted_sectors_ev]
            colors_bar = ["#dc2626" if r <
                          0 else "#16a34a" for r in returns_sorted]
            names_sorted = [
                f"{t} ({next(s.name for s in SECTOR_ETFS if s.ticker == t)})"
                for t in tickers_sorted]
            fig_bar = go.Figure(go.Bar(
                y=names_sorted, x=returns_sorted,
                orientation="h",
                marker_color=colors_bar,
                text=[f"{r:+.2f}%" for r in returns_sorted],
                textposition="auto",
                hovertemplate="%{y}<br>Return: %{x:+.2f}%<extra></extra>",
            ))
            fig_bar.update_layout(
                height=max(350, len(selected_sectors) * 35),
                template="plotly_white",
                xaxis_title="10-Day Post-Event Return (%)",
                xaxis_ticksuffix="%",
                margin=dict(t=20, l=180),
            )
            fig_bar.add_vline(x=0, line_color="#94a3b8")
            st.plotly_chart(fig_bar, width="stretch")

        vix_col = "^VIX" if "^VIX" in prices.columns else None
        if vix_col:
            st.subheader("VIX Around Event")
            vix_window = prices[vix_col].iloc[pre_start:post_end + 1]
            fig_vix = go.Figure()
            fig_vix.add_trace(go.Scatter(
                x=vix_window.index, y=vix_window.values,
                name="VIX", fill="tozeroy",
                line=dict(color="#f59e0b", width=2),
                fillcolor="rgba(245, 158, 11, 0.15)",
                hovertemplate="VIX: %{y:.1f}<extra></extra>",
            ))
            fig_vix.add_shape(
                type="line", x0=ev_actual, x1=ev_actual,
                y0=0, y1=1, yref="paper",
                line=dict(color="#dc2626", width=2),
            )
            fig_vix.add_annotation(
                x=ev_actual, y=1.05, yref="paper",
                text="EVENT", showarrow=False,
                font=dict(size=10, color="#dc2626"),
            )
            fig_vix.update_layout(
                height=250, template="plotly_white",
                yaxis_title="VIX Level", margin=dict(t=30, b=30),
            )
            st.plotly_chart(fig_vix, width="stretch")


# =========================================================================
# TAB 4: RISK METRICS
# =========================================================================
with tab_risk:
    st.subheader("Risk-Adjusted Performance Comparison")

    all_series = {}
    for bm in selected_benchmarks:
        if bm in prices.columns:
            all_series[benchmark_name] = prices[bm] / \
                prices[bm].iloc[0] * initial_investment

    all_series["Contagion Strategy"] = strat_norm
    all_series["Equal-Weight Sectors"] = eq_norm

    metrics_rows = []
    for name, series in all_series.items():
        m = compute_risk_metrics(series)
        m["Strategy"] = name
        metrics_rows.append(m)

    metrics_df = pd.DataFrame(metrics_rows).set_index("Strategy")
    st.dataframe(metrics_df, width="stretch")

    # Sharpe comparison bar chart
    st.subheader("Sharpe Ratio Comparison")
    sharpe_vals = {}
    for name, series in all_series.items():
        m = compute_risk_metrics(series)
        try:
            sharpe_vals[name] = float(m.get("Sharpe Ratio", 0))
        except (ValueError, TypeError):
            sharpe_vals[name] = 0

    fig_sharpe = go.Figure(go.Bar(
        x=list(sharpe_vals.keys()),
        y=list(sharpe_vals.values()),
        marker_color=["#dc2626" if k == "Contagion Strategy" else "#3b82f6"
                      for k in sharpe_vals.keys()],
        text=[f"{v:.2f}" for v in sharpe_vals.values()],
        textposition="auto",
    ))
    fig_sharpe.update_layout(
        height=350, template="plotly_white",
        yaxis_title="Sharpe Ratio",
        margin=dict(t=30),
    )
    st.plotly_chart(fig_sharpe, width="stretch")

    # Rolling volatility
    st.subheader("Rolling 30-Day Volatility")
    fig_vol = go.Figure()
    for name, series in all_series.items():
        rvol = series.pct_change().rolling(30).std() * np.sqrt(252) * 100
        fig_vol.add_trace(go.Scatter(
            x=rvol.index, y=rvol.values,
            name=name,
            line=dict(width=1.5),
            hovertemplate=f"{name}<br>Vol: %{{y:.1f}}%<extra></extra>",
        ))
    fig_vol.update_layout(
        height=350, template="plotly_white",
        yaxis_title="Annualized Vol (%)", yaxis_ticksuffix="%",
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1),
        margin=dict(t=40),
    )
    st.plotly_chart(fig_vol, width="stretch")

    # Monthly return heatmap
    st.subheader("Monthly Returns Heatmap - Contagion Strategy")
    monthly = strat_norm.resample("ME").last().pct_change().dropna() * 100
    monthly_df = pd.DataFrame({
        "Year": monthly.index.year,
        "Month": monthly.index.month,
        "Return": monthly.values,
    })
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = monthly_df.pivot_table(
        index="Year", columns="Month", values="Return", aggfunc="mean")
    pivot.columns = [month_names[int(c) - 1] for c in pivot.columns]

    fig_monthly = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="RdYlGn",
        text=np.round(pivot.values, 1),
        texttemplate="%{text}%",
        textfont={"size": 11},
        hovertemplate="Year %{y}, %{x}<br>Return: %{z:.1f}%<extra></extra>",
    ))
    fig_monthly.update_layout(
        height=300, template="plotly_white",
        yaxis=dict(dtick=1),
        margin=dict(t=30, b=30),
    )
    st.plotly_chart(fig_monthly, width="stretch")


# =========================================================================
# TAB 5: MODEL RESULTS
# =========================================================================
with tab_model:
    st.subheader("GAT Model Performance Summary")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("GAT F1 Score", "0.553", "+143% vs RF baseline")
    col_b.metric("GAT Recall", "65.6%", "+37.5pp vs RF")
    col_c.metric("Backtest Hit Rate", "4/6 Events", "67%")

    st.divider()

    st.subheader("Model Comparison")
    model_data = pd.DataFrame({
        "Model": ["GAT (Ours)", "Random Forest", "Logistic Regression"],
        "Accuracy": [0.485, 0.439, 0.485],
        "Precision": [0.477, 0.391, 0.417],
        "Recall": [0.656, 0.281, 0.156],
        "F1 Score": [0.553, 0.327, 0.227],
        "True Negative Rate": [0.324, 0.588, 0.794],
    })
    st.dataframe(model_data.set_index("Model"), width="stretch")

    fig_model = go.Figure()
    metrics_to_plot = ["Precision", "Recall", "F1 Score"]
    colors_model = ["#dc2626", "#3b82f6", "#16a34a"]
    for i, row in model_data.iterrows():
        fig_model.add_trace(go.Bar(
            x=metrics_to_plot,
            y=[row[m] for m in metrics_to_plot],
            name=row["Model"],
            marker_color=colors_model[i],
            text=[f"{row[m]:.3f}" for m in metrics_to_plot],
            textposition="auto",
        ))
    fig_model.update_layout(
        height=400, template="plotly_white",
        barmode="group", yaxis_title="Score",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=50),
    )
    st.plotly_chart(fig_model, width="stretch")

    st.divider()

    st.subheader("Per-Event Backtest Detail")
    backtest_data = pd.DataFrame([
        {"Event": "US election - Trump victory", "Date": "2024-11-05",
         "Predicted": "All 11 sectors", "Actual": "XLK, XLV, XLP, XLRE, XLB", "Hit": True},
        {"Event": "Fed cuts 25bp, hawkish guidance", "Date": "2024-12-18",
         "Predicted": "None", "Actual": "XLI, XLY, XLP, XLRE, XLB", "Hit": False},
        {"Event": "Trump inauguration", "Date": "2025-01-21",
         "Predicted": "All 11 sectors", "Actual": "XLK, XLE, XLF, XLY, XLRE, XLU", "Hit": True},
        {"Event": "25% tariffs Canada/Mexico", "Date": "2025-02-03",
         "Predicted": "All 11 sectors", "Actual": "XLY", "Hit": True},
        {"Event": "Tariffs take effect", "Date": "2025-03-04",
         "Predicted": "None", "Actual": "XLV, XLF, XLC, XLY, XLRE, XLU", "Hit": False},
        {"Event": "Liberation Day tariffs", "Date": "2025-04-02",
         "Predicted": "All 11 sectors", "Actual": "9 of 11 sectors", "Hit": True},
    ])
    backtest_data["Hit"] = backtest_data["Hit"].map(
        {True: "HIT", False: "MISS"})
    st.dataframe(backtest_data.set_index("Event"), width="stretch")

    st.divider()

    st.subheader("Event Timeline")
    fig_timeline = go.Figure()
    if filtered_events:
        ev_df = pd.DataFrame(filtered_events)
        ev_df["date"] = pd.to_datetime(ev_df["date"])
        for cat, color in EVENT_COLORS.items():
            cat_events = ev_df[ev_df["cat"] == cat]
            if cat_events.empty:
                continue
            fig_timeline.add_trace(go.Scatter(
                x=cat_events["date"],
                y=cat_events["sev"],
                mode="markers+text",
                name=cat.name.replace("_", " ").title(),
                marker=dict(size=cat_events["sev"] * 6, color=color, opacity=0.8,
                            line=dict(width=1, color="white")),
                text=cat_events["name"].str[:25],
                textposition="top center",
                textfont=dict(size=8),
                hovertemplate="%{text}<br>Date: %{x}<br>Severity: %{y}/5<extra></extra>",
            ))
    fig_timeline.update_layout(
        height=400, template="plotly_white",
        yaxis_title="Severity (1-5)", xaxis_title="Date",
        yaxis=dict(range=[0, 6]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=50),
    )
    st.plotly_chart(fig_timeline, width="stretch")

    # Architecture summary
    with st.expander("GAT Architecture Details"):
        st.markdown("""
**Graph Attention Network (GAT) - Event Conditioned**

| Component | Details |
|---|---|
| **Input** | 17 features per node (10 market + 7 event context) |
| **Layer 1** | GATConv (17 -> 48, 4 attention heads) + BatchNorm + ELU |
| **Layer 2** | GATConv (192 -> 48, 2 attention heads) + BatchNorm + ELU |
| **Classifier** | Linear(48->32) -> ReLU -> Linear(32->16) -> ReLU -> Linear(16->2) |
| **Loss** | Focal Loss (gamma=2.0, class-weighted) |
| **Optimizer** | AdamW (lr=0.003, wd=5e-4) |
| **Scheduler** | CosineAnnealingWarmRestarts (T_0=30) |
| **Nodes** | 11 sector ETFs per graph |
| **Edges** | Rolling 20-day correlation > 0.3 threshold |
| **Training** | Walk-forward split: 25 train / 6 val / 6 test graphs |
| **Epochs** | 150 with early stopping on best Val F1 |
""")

    with st.expander("How the Contagion Strategy Works"):
        st.markdown("""
**Strategy Logic:**

1. Start with equal weights across all 11 sector ETFs
2. When a macro/geopolitical event occurs:
   - Compute each sector's recent abnormal return vs its beta-adjusted expected return
   - Sectors with significantly negative abnormal returns are flagged as "stressed"
   - Reduce allocation to stressed sectors proportionally to event severity
   - Redistribute weight to non-stressed sectors
3. After the hedge window expires, gradually revert back to equal weights
4. Repeat for every event in the timeline

**This is a simplified proxy for what the GAT model would recommend** - the actual
GAT model learns more complex graph-based propagation patterns, but this
rule-based strategy demonstrates the core idea of rotating away from contagion.
""")


# =========================================================================
# FOOTER
# =========================================================================
st.divider()
st.caption("Cross-Sector Contagion Dashboard | Isma'il Amin & Danielson Azumah | CSCI-3412 Machine Learning")
st.caption(
    "Data: yfinance | Model: PyTorch Geometric GAT | Dashboard: Streamlit + Plotly")

import warnings
from datetime import date
from itertools import product

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Stock Price Forecast — Auto ARIMA", page_icon="📈", layout="wide")
st.title("📈 Stock Price Forecast — ARIMA / Auto ARIMA")
st.caption(
    "Explore historical prices and moving averages, compare ARIMA orders using AIC, BIC, or AICc, "
    "and generate a future forecast. Forecasts are estimates, not guarantees."
)


# -----------------------------------------------------------------------------
# Data and model helpers
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=1800)
def download_close(ticker: str, start_str: str, end_exclusive_str: str) -> pd.Series:
    """Download and normalize adjusted closing prices from Yahoo Finance."""
    raw = yf.download(
        ticker,
        start=start_str,
        end=end_exclusive_str,
        auto_adjust=True,
        progress=False,
        timeout=20,
        threads=False,
    )
    if raw is None or raw.empty:
        return pd.Series(dtype=float, name="Close")

    if isinstance(raw.columns, pd.MultiIndex):
        # yfinance column layout has changed between releases; find the Close level.
        close_col_level = None
        for level in range(raw.columns.nlevels):
            if "Close" in raw.columns.get_level_values(level):
                close_col_level = level
                break
        if close_col_level is not None:
            raw.columns = raw.columns.get_level_values(close_col_level)

    if "Close" not in raw.columns:
        raise ValueError("Yahoo Finance data did not contain a Close price column.")

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna().sort_index().astype(float)
    close = close[~close.index.duplicated(keep="last")]
    close.name = "Close"
    return close


def resolve_ticker(user_input: str) -> tuple[str, str]:
    """Resolve a ticker or company name to a Yahoo Finance symbol."""
    query = user_input.strip()
    if not query:
        raise ValueError("Enter a company name or ticker symbol.")

    # Explicit ticker syntax (e.g. TSLA, TCS.NS, ^NSEI, BTC-USD) should be
    # tried directly first. Company names containing spaces use Yahoo search.
    looks_like_symbol = " " not in query and all(ch.isalnum() or ch in ".^-_=" for ch in query)
    if looks_like_symbol:
        symbol = query.upper()
        try:
            info = yf.Ticker(symbol).get_info()
            if info and (info.get("shortName") or info.get("longName")):
                return symbol, str(info.get("shortName") or info.get("longName"))
        except Exception:
            pass
        # Some symbols may have price history even when get_info is unavailable.
        try:
            probe = yf.download(symbol, period="5d", auto_adjust=True, progress=False, timeout=10, threads=False)
            if probe is not None and not probe.empty:
                return symbol, symbol
        except Exception:
            pass

    try:
        results = yf.Search(query, max_results=10, news_count=0).quotes
    except Exception as exc:
        raise ValueError(f"Could not search Yahoo Finance for '{query}'. Check your internet connection or enter a ticker symbol.") from exc

    if not results:
        raise ValueError(f"No matching company or ticker found for '{query}'. Try a more specific company name or ticker symbol.")

    # Prefer equity/index/crypto symbols and an exact name match when available.
    q = query.casefold()
    def score(item):
        name = str(item.get("shortname") or item.get("longname") or item.get("name") or "")
        symbol = str(item.get("symbol") or "")
        quote_type = str(item.get("quoteType") or item.get("typeDisp") or "").casefold()
        value = 0
        if name.casefold() == q:
            value += 100
        elif q in name.casefold() or name.casefold() in q:
            value += 40
        if quote_type in {"equity", "index", "etf", "cryptocurrency"}:
            value += 10
        if symbol.endswith((".NS", ".BO")) and any(x in q for x in ("india", "indian", "nse", "bse", "reliance", "tcs", "infosys")):
            value += 5
        return value

    best = max(results, key=score)
    symbol = str(best.get("symbol") or "").strip().upper()
    name = str(best.get("shortname") or best.get("longname") or best.get("name") or symbol)
    if not symbol:
        raise ValueError(f"Yahoo Finance returned no ticker for '{query}'.")
    return symbol, name


def adf_summary(values) -> dict:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) < 20:
        return {
            "ADF Statistic": "N/A",
            "p-value": "N/A",
            "Result": "Too few observations (20+ recommended).",
        }
    try:
        stat, p_value, *_ = adfuller(clean, autolag="AIC")
        return {
            "ADF Statistic": round(float(stat), 4),
            "p-value": round(float(p_value), 6),
            "Result": "Stationary at 5% level" if p_value < 0.05 else "Not stationary at 5% level",
        }
    except Exception as exc:
        return {"ADF Statistic": "N/A", "p-value": "N/A", "Result": f"ADF test failed: {exc}"}


def choose_d_by_adf(values, max_d: int = 2) -> tuple[int, list[dict]]:
    """Choose the smallest differencing order whose ADF p-value is below 0.05.

    Tests the original series, then first and second differences if needed.
    Returns d and diagnostic rows for display.
    """
    series = pd.Series(np.asarray(values, dtype=float)).replace([np.inf, -np.inf], np.nan).dropna()
    diagnostics = []
    for d in range(max_d + 1):
        tested = series if d == 0 else series.diff(d).dropna()
        result = adf_summary(tested.to_numpy())
        p_value = result.get("p-value")
        diagnostics.append({"d": d, **result})
        if isinstance(p_value, (float, int)) and p_value < 0.05:
            return d, diagnostics
    # If none passed, use the maximum tested differencing order and expose diagnostics.
    return max_d, diagnostics


def fit_arima(values, order):
    """Fit ARIMA on an explicit RangeIndex to avoid unsupported date-index errors."""
    y = np.asarray(values, dtype=float).reshape(-1)
    y = y[np.isfinite(y)]
    if len(y) < 10:
        raise ValueError("At least 10 clean observations are needed to fit ARIMA.")

    # A RangeIndex is deliberately used for the model's internal index. We map all
    # predictions back to real dates ourselves, so no inferred date frequency is needed.
    indexed_y = pd.Series(y, index=pd.RangeIndex(start=0, stop=len(y)), name="Close")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(
            indexed_y,
            order=tuple(int(x) for x in order),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(method_kwargs={"maxiter": 50, "disp": 0})
    return fitted


def calc_aicc(aic: float, nobs: int, parameter_count: int) -> float:
    """Small-sample corrected AIC: AICc = AIC + 2k(k+1)/(n-k-1)."""
    denominator = nobs - parameter_count - 1
    if denominator <= 0:
        return float("nan")
    return float(aic + (2 * parameter_count * (parameter_count + 1)) / denominator)


@st.cache_data(show_spinner=False, ttl=3600)
def auto_arima_search(values: np.ndarray, max_pq: int, d_value: int, criterion: str):
    """Search p and q for the ADF-guided differencing order d_value."""
    y = np.asarray(values, dtype=float).reshape(-1)
    y = y[np.isfinite(y)]
    rows = []

    for p, d, q in product(range(max_pq + 1), [int(d_value)], range(max_pq + 1)):
        # Avoid obviously excessive parameter counts for shorter training histories.
        if len(y) <= p + q + d + 10:
            continue
        try:
            fitted = fit_arima(y, (p, d, q))
            aic = float(fitted.aic)
            bic = float(fitted.bic)
            k = len(fitted.params)
            aicc = calc_aicc(aic, int(fitted.nobs), int(k))
            score = {"AIC": aic, "BIC": bic, "AICc": aicc}[criterion]
            if not np.isfinite(score):
                continue
            retvals = getattr(fitted, "mle_retvals", {})
            converged = bool(retvals.get("converged", True)) if isinstance(retvals, dict) else True
            rows.append({
                "p": p, "d": d, "q": q,
                "AIC": aic, "BIC": bic, "AICc": aicc,
                "Parameters": int(k), "Converged": converged,
            })
        except Exception:
            # One unstable candidate should not stop the remaining grid search.
            continue

    if not rows:
        raise RuntimeError(
            "No candidate model could be fitted for this criterion. Try a longer history or lower p/q limits."
        )

    candidates = pd.DataFrame(rows)
    # Prefer converged models if at least one converged model is available.
    if candidates["Converged"].any():
        candidates = candidates[candidates["Converged"]].copy()
    candidates = candidates.sort_values(criterion, ascending=True).reset_index(drop=True)
    best = candidates.iloc[0]
    best_order = (int(best["p"]), int(best["d"]), int(best["q"]))
    best_score = float(best[criterion])
    return best_order, best_score, candidates


@st.cache_data(show_spinner=False, ttl=3600)
def rank_orders_by_inner_rmse(core_values: np.ndarray, inner_values: np.ndarray, candidate_orders: tuple):
    """Rank a small information-criterion shortlist on an inner validation window.

    The final outer holdout remains untouched until after order selection, which avoids
    selecting a model directly on the observations used for the displayed test RMSE.
    """
    core = np.asarray(core_values, dtype=float).reshape(-1)
    inner = np.asarray(inner_values, dtype=float).reshape(-1)
    rows = []
    for order in candidate_orders:
        try:
            fitted = fit_arima(core, tuple(order))
            predicted = np.asarray(fitted.forecast(steps=len(inner)), dtype=float).reshape(-1)
            errors = inner - predicted
            rows.append({
                "Order": tuple(order),
                "Inner RMSE": float(np.sqrt(np.mean(errors ** 2))),
                "Inner MAE": float(np.mean(np.abs(errors))),
            })
        except Exception:
            continue
    if not rows:
        raise RuntimeError("All shortlisted models failed during inner validation.")
    return pd.DataFrame(rows).sort_values("Inner RMSE").reset_index(drop=True)


def automatic_ma_windows(n_observations: int):
    """Choose practical short/long MA windows based on amount of history available."""
    short_window = 5 if n_observations < 100 else 7
    long_window = min(50, max(short_window + 5, round(n_observations * 0.20)))
    long_window = min(long_window, max(short_window + 1, n_observations - 1))
    if long_window <= short_window:
        long_window = short_window + 1
    return int(short_window), int(long_window)


def money(value, currency_symbol):
    return f"{currency_symbol}{value:,.2f}"


def make_price_chart(frame, ticker, short_window, long_window, tick_style="Years only"):
    """Interactive historical chart; hover displays exact date, price, and MA values."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frame.index, y=frame["Close"], mode="lines", name="Closing price",
        line={"width": 1.8},
        hovertemplate="Date: %{x|%d %b %Y}<br>Close: %{y:,.2f}<extra></extra>",
    ))
    short_col = f"MA_{short_window}"
    long_col = f"MA_{long_window}"
    fig.add_trace(go.Scatter(
        x=frame.index, y=frame[short_col], mode="lines", name=f"{short_window}-day MA",
        line={"width": 1.5, "dash": "dash"},
        hovertemplate=f"Date: %{{x|%d %b %Y}}<br>{short_window}-day MA: %{{y:,.2f}}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=frame.index, y=frame[long_col], mode="lines", name=f"{long_window}-day MA",
        line={"width": 1.5, "dash": "dot"},
        hovertemplate=f"Date: %{{x|%d %b %Y}}<br>{long_window}-day MA: %{{y:,.2f}}<extra></extra>",
    ))
    tickformat = "%d %b %Y" if tick_style == "Calendar dates" else "%Y"
    fig.update_layout(
        title=f"{ticker} — Historical Price and Moving Averages",
        xaxis_title="Date", yaxis_title="Adjusted closing price",
        hovermode="x unified", template="plotly_white", height=520,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 20, "r": 20, "t": 80, "b": 20},
    )
    fig.update_xaxes(tickformat=tickformat, rangeslider_visible=True)
    fig.update_yaxes(tickformat=",.2f")
    return fig


def make_forecast_chart(close, forecast_dates, forecast_values, ci_lower, ci_upper,
                        short_window, long_window, ticker, order, tick_style="Calendar dates"):
    lookback = min(180, len(close))
    recent = close.iloc[-lookback:]
    plot_df = pd.DataFrame({"Close": recent})
    plot_df[f"MA_{short_window}"] = recent.rolling(short_window).mean()
    plot_df[f"MA_{long_window}"] = recent.rolling(long_window).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df["Close"], mode="lines", name="Historical close",
        line={"width": 1.8},
        hovertemplate="Date: %{x|%d %b %Y}<br>Close: %{y:,.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df[f"MA_{short_window}"], mode="lines", name=f"{short_window}-day MA",
        line={"dash": "dash", "width": 1.5},
        hovertemplate=f"Date: %{{x|%d %b %Y}}<br>{short_window}-day MA: %{{y:,.2f}}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df[f"MA_{long_window}"], mode="lines", name=f"{long_window}-day MA",
        line={"dash": "dot", "width": 1.5},
        hovertemplate=f"Date: %{{x|%d %b %Y}}<br>{long_window}-day MA: %{{y:,.2f}}<extra></extra>",
    ))
    # Anchor the forecast at the latest close so the transition is visually clear.
    last_date = pd.Timestamp(close.index[-1])
    if last_date.tzinfo is not None:
        last_date = last_date.tz_localize(None)
    forecast_x = [last_date] + list(forecast_dates)
    forecast_y = [float(close.iloc[-1])] + list(forecast_values)
    lower_y = [float(close.iloc[-1])] + list(ci_lower)
    upper_y = [float(close.iloc[-1])] + list(ci_upper)
    # Shaded 95% interval follows the forecast anchor and then its future interval.
    fig.add_trace(go.Scatter(
        x=forecast_x + forecast_x[::-1],
        y=upper_y + lower_y[::-1],
        fill="toself", fillcolor="rgba(220, 50, 50, 0.14)",
        line={"color": "rgba(255,255,255,0)"}, name="95% confidence interval",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=forecast_x, y=forecast_y, mode="lines+markers", name=f"Forecast ARIMA{order}",
        line={"dash": "dash", "width": 2.7}, marker={"size": 7},
        hovertemplate="Date: %{x|%d %b %Y}<br>Forecast / last close: %{y:,.2f}<extra></extra>",
    ))
    tickformat = "%d %b %Y" if tick_style == "Calendar dates" else "%Y"
    fig.update_layout(
        title=f"{ticker} — Forecast using ARIMA{order}",
        xaxis_title="Date", yaxis_title="Price", hovermode="x unified",
        template="plotly_white", height=540,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 20, "r": 20, "t": 80, "b": 20},
    )
    fig.update_xaxes(tickformat=tickformat, rangeslider_visible=True)
    fig.update_yaxes(tickformat=",.2f")
    return fig


# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Configuration")
ticker_input = st.sidebar.text_input(
    "Company name or ticker", value="TCS.NS",
    help="Enter a company name (e.g. Tesla, Reliance Industries) or ticker (e.g. TSLA, RELIANCE.NS, ^NSEI). The app will look up the ticker automatically.",
).strip()

history_style = st.sidebar.radio(
    "Historical range input",
    ["Calendar dates", "By year"], horizontal=True,
    help="Choose exact dates using calendar pickers, or choose start/end years.",
)
today = date.today()
current_year = today.year
if history_style == "Calendar dates":
    date_col1, date_col2 = st.sidebar.columns(2)
    with date_col1:
        start_date = st.date_input("Start date", value=date(2021, 1, 1), max_value=today)
    with date_col2:
        end_date = st.date_input("End date", value=today, max_value=today)
else:
    years = list(range(1990, current_year + 1))
    y1, y2 = st.sidebar.columns(2)
    with y1:
        start_year = st.selectbox("Start year", years, index=years.index(min(2021, current_year)))
    with y2:
        end_year = st.selectbox("End year", years, index=len(years) - 1)
    start_date = date(int(start_year), 1, 1)
    end_date = today if int(end_year) == current_year else date(int(end_year), 12, 31)
    st.sidebar.caption(f"Selected period: {start_date:%d %b %Y} – {end_date:%d %b %Y}")

forecast_days = st.sidebar.slider("Forecast horizon (trading days)", 3, 30, 7, 1)
st.sidebar.caption("7 means the next 7 trading days, not 7 calendar days.")

st.sidebar.subheader("Auto ARIMA")
selection_criterion = st.sidebar.selectbox(
    "Model selection criterion", ["AIC", "BIC", "AICc"], index=2,
    help="AIC balances fit and complexity; BIC penalizes complexity more; AICc adds a small-sample correction.",
)
max_pq = st.sidebar.slider("Maximum p and q", min_value=0, max_value=5, value=2, step=1,
                            help="Higher values search more candidate models and take longer.")

st.sidebar.subheader("Moving averages")
auto_ma = st.sidebar.checkbox("Automatically choose MA windows", value=True)
ma_short_input = st.sidebar.number_input("Short-term MA window", min_value=3, max_value=30, value=7,
                                          step=1, disabled=auto_ma)
ma_long_input = st.sidebar.number_input("Long-term MA window", min_value=10, max_value=200, value=50,
                                         step=1, disabled=auto_ma)
chart_tick_style = st.sidebar.radio(
    "Chart date labels", ["Calendar dates", "Years only"], index=1, horizontal=True,
    help="Hover tooltips always show the exact calendar date and values.",
)
rolling_validation = st.sidebar.checkbox(
    "Use rolling validation (slower)", value=False,
    help="Off = one fit and multi-step holdout forecast (fast). On = refit before every test-day prediction (slower).",
)
run_button = st.sidebar.button("🚀 Run Forecast", type="primary", use_container_width=True)

# Store inputs on submit so the forecast can be rerun consistently after a Streamlit rerun.
if run_button:
    if not ticker_input:
        st.error("Enter a stock ticker symbol.")
        st.stop()
    if start_date >= end_date:
        st.error("Start date must be earlier than end date.")
        st.stop()
    if not auto_ma and int(ma_short_input) >= int(ma_long_input):
        st.error("The short-term MA window must be smaller than the long-term MA window.")
        st.stop()
    try:
        resolved_ticker, resolved_name = resolve_ticker(ticker_input)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    st.sidebar.success(f"Resolved: {resolved_name} ({resolved_ticker})")
    st.session_state["forecast_config"] = {
        "ticker": resolved_ticker,
        "company_name": resolved_name,
        "start_date": start_date,
        "end_date": end_date,
        "forecast_days": int(forecast_days),
        "criterion": selection_criterion,
        "max_pq": int(max_pq),
        "auto_ma": bool(auto_ma),
        "ma_short": int(ma_short_input),
        "ma_long": int(ma_long_input),
        "chart_tick_style": chart_tick_style,
        "rolling_validation": bool(rolling_validation),
    }

cfg = st.session_state.get("forecast_config")
if cfg is None:
    st.info("👈 Choose the ticker and historical period in the sidebar, then click **Run Forecast**.")
    st.markdown(
        """
### Getting started
- **Calendar dates** lets you select an exact start/end date from a calendar.
- **By year** lets you select start and end years for longer historical ranges.
- **Auto ARIMA** selects the fitted model with the lowest value of the AIC, BIC, or AICc criterion chosen in the sidebar. Inner-validation RMSE is diagnostic only.
- The final holdout is kept separate so the reported RMSE is not directly used to choose the model.
- The app compares ARIMA with a last-price baseline because no ARIMA method can guarantee a lower future RMSE.
- **Auto moving averages** selects short- and long-term windows from the amount of historical data downloaded.
- Hover over charts to inspect exact dates, prices, predictions, baselines, and moving-average values.
        """
    )
    st.stop()


# -----------------------------------------------------------------------------
# Download data and run the forecast using submitted settings
# -----------------------------------------------------------------------------
ticker = cfg["ticker"]
horizon = cfg["forecast_days"]
currency_symbol = "₹" if ticker.endswith((".NS", ".BO")) else ("$" if not ticker.startswith("^") else "")

with st.spinner(f"Downloading historical prices for {ticker}..."):
    try:
        end_exclusive = (pd.Timestamp(cfg["end_date"]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        close = download_close(
            ticker,
            pd.Timestamp(cfg["start_date"]).strftime("%Y-%m-%d"),
            end_exclusive,
        )
    except Exception as exc:
        st.error(f"Could not download or prepare data for {ticker}: {exc}")
        st.stop()

if close.empty:
    st.error(f"No historical prices were returned for **{ticker}**. Check the ticker and date range.")
    st.stop()
if len(close) < horizon + 30:
    st.error(
        f"Only {len(close)} observations were downloaded. For a {horizon}-day holdout, use a longer date range; "
        f"at least {horizon + 30} observations are recommended."
    )
    st.stop()

if cfg["auto_ma"]:
    ma_short, ma_long = automatic_ma_windows(len(close))
else:
    ma_short, ma_long = cfg["ma_short"], cfg["ma_long"]

st.success(f"Loaded **{len(close):,} trading-day observations** for **{ticker}**.")
metric_cols = st.columns(4)
metric_cols[0].metric("First observation", close.index[0].strftime("%d %b %Y"))
metric_cols[1].metric("Latest observation", close.index[-1].strftime("%d %b %Y"))
metric_cols[2].metric("Short-term MA", f"{ma_short} trading days" + (" · Auto" if cfg["auto_ma"] else " · Manual"))
metric_cols[3].metric("Long-term MA", f"{ma_long} trading days" + (" · Auto" if cfg["auto_ma"] else " · Manual"))

# Historical graph, with both calendar and year-based controls.
chart_df = pd.DataFrame({"Close": close})
chart_df[f"MA_{ma_short}"] = close.rolling(ma_short).mean()
chart_df[f"MA_{ma_long}"] = close.rolling(ma_long).mean()
st.subheader("📊 Historical Trend and Moving Averages")
st.caption("Move your cursor over the graph to inspect the exact date, close price, and both MA values. Use the bottom range slider to zoom into a time period.")
st.plotly_chart(
    make_price_chart(chart_df, ticker, ma_short, ma_long, cfg["chart_tick_style"]),
    use_container_width=True,
    config={"displaylogo": False, "scrollZoom": True},
)

d_value, adf_diagnostics = choose_d_by_adf(close.to_numpy(), max_d=2)
with st.expander("🔍 Stationarity check (ADF test)", expanded=False):
    st.markdown("The app chooses **d automatically using the ADF test**: it uses the smallest differencing order with p-value below 0.05, testing up to d = 2.")
    st.dataframe(pd.DataFrame(adf_diagnostics), hide_index=True, use_container_width=True)
    st.info(f"ADF-guided differencing order: d = {d_value}")
    if not any(isinstance(row.get("p-value"), (float, int)) and row["p-value"] < 0.05 for row in adf_diagnostics):
        st.warning("The series did not pass the ADF test at 5% through d = 2. The app uses d = 2; interpret results cautiously.")

# Separate the final outer holdout from model selection. The final horizon is not used
# to choose the ARIMA order, so the reported holdout score remains a fairer estimate.
train = close.iloc[:-horizon]
test = close.iloc[-horizon:]
train_values = train.to_numpy(dtype=float)
test_values = test.to_numpy(dtype=float)

st.subheader("🧠 Auto ARIMA Model Selection")
# Use a small inner validation window at the end of the training split. Rank the
# information-criterion candidates on the earlier core, then refine the shortlist by
# inner-validation RMSE. This commonly avoids AIC-only models with poor forecast paths.
inner_horizon = min(max(5, min(horizon, 10)), max(3, len(train_values) // 5))
if len(train_values) - inner_horizon < 20:
    inner_horizon = max(3, min(5, len(train_values) // 6))
core_values = train_values[:-inner_horizon]
inner_values = train_values[-inner_horizon:]

with st.spinner(
    f"Searching ARIMA candidates and selecting the lowest {cfg['criterion']}…"
):
    try:
        validation_order, _, ic_leaderboard = auto_arima_search(
            train_values, cfg["max_pq"], d_value, cfg["criterion"]
        )
        # Re-derive the score from the criterion-sorted leaderboard so the exported
        # table always reports the exact metric used to select the winning model.
        ic_leaderboard = ic_leaderboard.sort_values(
            cfg["criterion"], ascending=True
        ).reset_index(drop=True)
        winner = ic_leaderboard.iloc[0]
        validation_order = (int(winner["p"]), int(winner["d"]), int(winner["q"]))
        selected_score = float(winner[cfg["criterion"]])
        # Keep a small shortlist for transparency only. The final selected model is
        # strictly the lowest score under the criterion chosen in the sidebar.
        shortlist = [
            (int(row.p), int(row.d), int(row.q))
            for row in ic_leaderboard.head(7).itertuples(index=False)
        ]
        shortlist = tuple(dict.fromkeys(shortlist))
        # Inner validation scores are diagnostic; they do NOT override AIC/BIC/AICc.
        inner_horizon = min(max(3, min(horizon, 7)), max(3, len(train_values) // 5))
        core_values = train_values[:-inner_horizon]
        inner_values = train_values[-inner_horizon:]
        try:
            inner_scores = rank_orders_by_inner_rmse(core_values, inner_values, shortlist)
        except Exception:
            inner_scores = pd.DataFrame(columns=["Order", "Inner RMSE", "Inner MAE"])
    except Exception as exc:
        st.error(f"Auto ARIMA selection failed: {exc}")
        st.stop()

st.info(
    f"Selected **ARIMA{validation_order}** because it has the lowest **{cfg['criterion']}** "
    f"({selected_score:,.3f}) among the fitted candidate models."
)
st.caption(
    f"The sidebar criterion controls model selection directly. Inner-validation RMSE is shown only "
    "as a diagnostic and does not replace your chosen AIC/BIC/AICc criterion. The final holdout "
    "is kept separate for evaluation."
)

with st.expander("View candidate models and scores", expanded=False):
    display_scores = ic_leaderboard.head(10).copy()
    display_scores["ARIMA order"] = display_scores.apply(
        lambda row: f"({int(row['p'])}, {int(row['d'])}, {int(row['q'])})", axis=1
    )
    if not inner_scores.empty:
        inner_display = inner_scores.copy()
        inner_display["ARIMA order"] = inner_display["Order"].map(str)
        inner_display = inner_display.drop(columns=["Order"])
        display_scores = display_scores.merge(inner_display, on="ARIMA order", how="left")
    display_cols = ["ARIMA order", "AIC", "BIC", "AICc", "Converged"]
    if "Inner RMSE" in display_scores.columns:
        display_cols += ["Inner RMSE", "Inner MAE"]
    st.dataframe(display_scores[display_cols].round(3), use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# Holdout validation: fast by default, optional refit-every-step rolling mode
# -----------------------------------------------------------------------------
if cfg["rolling_validation"]:
    st.subheader(f"🔄 {horizon}-Trading-Day Rolling Validation")
else:
    st.subheader(f"🔄 {horizon}-Trading-Day Holdout Validation")
progress = st.progress(0, text="Fitting validation model…")
predictions = []
try:
    if cfg["rolling_validation"]:
        history = train_values.tolist()
        for i, actual in enumerate(test_values, start=1):
            fitted = fit_arima(history, validation_order)
            next_value = np.asarray(fitted.forecast(steps=1), dtype=float).reshape(-1)[0]
            predictions.append(float(next_value))
            history.append(float(actual))
            progress.progress(i / len(test_values), text=f"Rolling validation: step {i} of {len(test_values)}")
    else:
        fitted_validation = fit_arima(train_values, validation_order)
        predictions = np.asarray(fitted_validation.forecast(steps=horizon), dtype=float).reshape(-1).tolist()
        progress.progress(1.0, text="Fast holdout forecast complete")
except Exception as exc:
    progress.empty()
    st.error(f"Validation failed for ARIMA{validation_order}: {exc}")
    st.stop()
progress.empty()

predicted_values = np.asarray(predictions, dtype=float)
errors = test_values - predicted_values
mae = float(np.mean(np.abs(errors)))
rmse = float(np.sqrt(np.mean(errors ** 2)))
nonzero = np.abs(test_values) > 1e-12
mape = float(np.mean(np.abs(errors[nonzero] / test_values[nonzero])) * 100) if nonzero.any() else np.nan

# Compare with a simple last-price baseline. In fast mode this repeats the final
# training close for the whole holdout; in rolling mode it carries the last observed
# actual forward one step at a time.
if cfg["rolling_validation"]:
    baseline_values = np.concatenate(([train_values[-1]], test_values[:-1]))
else:
    baseline_values = np.repeat(train_values[-1], horizon)
baseline_rmse = float(np.sqrt(np.mean((test_values - baseline_values) ** 2)))

m1, m2, m3, m4 = st.columns(4)
m1.metric("MAE", money(mae, currency_symbol))
m2.metric("Model RMSE", money(rmse, currency_symbol))
m3.metric("Last-price baseline RMSE", money(baseline_rmse, currency_symbol),
          delta=f"{rmse - baseline_rmse:+.2f} vs baseline", delta_color="inverse")
m4.metric("MAPE", f"{mape:.2f}%" if np.isfinite(mape) else "N/A")
if rmse > baseline_rmse:
    st.warning(
        "This ARIMA model scored worse than the simple last-price baseline on this holdout. "
        "The selected ARIMA order will not necessarily improve every period; consider using "
        "the baseline when it performs better."
    )
if cfg["rolling_validation"]:
    st.caption("Rolling mode refits before each prediction and uses the latest actual price at each step, so it is slower than fast mode.")
else:
    st.caption("Fast mode fits once and forecasts the entire holdout without using test-period actuals. Stock-price jumps cannot be predicted reliably from past prices alone.")

validation_df = pd.DataFrame({
    "Date": test.index.strftime("%d %b %Y"),
    "Actual price": np.round(test_values, 2),
    "Predicted price": np.round(predicted_values, 2),
    "Error (actual - predicted)": np.round(errors, 2),
})
st.dataframe(validation_df, use_container_width=True, hide_index=True)

validation_fig = go.Figure()
validation_fig.add_trace(go.Scatter(
    x=test.index, y=test_values, mode="lines+markers", name="Actual price",
    line={"width": 3}, marker={"size": 8},
    hovertemplate="Date: %{x|%d %b %Y}<br>Actual: %{y:,.2f}<extra></extra>",
))
validation_fig.add_trace(go.Scatter(
    x=test.index, y=predicted_values, mode="lines+markers", name="ARIMA prediction",
    line={"dash": "dash", "width": 2.5}, marker={"size": 7, "symbol": "diamond"},
    hovertemplate="Date: %{x|%d %b %Y}<br>Predicted: %{y:,.2f}<extra></extra>",
))
validation_fig.add_trace(go.Scatter(
    x=test.index, y=baseline_values, mode="lines", name="Last-price baseline",
    line={"dash": "dot", "width": 1.8},
    hovertemplate="Date: %{x|%d %b %Y}<br>Baseline: %{y:,.2f}<extra></extra>",
))
validation_fig.update_layout(
    title="Actual vs predicted prices — held-out period",
    template="plotly_white", hovermode="x unified", height=440,
    xaxis_title="Trading date", yaxis_title=f"Price ({currency_symbol})" if currency_symbol else "Price",
    legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    margin={"l": 20, "r": 20, "t": 75, "b": 25},
)
validation_fig.update_xaxes(tickformat="%d %b %Y", showgrid=True, rangeslider_visible=True)
validation_fig.update_yaxes(tickformat=",.2f", showgrid=True, zeroline=False)
st.plotly_chart(validation_fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})

# -----------------------------------------------------------------------------
# Final fit + future forecast. Reuse selected order; do not repeat the grid search.
# -----------------------------------------------------------------------------
st.subheader("📅 Future Price Forecast")
with st.spinner("Fitting the selected model on all available observations and forecasting…"):
    try:
        final_model = fit_arima(close.to_numpy(dtype=float), validation_order)
        forecast_result = final_model.get_forecast(steps=horizon)
        forecast_values = np.asarray(forecast_result.predicted_mean, dtype=float).reshape(-1)
        ci_values = np.asarray(forecast_result.conf_int(alpha=0.05), dtype=float)
        ci_lower, ci_upper = ci_values[:, 0], ci_values[:, 1]
    except Exception as exc:
        st.error(f"Could not generate the future forecast: {exc}")
        st.stop()

last_date = pd.Timestamp(close.index[-1])
if last_date.tzinfo is not None:
    last_date = last_date.tz_localize(None)
# Weekday approximation; specific exchange holidays are not modeled.
future_dates = pd.bdate_range(start=last_date.normalize() + pd.Timedelta(days=1), periods=horizon)
forecast_df = pd.DataFrame({
    "Date": future_dates.strftime("%d %b %Y"),
    "Ticker": ticker,
    "Predicted price": np.round(forecast_values, 2),
    "95% CI lower": np.round(ci_lower, 2),
    "95% CI upper": np.round(ci_upper, 2),
    "ARIMA order": [str(validation_order)] * horizon,
    "Selection criterion": [cfg["criterion"]] * horizon,
    "Selection score": [round(selected_score, 3)] * horizon,
})
st.dataframe(forecast_df, use_container_width=True, hide_index=True)
st.caption("Forecast dates skip weekends but do not exclude exchange-specific holidays. A nearly flat forecast can be normal for ARIMA multi-step forecasts: without future observations or external predictors, the model often projects toward a stable level. The actual line shows observed market movements, which the model cannot know in advance. Confidence intervals are model-based and may not capture all market risks.")

forecast_fig = make_forecast_chart(
    close, future_dates, forecast_values, ci_lower, ci_upper,
    ma_short, ma_long, ticker, validation_order, "Calendar dates" if cfg["chart_tick_style"] == "Calendar dates" else "Years only",
)
st.plotly_chart(forecast_fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})

st.subheader("⬇️ Download results")
csv_bytes = forecast_df.to_csv(index=False).encode("utf-8")
safe_ticker = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in ticker)
st.download_button(
    "📥 Download forecast CSV", data=csv_bytes,
    file_name=f"{safe_ticker}_forecast_{horizon}days.csv", mime="text/csv",
)
st.caption("Educational use only. This tool does not provide financial advice.")

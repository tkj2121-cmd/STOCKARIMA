import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ───────────────────────────────
st.set_page_config(
    page_title="Stock Forecast (ARIMA)",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Price Forecast — ARIMA Model")
st.caption("Flexible forecasting tool for any NSE/BSE/Global ticker")

# ── Sidebar: User Inputs ──────────────────────
st.sidebar.header("⚙️ Configuration")

# Ticker input
ticker_input = st.sidebar.text_input(
    "Stock Ticker Symbol",
    value="TCS.NS",
    help="Examples: TCS.NS, RELIANCE.NS, INFY.NS, AAPL, TSLA, ^NSEI"
)

# Date range
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=pd.to_datetime("2021-01-01"))
with col2:
    end_date = st.date_input("End Date", value=pd.to_datetime("2025-01-31"))

# Forecast horizon
forecast_days = st.sidebar.slider(
    "Forecast Horizon (Trading Days)",
    min_value=3, max_value=30, value=7, step=1
)

# ARIMA order
st.sidebar.subheader("ARIMA Parameters")
p_order = st.sidebar.number_input("p (AR order)", min_value=0, max_value=10, value=3)
d_order = st.sidebar.number_input("d (Integration)", min_value=0, max_value=2, value=1)
q_order = st.sidebar.number_input("q (MA order)", min_value=0, max_value=10, value=2)

# Moving average windows
ma_short = st.sidebar.number_input("Short-term MA window", min_value=3, max_value=30, value=7)
ma_long  = st.sidebar.number_input("Long-term MA window",  min_value=10, max_value=200, value=50)

run_button = st.sidebar.button("🚀 Run Forecast", type="primary", use_container_width=True)

# ── Helper: ADF Test ──────────────────────────
def adf_test(series):
    result = adfuller(series.dropna())
    return {
        "ADF Statistic": round(result[0], 4),
        "p-value": round(result[1], 4),
        "Stationary": "✅ Yes" if result[1] < 0.05 else "❌ No (differencing needed)"
    }

# ── Main Logic ────────────────────────────────
if run_button:
    ticker = ticker_input.strip().upper()

    with st.spinner(f"Downloading data for **{ticker}**..."):
        df = yf.download(
            ticker,
            start=str(start_date),
            end=str(end_date),
            auto_adjust=True,
            progress=False
        )

    if df.empty:
        st.error(f"❌ No data found for ticker **{ticker}**. Please check the symbol and try again.")
        st.stop()

    # Flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Close']].dropna()

    st.success(f"✅ Loaded **{len(df)}** trading days of data for **{ticker}**")

    # ── Section 1: Moving Averages ────────────
    df[f"MA_{ma_short}"] = df["Close"].rolling(ma_short).mean()
    df[f"MA_{ma_long}"]  = df["Close"].rolling(ma_long).mean()

    st.subheader("📊 Trend Analysis")

    fig1, ax1 = plt.subplots(figsize=(13, 5))
    ax1.plot(df["Close"], label="Actual Price", linewidth=1.5, color="#1f77b4")
    ax1.plot(df[f"MA_{ma_short}"], label=f"{ma_short}-Day MA", linestyle="--", color="#ff7f0e")
    ax1.plot(df[f"MA_{ma_long}"],  label=f"{ma_long}-Day MA",  linestyle="--", color="#2ca02c")
    ax1.set_title(f"{ticker} — Price & Moving Averages")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    st.pyplot(fig1)

    # ── Section 2: Stationarity Check ────────
    with st.expander("🔍 Stationarity Check (ADF Test)", expanded=False):
        adf_result = adf_test(df["Close"])
        st.json(adf_result)

    # ── Section 3: Train-Test Split ───────────
    if len(df) <= forecast_days:
        st.error("Not enough data for the selected forecast horizon. Reduce forecast days or extend date range.")
        st.stop()

    train = df['Close'].iloc[:-forecast_days]
    test  = df['Close'].iloc[-forecast_days:]

    # ── Section 4: Rolling Forecast ──────────
    st.subheader(f"🔄 Rolling {forecast_days}-Day Validation")

    with st.spinner("Running rolling forecast on validation set..."):
        history = train.copy()  # keeps datetime index — fixes ValueWarning & FutureWarning
        predictions = []
        for date, actual in test.items():
            model = ARIMA(history, order=(int(p_order), int(d_order), int(q_order))).fit(maxiter=200)  # maxiter fixes ConvergenceWarning
            yhat = model.forecast(steps=1).iloc[0]
            predictions.append(yhat)
            new_row = pd.Series([actual], index=[date])
            history = pd.concat([history, new_row])

    pred_series = pd.Series(predictions, index=test.index)

    # ── Section 5: Accuracy Metrics ──────────
    mae  = np.mean(np.abs(test - pred_series))
    mape = np.mean(np.abs((test - pred_series) / test)) * 100
    rmse = np.sqrt(np.mean((test - pred_series) ** 2))
    acc  = round(100 - mape, 2)

    st.subheader("📋 Model Accuracy")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE",      f"₹{round(mae,2)}"  if ".NS" in ticker or ".BO" in ticker else f"{round(mae,2)}")
    m2.metric("RMSE",     f"₹{round(rmse,2)}" if ".NS" in ticker or ".BO" in ticker else f"{round(rmse,2)}")
    m3.metric("MAPE",     f"{round(mape,2)}%")
    m4.metric("Accuracy", f"{acc}%")

    # ── Section 6: Validation Table ──────────
    validation_df = pd.DataFrame({
        "Date":             test.index.strftime("%d %b %Y"),
        "Actual Price":     test.values.round(2),
        "Predicted Price":  pred_series.values.round(2),
        "Error":            (test.values - pred_series.values).round(2)
    })
    st.dataframe(validation_df, use_container_width=True)

    # ── Section 7: Final Forecast ─────────────
    with st.spinner("Generating future forecast..."):
        final_model = ARIMA(df['Close'], order=(int(p_order), int(d_order), int(q_order))).fit(maxiter=200)
        future_forecast = final_model.forecast(steps=forecast_days)

    future_dates = pd.bdate_range(
        start=df.index[-1] + pd.Timedelta(days=1),
        periods=forecast_days
    )

    currency = "₹" if (".NS" in ticker or ".BO" in ticker) else ""

    forecast_df = pd.DataFrame({
        "Date":                  future_dates.strftime("%d %b %Y"),
        f"Predicted Price ({currency})": future_forecast.values.round(2)
    })

    st.subheader(f"📅 Next {forecast_days}-Day Forecast")
    st.dataframe(forecast_df, use_container_width=True)

    # ── Section 8: Forecast Graph ─────────────
    st.subheader("📈 Forecast Visualization")

    lookback = min(60, len(df))

    fig2, ax2 = plt.subplots(figsize=(13, 6))
    ax2.plot(df.index[-lookback:], df['Close'][-lookback:],
             label="Recent Price", marker='o', markersize=3, color="#1f77b4")
    ax2.plot(future_dates, future_forecast.values,
             linestyle='--', marker='s', markersize=5,
             color="#d62728", label=f"{forecast_days}-Day Forecast")

    ax2.fill_between(future_dates,
                     future_forecast.values * 0.97,
                     future_forecast.values * 1.03,
                     alpha=0.15, color="#d62728", label="±3% Band")

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    fig2.autofmt_xdate()

    ax2.set_title(f"{ticker} — {forecast_days}-Day Price Forecast (ARIMA {p_order},{d_order},{q_order})")
    ax2.set_ylabel("Price")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    st.pyplot(fig2)

    # ── Section 9: Download ───────────────────
    st.subheader("⬇️ Download Results")

    combined = pd.DataFrame({
        "Date":   future_dates.strftime("%d %b %Y"),
        "Ticker": ticker,
        "Predicted Price": future_forecast.values.round(2)
    })

    csv = combined.to_csv(index=False)
    st.download_button(
        label="📥 Download Forecast CSV",
        data=csv,
        file_name=f"{ticker}_forecast_{forecast_days}days.csv",
        mime="text/csv"
    )

else:
    st.info("👈 Configure the inputs in the sidebar and click **Run Forecast** to begin.")

    st.markdown("""
    ### How to Use
    | Field | Example |
    |---|---|
    | NSE Stock | `TCS.NS`, `RELIANCE.NS`, `INFY.NS` |
    | BSE Stock | `TCS.BO`, `HDFCBANK.BO` |
    | US Stock | `AAPL`, `TSLA`, `MSFT` |
    | Index | `^NSEI` (Nifty 50), `^BSESN` (Sensex), `^GSPC` (S&P 500) |
    | Crypto | `BTC-USD`, `ETH-USD` |

    ### ARIMA Parameters (p, d, q)
    - **p**: Number of lag observations (autoregressive part). Try 1–5.
    - **d**: Degree of differencing. Usually 1 for stock prices.
    - **q**: Size of moving average window. Try 1–3.
    """)

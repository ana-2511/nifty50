import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from datetime import datetime

# ===================== CONFIG & CONSTANTS ===================== #

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEQ_LEN = 60
FUTURE_STEPS = 12
HORIZONS = [1, 3, 6, 12]

MODEL_PATH = "nifty_transformer_dprice.pth"
SCALER_PATH = "feature_scaler.npy"

# same feature cols as in training
FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Return", "LogReturn",
    "HL_Spread", "OC_Spread", "Volatility", "RollingMean", "RollingStd"
]

# ===================== MODEL DEFINITION ===================== #

class NiftyTransformer(nn.Module):
    def __init__(self, n_features=len(FEATURE_COLS), d_model=96, nhead=4, num_layers=4):
        super().__init__()
        self.inp = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.05,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos = nn.Parameter(torch.randn(1, 512, d_model))

        self.price_head = nn.Linear(d_model, len(HORIZONS))
        self.max_head = nn.Linear(d_model, FUTURE_STEPS)
        self.min_head = nn.Linear(d_model, FUTURE_STEPS)

    def forward(self, x):
        # x: [B, T, F]
        B, T, F = x.size()
        x = self.inp(x) + self.pos[:, :T, :]
        h = self.encoder(x)
        h_last = h[:, -1, :]
        price_pred = self.price_head(h_last)
        max_logits = self.max_head(h_last)
        min_logits = self.min_head(h_last)
        return price_pred, max_logits, min_logits


@st.cache_resource
def load_model_and_scaler():
    # load scaler params
    scaler_data = np.load(SCALER_PATH, allow_pickle=True).item()
    scaler_mean = scaler_data["mean"]
    scaler_scale = scaler_data["scale"]

    # simple custom scaler callable
    class SimpleScaler:
        def __init__(self, mean, scale):
            self.mean_ = mean
            self.scale_ = scale

        def transform(self, X):
            return (X - self.mean_) / self.scale_

    scaler = SimpleScaler(scaler_mean, scaler_scale)

    # load model
    model = NiftyTransformer().to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    return model, scaler

# ===================== DATA PREPROCESSING ===================== #

def preprocess_raw_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Expects columns:
        - start_time (datetime-like)
        - open, high, low, close
    Optionally: end_time
    """
    df = df_raw.copy()

    # handle datetime
    if "start_time" not in df.columns:
        raise ValueError("Input CSV must contain 'start_time' column.")

    df["start_time"] = pd.to_datetime(df["start_time"])
    df = df.sort_values("start_time").reset_index(drop=True)

    # rename OHLC
    col_map = {}
    if "open" in df.columns: col_map["open"] = "Open"
    if "high" in df.columns: col_map["high"] = "High"
    if "low" in df.columns:  col_map["low"]  = "Low"
    if "close" in df.columns: col_map["close"] = "Close"
    df = df.rename(columns=col_map)

    required_ohlc = ["Open", "High", "Low", "Close"]
    for c in required_ohlc:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' missing after renaming. Check your CSV format.")

    # basic returns
    df["Return"] = df["Close"].pct_change()
    df["LogReturn"] = np.log(df["Close"]).diff()

    # engineered features
    df["HL_Spread"] = df["High"] - df["Low"]
    df["OC_Spread"] = df["Open"] - df["Close"]
    df["Volatility"] = df["Close"].pct_change().rolling(5).std()
    df["RollingMean"] = df["Close"].rolling(10).mean()
    df["RollingStd"] = df["Close"].rolling(10).std()

    df = df.dropna().reset_index(drop=True)

    return df

def scale_features(df: pd.DataFrame, scaler) -> pd.DataFrame:
    X = df[FEATURE_COLS].values
    X_scaled = scaler.transform(X)
    df_scaled = df.copy()
    df_scaled[FEATURE_COLS] = X_scaled
    return df_scaled

# ===================== INFERENCE HELPERS ===================== #

def predict_latest_window(df_scaled: pd.DataFrame, model, threshold_scale: float = 0.4):
    """
    Take last SEQ_LEN timesteps and predict:
      - multi-horizon Δ prices
      - max/min timing
      - Buy/Sell/Hold signal from horizon-1 Δ
    """
    if len(df_scaled) < SEQ_LEN + FUTURE_STEPS:
        raise ValueError(f"Not enough data. Need at least {SEQ_LEN + FUTURE_STEPS} rows after preprocessing.")

    x = df_scaled[FEATURE_COLS].iloc[-SEQ_LEN:].values.astype(np.float32)
    x_tensor = torch.tensor(x).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pr, max_logits, min_logits = model(x_tensor)

    deltas = pr.cpu().numpy()[0]               # Δ price for HORIZONS
    max_idx = int(max_logits.argmax(1).item())
    min_idx = int(min_logits.argmax(1).item())

    # dynamic threshold using small window of recent changes (for display only)
    # here we approximate using last 200 ΔClose
    closes = df_scaled["Close"].values
    if len(closes) > 201:
        recent_ret = np.diff(closes[-201:])
    else:
        recent_ret = np.diff(closes)
    base_std = np.std(recent_ret) if len(recent_ret) > 0 else 1e-6
    threshold = threshold_scale * base_std

    d1 = deltas[0]
    if d1 > threshold:
        signal = 2
    elif d1 < -threshold:
        signal = 0
    else:
        signal = 1

    return {
        "deltas": deltas,
        "max_idx": max_idx,
        "min_idx": min_idx,
        "signal": signal,
        "threshold": threshold,
    }

def backtest_strategy(df_scaled: pd.DataFrame, model, threshold_scale: float = 0.4, max_points: int = 2000):
    """
    Run a simple backtest over the last `max_points` samples using horizon-1 Δ predictions.
    """
    # we will simulate from earliest point such that we have enough lookback
    n = len(df_scaled)
    if n < SEQ_LEN + FUTURE_STEPS + 2:
        raise ValueError("Not enough data for backtest.")

    start_idx = max(SEQ_LEN, n - max_points)
    closes = df_scaled["Close"].values

    preds = []
    true_deltas = []
    positions = []
    indices = []

    for t in range(start_idx, n - FUTURE_STEPS):
        window = df_scaled[FEATURE_COLS].iloc[t-SEQ_LEN:t].values.astype(np.float32)
        # true next-bar Δ close
        c_now = closes[t-1]
        c_next = closes[t]
        true_delta_1 = c_next - c_now

        x_tensor = torch.tensor(window).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pr, _, _ = model(x_tensor)
        delta_preds = pr.cpu().numpy()[0]
        d1 = delta_preds[0]

        # dynamic threshold from last 200 moves
        hist_start = max(1, t-200)
        ref_moves = np.diff(closes[hist_start-1:t+1])
        base_std = np.std(ref_moves) if len(ref_moves) > 0 else 1e-6
        threshold = threshold_scale * base_std

        if d1 > threshold:
            pos = 1
        elif d1 < -threshold:
            pos = -1
        else:
            pos = 0

        preds.append(d1)
        true_deltas.append(true_delta_1)
        positions.append(pos)
        indices.append(t)

    preds = np.array(preds)
    true_deltas = np.array(true_deltas)
    positions = np.array(positions)
    indices = np.array(indices)

    returns = positions * true_deltas
    equity = np.cumsum(returns)

    # stats
    total_return = equity[-1] if len(equity) > 0 else 0.0
    win_rate = float(np.mean(returns > 0)) if len(returns) > 0 else 0.0
    max_dd = float(np.max(np.maximum.accumulate(equity) - equity)) if len(equity) > 0 else 0.0
    sharpe = float(np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252*48)) if len(returns) > 0 else 0.0

    bt_df = pd.DataFrame({
        "idx": indices,
        "Close": closes[indices],
        "PredDelta1": preds,
        "TrueDelta1": true_deltas,
        "Position": positions,
        "PnL": returns,
        "Equity": equity
    })

    stats = {
        "total_return": total_return,
        "win_rate": win_rate,
        "max_dd": max_dd,
        "sharpe": sharpe
    }

    return bt_df, stats

# ===================== STREAMLIT UI ===================== #

st.set_page_config(
    page_title="NIFTY50 Transformer Trading Dashboard",
    layout="wide"
)

st.title("📈 NIFTY50 Transformer – Multi-Horizon Δ Price & Trading Signals")

st.sidebar.header("1️⃣ Upload NIFTY50 OHLC Data")
uploaded = st.sidebar.file_uploader(
    "Upload CSV with columns: start_time, open, high, low, close",
    type=["csv"]
)

st.sidebar.header("2️⃣ Signal Settings")
threshold_scale = st.sidebar.slider(
    "Signal Threshold Scale (× volatility)",
    min_value=0.1, max_value=1.0, value=0.4, step=0.05
)
# Sidebar configuration panel
st.sidebar.header("⚙ Configuration")

with st.sidebar.expander("Model Parameters (Advanced Users)", expanded=False):
    SEQ_LEN = st.number_input("Lookback Window (SEQ_LEN)", min_value=30, max_value=200, value=60, step=5)
    FUTURE_STEPS = st.number_input("Future Window for Extremes", min_value=5, max_value=60, value=12, step=1)

    HORIZONS_input = st.text_input(
        "Price Δ Forecast Horizons (comma-separated)",
        value="1,3,6,12"
    )
    try:
        HORIZONS = [int(x.strip()) for x in HORIZONS_input.split(",") if x.strip().isdigit()]
    except:
        st.error("Invalid HORIZONS format. Use comma-separated integers.")
run_bt = st.sidebar.checkbox("Run Backtest on last N points", value=True)
max_points = st.sidebar.slider("Max points for backtest", min_value=500, max_value=5000, value=2000, step=500)

st.sidebar.header("3️⃣ Model / Scaler")
st.sidebar.write(f"Using device: `{DEVICE}`")
st.sidebar.write(f"Model file: `{MODEL_PATH}`")
st.sidebar.write(f"Scaler file: `{SCALER_PATH}`")

# load model & scaler once
try:
    model, scaler = load_model_and_scaler()
except Exception as e:
    st.error(f"Error loading model/scaler: {e}")
    st.stop()

if uploaded is None:
    st.info("👆 Upload your NIFTY50 intraday CSV to start.")
    st.stop()

# ===================== MAIN LOGIC ===================== #

try:
    # use the file we already uploaded in the sidebar
    # DO NOT create a second file_uploader here
    df_raw = pd.read_csv(uploaded)

    st.subheader("Raw Data Preview")
    st.dataframe(df_raw.head())
    st.write(f"Rows: {len(df_raw)}")

    df_pp = preprocess_raw_df(df_raw)
    df_scaled = scale_features(df_pp, scaler)

    st.subheader("Preprocessed & Scaled Data (Head)")
    st.dataframe(df_scaled.head())

    # ---- Latest prediction ---- #
    st.subheader("🔮 Latest Window Prediction")

    latest = predict_latest_window(df_scaled, model, threshold_scale=threshold_scale)
    deltas = latest["deltas"]
    max_idx = latest["max_idx"]
    min_idx = latest["min_idx"]
    signal = latest["signal"]
    threshold_used = latest["threshold"]

    signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Latest Signal", signal_map[signal])
        st.write("Multi-Horizon Δ Price Predictions:")
        for h, d in zip(HORIZONS, deltas):
            st.write(f"• Horizon {h}: {d:.5f}")

    with col2:
        st.write("Extreme Timing (in next FUTURE_STEPS bars):")
        st.write(f"• Max High expected at +{max_idx} steps")
        st.write(f"• Min Low expected at +{min_idx} steps")
        st.write(f"Threshold used for 1-step signal: `{threshold_used:.6f}`")

    # ---- Price plot with last 500 points ---- #
    st.subheader("📊 Recent Price Chart")
    show_n = min(500, len(df_scaled))
    sub_df = df_scaled.iloc[-show_n:]

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=sub_df["start_time"] if "start_time" in sub_df.columns else sub_df.index,
        y=sub_df["Close"],
        mode="lines",
        name="Close"
    ))
    fig_price.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_price, use_container_width=True)

    # ---- Backtest ---- #
    if run_bt:
        st.subheader("📈 Strategy Backtest (1-step Δ-based signals)")

        bt_df, stats = backtest_strategy(df_scaled, model, threshold_scale=threshold_scale, max_points=max_points)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Return (Δ units)", f"{stats['total_return']:.4f}")
        c2.metric("Win Rate", f"{stats['win_rate']*100:.2f}%")
        c3.metric("Max Drawdown", f"{stats['max_dd']:.4f}")
        c4.metric("Sharpe Ratio", f"{stats['sharpe']:.4f}")

        # Plot price + signals + equity
        fig = go.Figure()

        # price
        fig.add_trace(go.Scatter(
            x=bt_df["idx"],
            y=bt_df["Close"],
            mode="lines",
            name="Price"
        ))

        # buys
        buys = bt_df[bt_df["Position"] == 1]
        fig.add_trace(go.Scatter(
            x=buys["idx"],
            y=buys["Close"],
            mode="markers",
            marker=dict(color="green", size=7),
            name="BUY"
        ))

        # sells
        sells = bt_df[bt_df["Position"] == -1]
        fig.add_trace(go.Scatter(
            x=sells["idx"],
            y=sells["Close"],
            mode="markers",
            marker=dict(color="red", size=7),
            name="SELL"
        ))

        fig.update_layout(
            title="Price with Buy/Sell Signals",
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)

        # equity curve
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=bt_df["idx"],
            y=bt_df["Equity"],
            mode="lines",
            name="Equity"
        ))
        fig_eq.update_layout(
            title="Equity Curve",
            height=300,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_eq, use_container_width=True)

except Exception as e:
    st.error(f"Error while processing: {e}")

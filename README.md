📈 NIFTY50 Transformer – Multi-Horizon Price Forecasting & Trading Signals
AI-powered intraday market prediction using Transformers
🚀 Overview

This project builds an AI-driven stock analytics system designed to assist intraday traders by forecasting market movement using a Transformer deep learning architecture.

📌 Key capabilities:

Predict future Δ-close price movements across multiple time horizons

Identify timing of intraday maximum & minimum price

Generate actionable Buy / Sell / Hold signals

Visualize price history, AI signals & equity curve

Backtest strategy on recent historical data to validate performance

Upload a CSV of intraday OHLC candles → see predictions instantly.

🧠 How It Works

The AI model processes the most recent 60 timesteps (5-min candles) to understand short-term market structure.
For the next 12 timesteps, the model predicts:

Output	Description
Δ Price Forecasts	Expected price change over horizons 1, 3, 6 & 12
Max Timing	Step index where price is most likely to peak
Min Timing	Step index where price is most likely to bottom
Trade Signal	BUY / HOLD / SELL based on short-term Δ

The result = zero-noise trading signal, derived purely from the price trajectory confidence of the model.

🗂 Input Data Format

Upload a CSV containing the following columns:

Column	Required	Description
start_time	✔	Timestamp of candle
open	✔	Candle open
high	✔	Candle high
low	✔	Candle low
close	✔	Candle close
end_time	optional	Ignored if present

📌 Recommended dataset: NIFTY50 or any Nifty index 5-minute OHLC

🧩 Tech Stack
Component	Technology
Model	PyTorch Transformer Encoder
UI	Streamlit
Data	CSV-based OHLC candles
Visualization	Plotly
Backtesting	Custom Δ-price based engine
🎯 Features in the Dashboard
Section	What You Get
Raw Data Preview	Before preprocessing
Scaled & engineered features	For transparency
Latest Forecast	Multi-horizon Δ prices & extremes timing
Trading Signal	BUY / HOLD / SELL display
Interactive Chart	500-point price history + prediction markers
Strategy Backtest	Win rate, drawdown, Sharpe & equity curve
📸 Screenshots (Suggested)
AI Signal	Backtest	Equity Curve
BUY/HOLD/SELL	Marked entries/exits	Performance over time

You can add screenshots from your running app here.

🏁 Getting Started
1️⃣ Install requirements
pip install -r requirements.txt

2️⃣ Place model & scaler in project root
nifty_transformer_dprice.pth
feature_scaler.npy

3️⃣ Launch the app
streamlit run app.py

📌 Notes
Setting	Recommendation
SEQ_LEN	keep 60
FUTURE_STEPS	keep 12
Δ horizons	keep 1,3,6,12
Dataset size	≥ 100 rows for prediction
🧾 Citation (If used for research)
Transformer-Based Intraday Price & Trend Forecasting for NIFTY50

🤝 Contribution

Pull requests are welcome — ideas to add:

Risk-adjusted reinforcement learning strategy

Telegram / WhatsApp signal alerts

Real-time live feed integration (5 sec tick)

⭐ Show Your Support

If this project helps your learning or trading research:

🌟 Star this repository


and feel free to connect!

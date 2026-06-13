# 📈 NIFTY-50 Investment Intelligence Platform

> An AI-powered investment decision-support system built on 20+ years of NIFTY-50 historical market data (2000–2021).

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)

---

## 📌 Project Overview

This project transforms raw NIFTY-50 historical stock market data into actionable investment intelligence through an interactive web application. It demonstrates the end-to-end data science workflow: exploratory data analysis, feature engineering, machine learning modeling, portfolio optimization, and risk analytics, all delivered through a polished Streamlit dashboard.

The platform is built around **decision support, not point prediction** — instead of presenting a single "the price will be X" forecast, every prediction is shown alongside its uncertainty (Monte Carlo price ranges, confusion matrices, rolling accuracy vs. a coin-flip baseline) so users can make informed decisions rather than relying on a false sense of certainty.

### Key Modules

| Module | Description |
|--------|-------------|
| **Stock Analysis Explorer** | Interactive charts and technical indicators for any NIFTY-50 stock |
| **AI Stock Predictor** | Random Forest & XGBoost direction classifiers, an XGBoost regressor for point price/return forecasts, and a Monte Carlo (Geometric Brownian Motion) simulation that turns the regressor's estimate into a probabilistic price range |
| **Portfolio Builder** | Optimized portfolios for 3 investor profiles: Conservative, Balanced, Aggressive |
| **Risk Assessment** | Volatility, Sharpe, Sortino, Calmar, Value-at-Risk (VaR), CVaR, and Maximum Drawdown metrics |

---

## 📁 Project Structure

```
nifty-50-investment-intelligence/
│
├── README.md                        ← You are here
├── requirements.txt                 ← All Python dependencies
├── app.py                           ← Main Streamlit entry point
├── .gitignore
│
├── .streamlit/
│   └── config.toml                  ← Streamlit theme configuration
│
├── config/
│   └── settings.py                  ← Global config (paths, constants)
│
├── data/                            ← Place downloaded CSV files here (see Dataset section)
│
├── notebooks/                       ← Analysis pipeline (run in order)
│   ├── 01_EDA.py                    ← Exploratory Data Analysis
│   ├── 02_Feature_Engineering.py    ← Technical indicator engineering
│   ├── 03_Stock_Prediction.py       ← Model training & evaluation
│   └── 04_Portfolio_Risk.py         ← Portfolio optimization & risk
│
└── src/
    ├── utils/
    │   ├── data_loader.py           ← Load & preprocess CSV data
    │   └── indicators.py            ← Technical indicator calculations
    │
    ├── models/
    │   ├── predictor.py             ← Stock prediction models (RF, XGBoost, Regressor, Monte Carlo)
    │   ├── portfolio.py             ← Portfolio construction & optimization
    │   └── risk.py                  ← Risk metric calculations
    │
    ├── components/
    │   ├── charts.py                ← Reusable Plotly chart functions
    │   └── metrics_display.py       ← KPI cards and metric formatters
    │
    └── pages/
        ├── home.py                  ← Landing / overview page
        ├── stock_analysis.py        ← Individual stock explorer
        ├── predictor.py             ← Prediction engine UI
        ├── portfolio.py             ← Portfolio builder UI
        └── risk.py                  ← Risk assessment UI
```

---

## ⚙️ Environment Setup

### Step 1 — Clone the Repository

```bash
git clone https://github.com/rakshit11147/nifty-50-investment-intelligence.git
cd nifty-50-investment-intelligence
```

### Step 2 — Create a Virtual Environment

```bash
# windows:
python -m venv venv
venv\Scripts\activate

# mac/linux
python -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note (Apple Silicon / M1/M2/M3 Macs):** if `lightgbm` fails to build, run `brew install libomp` first, then re-run `pip install -r requirements.txt`. LightGBM is currently an optional/unused dependency and can be safely removed from `requirements.txt` if you don't need it.

---

## 📦 Step 4 — Dataset

The combined dataset `data/NIFTY50_all.csv` is **not included** in this repository due to size constraints.

Download the full [Kaggle NIFTY-50 Stock Market Dataset](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data/data), extract it, and place all the per-stock CSV files in a `data/` folder in the project root:

```
data/
├── ADANIPORTS.csv
├── ASIANPAINT.csv
├── AXISBANK.csv
├── ...
└── WIPRO.csv
```

---

## 🚀 Running the Application

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

---

## 🧪 Reproducing the Analysis

Run the analysis pipeline scripts in order:

```bash
python notebooks/01_EDA.py
python notebooks/02_Feature_Engineering.py
python notebooks/03_Stock_Prediction.py
python notebooks/04_Portfolio_Risk.py
```

---

## 👥 Team

| Member | Name |
|--------|------|
| Member 1 | Rakshit Raj |
| Member 2 | Anuj Jangir |

---

## 📊 Feature Engineering

The following technical indicators are computed in `src/utils/indicators.py`:

| Indicator | Description |
|-----------|-------------|
| SMA_20, SMA_50, SMA_200 | Simple Moving Averages |
| EMA_12, EMA_26 | Exponential Moving Averages |
| MACD, MACD_Signal, MACD_Hist | MACD and Signal Line |
| RSI_14 | Relative Strength Index |
| BB_Upper, BB_Lower, BB_Width, BB_Pct_B | Bollinger Bands |
| ATR_14, ATR_Pct | Average True Range |
| OBV | On-Balance Volume |
| ROC | Rate of Change |
| Daily_Return | Log Returns |
| Volatility_20, Volatility_50 | Rolling volatility |
| Momentum_5, Momentum_20 | Price momentum |

---

## 📐 Models and Methodology

| Task | Model / Technique | Library |
|------|-------------------|---------|
| Direction Prediction (h-day horizon) | Random Forest | scikit-learn |
| Direction Prediction (h-day horizon) | XGBoost Classifier | XGBoost |
| Point Price / Return Forecast | XGBoost Regressor | XGBoost |
| Probabilistic Price Outlook | Monte Carlo Simulation (Geometric Brownian Motion, 500 paths) | NumPy |
| Portfolio Optimization | Mean-Variance / Efficient Frontier | SciPy (SLSQP) |
| Model Explainability | Gain-based Feature Importance | XGBoost |

### Probabilistic Price Outlook

Rather than presenting a single predicted price, the dashboard simulates 500 Geometric Brownian Motion paths calibrated to each stock's historical drift and volatility, producing 50% and 90% confidence price bands for the forecast horizon. The XGBoost regressor's point estimate ("ML target") is overlaid on this distribution — giving users both the model's best guess **and** the realistic range of outcomes around it.

### Direction Model Diagnostics

For Random Forest and XGBoost direction classifiers, the dashboard shows:
- Accuracy, precision, and recall on a chronological (no-shuffle) holdout
- A confusion matrix for the last 100 test samples
- A rolling 20-sample accuracy chart benchmarked against the 50% coin-flip baseline

### Risk Metrics

- **Volatility** – annualized standard deviation of daily returns
- **Sharpe Ratio** – risk-adjusted return relative to total volatility
- **Sortino Ratio** – risk-adjusted return penalizing only downside volatility
- **Calmar Ratio** – annual return relative to maximum drawdown
- **Value at Risk (VaR)** – worst expected daily loss at 95% confidence (historical & parametric)
- **CVaR (Expected Shortfall)** – average loss in the tail beyond VaR
- **Maximum Drawdown** – largest peak-to-trough decline

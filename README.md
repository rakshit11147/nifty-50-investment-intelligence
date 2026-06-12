# 📈 NIFTY-50 Investment Intelligence Platform

> An AI-powered investment decision-support system built on 20+ years of NIFTY-50 historical market data (2000–2021).

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)


---

## 📌 Project Overview

This project transforms raw NIFTY-50 historical stock market 
data into actionable investment intelligence through an interactive web application. It demonstrates the end-to-end data science workflow: exploratory data analysis, feature engineering, machine learning modeling, portfolio optimization, and risk analytics, all delivered through a polished Streamlit dashboard.

### Key Modules

| Module | Description |
|--------|-------------|
| **Stock Analysis Explorer** | Interactive charts and technical indicators for any NIFTY-50 stock |
| **AI Stock Predictor** | ML-based price and direction forecasting (Random Forest, XGBoost, LightGBM ensemble) |
| **Portfolio Builder** | Optimized portfolios for 3 investor profiles: Conservative, Balanced, Aggressive |
| **Risk Assessment** | Volatility, Sharpe, Sortino, Value-at-Risk (VaR), and Maximum Drawdown metrics |

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
├── data/
│   └── NIFTY50_all.csv              ← Combined NIFTY-50 dataset (included)
│
├── notebooks/                       ← Analysis pipeline (run in order)
│   ├── 01_EDA.PY                    ← Exploratory Data Analysis
│   ├── 02_Feature_Engineerng.py     ← Technical indicator engineering
│   ├── 03_Stock_pridiction.py       ← Model training & evaluation
│   └── 04_Portfolio_Risk.py         ← Portfolio optimization & risk
│
└── src/
    ├── utils/
    │   ├── data_loader.py           ← Load & preprocess CSV data
    │   └── indicators.py            ← Technical indicator calculations
    │
    ├── models/
    │   ├── predictor.py             ← Stock prediction models (RF, XGBoost)
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
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

---

## step 4- 📦 Dataset

The combined dataset `data/NIFTY50_all.csv` is **not included ** in this repository due to size constraints.

To work with per-stock CSV files , download the full
[Kaggle NIFTY-50 Stock Market Dataset](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data/data) . 

Create a folder name 'data' in the project root
and place the extracted CSV files in the `data/` folder:


```
data/
├── ADANIPORTS.csv
├── ASIANPAINT.csv
├── AXISBANK.csv
├── ...
├── WIPRO.csv
└── NIFTY50_all.csv
```

---

## 🚀 Running the Application

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.



---

## 👥 Team

| Member 1 | Rakshit Raj |
| Member 2 | Anuj Jangir |

---


## 📚 Research and Development

The `notebooks/` directory contains the complete analysis pipeline used during development, including:

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Stock Prediction Model Development
- Portfolio Optimization
- Risk Analysis

These notebooks document the research and experimentation process behind the application.


---

## 📊 Feature Engineering

The following technical indicators are computed in `src/utils/indicators.py`:

| Indicator | Description |
|-----------|-------------|
| SMA_20, SMA_50, SMA_200 | Simple Moving Averages |
| EMA_12, EMA_26 | Exponential Moving Averages |
| MACD, Signal | MACD and Signal Line |
| RSI_14 | Relative Strength Index |
| BB_upper, BB_lower | Bollinger Bands |
| ATR_14 | Average True Range |
| OBV | On-Balance Volume |
| ROC | Rate of Change |
| Daily_Return | Log Returns |
| Volatility_20 | Rolling 20-day volatility |

---

## 📐 Models and Methodology

| Task | Model / Technique | Library |
|------|-------------------|---------|
| Direction Prediction | Random Forest | scikit-learn |
| Direction Prediction | XGBoost | XGBoost |
| Direction Prediction | LightGBM | LightGBM |
| Ensemble | Voting Classifier | scikit-learn |
| Portfolio Optimization | Mean-Variance / Efficient Frontier | PyPortfolioOpt, cvxpy |
| Model Explainability | SHAP values | shap |

### Risk Metrics

- **Sharpe Ratio** – risk-adjusted return relative to total volatility
- **Sortino Ratio** – risk-adjusted return penalizing only downside volatility
- **Value at Risk (VaR)** – worst expected loss at a given confidence level
- **Maximum Drawdown** – largest peak-to-trough decline
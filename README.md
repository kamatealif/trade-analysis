# Trader Performance vs Market Sentiment (Hyperliquid)

This repository is my complete submission for the **Data Science / Analytics Intern - Round-0 Assignment**.

## Executive Summary
The objective is to evaluate how market sentiment (Fear/Greed) relates to trader behavior and performance, and then convert findings into actionable strategy rules.

What this submission delivers:
- A reproducible end-to-end analysis pipeline (cleaning, alignment, feature engineering, analysis, visualization).
- Requirement-by-requirement coverage for Part A, Part B, and Part C.
- Advanced evaluation: significance testing, out-of-sample strategy validation, segment risk profiling, predictive baseline.
- A simple interactive dashboard for exploration.

## 1) Requirement Coverage (Traceability)

### Part A - Data Preparation (Must-have)
Required tasks:
1. Load both datasets and document rows/columns, missing values, duplicates.
2. Convert timestamps and align datasets by date.
3. Build key metrics (daily PnL, win rate, avg trade size, trades/day, long-short ratio, etc.).

Implemented in:
- `main.ipynb` -> Part A cells
- Output files: `fear_greed_index_clean.csv`, `historical_data_clean.csv`, `account_daily_metrics.csv`, `daily_market_metrics.csv`

### Part B - Analysis (Must-have)
Required tasks:
1. Compare performance in Fear vs Greed.
2. Analyze behavior changes by sentiment.
3. Create trader segments.
4. Provide insights backed by tables/charts.

Implemented in:
- `main.ipynb` -> Part B + Visualizations sections
- Output files: `daily_trade_metrics.csv`, `fear_greed_class_summary.csv`, `coin_summary.csv`, `trader_features_segments.csv`

### Part C - Actionable Output (Must-have)
Required task:
- Propose strategy rules from findings.

Implemented in:
- `main.ipynb` -> Part C section
- Output text summary: `results_summary.md`

### Bonus Implementations Added
- Statistical significance tests for Fear vs Greed differences.
- Out-of-sample validation of strategy rules (time split).
- Risk metrics by trader segment (drawdown, VaR, CVaR).
- Lightweight predictive baseline (next-day profitability bucket).
- Streamlit dashboard.

Implemented in:
- `main.ipynb` -> Part D and Part E sections
- `streamlit_app.py`

## 2) Data Used and Scope
- Fear/Greed cleaned rows: **2,644**
- Historical trade rows (cleaned): **211,224**
- Daily sentiment match coverage: **211,218 / 211,224**
- Analysis period: **2023-05-01 to 2025-05-01**

## 3) Data Quality and Cleaning Decisions

### 3.1 Normalization and Standardization
- Renamed columns to consistent snake_case.
- Standardized text fields (`coin`, `side`, `direction`) for stable grouping.
- Converted numeric fields with `errors='coerce'` to avoid silent parse failures.

### 3.2 Timestamp Handling
- Parsed `Timestamp IST` as the authoritative event time.
- Built `timestamp_utc` from IST for consistent chronological analysis.
- Derived `trade_date` for daily alignment with sentiment data.

### 3.3 ID / Timestamp Integrity Issue in Raw Export
- Raw `Timestamp` contains only **7 unique values** over **211,224 rows**, indicating low resolution.
- Raw IDs/timestamps were exported in scientific notation.
- Recovery logic converts scientific-notation values to integer-like strings when possible.
- Final alignment and time-series analysis rely on `Timestamp IST` to avoid sequencing errors.

### 3.4 Notable Edge Cases Retained in Analysis
- Negative fee rows: **2,476** (likely rebates/maker behavior).
- Zero-USD rows: **43**.
- Non-zero PnL rows: **104,408**.

## 4) Methodology

### 4.1 Feature Engineering
Constructed metrics at multiple levels.

Account-day metrics:
- `total_pnl`: sum of realized/closed PnL at account-day level.
- `win_rate`: wins / non-zero PnL events.
- `trades`: trade count per account-day.
- `avg_trade_size_usd`: mean position size in USD.
- `long_short_ratio`: long-trade count / short-trade count.
- `drawdown_proxy`: cumulative PnL minus running cumulative peak.

Market-day metrics:
- `trades`, `total_volume_usd`, `total_closed_pnl`, `buy_ratio`, `fg_value`.

Segment labels:
- `segment_frequency`: frequent vs infrequent traders.
- `segment_size`: high-size vs low-size traders.
- `segment_consistency`: consistent winners vs inconsistent.

### 4.2 Sentiment Alignment Strategy
- Joined historical trades with Fear/Greed on daily date.
- Used both coarse regime (`Fear`, `Greed`, `Neutral`) and full class labels (`Extreme Fear`, `Fear`, `Neutral`, `Greed`, `Extreme Greed`).

### 4.3 Advanced Statistical/Evaluation Layer
- Significance tests:
  - permutation test for mean difference (Fear minus Greed)
  - bootstrap CI for difference estimates
  - Cohen's d effect size
- Out-of-sample validation:
  - chronological train/test split
  - compared baseline vs regime-only vs selected-account strategy
- Segment risk metrics:
  - max drawdown
  - VaR 95
  - CVaR 95
  - volatility
- Predictive baseline:
  - target: next-day profitability bucket
  - models: majority baseline and logistic regression baseline (if available)

## 5) Results and Findings

### 5.1 Sentiment vs Activity / Performance
- Correlation(FG value, daily volume): **-0.264**
- Correlation(FG value, daily non-zero closed PnL): **-0.083**
- Correlation(FG value, daily buy ratio): **-0.049**

Interpretation:
- Lower sentiment regimes are associated with higher activity/volume.
- Linear relationship between sentiment and realized PnL is weaker than sentiment-volume relationship.

### 5.2 Fear vs Greed Performance Comparison
- Mean daily PnL on Fear days: **5185.15**
- Mean daily PnL on Greed days: **4144.21**
- Mean win rate on Fear days: **0.8423**
- Mean win rate on Greed days: **0.8563**

Interpretation:
- Fear regime shows higher mean daily PnL in this sample.
- Greed regime shows slightly higher mean win rate.

### 5.3 Class-Level Behavior
- Highest aggregate volume is in `Fear`:
  - total volume: **483,324,789.79**
  - trades: **61,837**

### 5.4 Coin-Level PnL Concentration
- Best total PnL coin: **@107** (**2,783,912.92**)
- Worst total PnL coin: **TRUMP** (**-364,824.91**)

### 5.5 Execution/Direction Pattern
- Realized PnL is concentrated in closing directions (`Close Long`, `Close Short`, `Sell`).
- Opening directions are near-zero realized PnL by construction/behavior.

### 5.6 Intraday Pattern
- Most active IST hours include **19, 20, 21, 01, 03, 04**.

## 6) Actionable Strategy Rules
1. Regime-aware risk sizing:
   - keep normal sizing in the stronger validated regime,
   - tighten risk limits in the weaker regime.
2. Segment-aware capital allocation:
   - prioritize `ConsistentWinner` profiles,
   - cap exposure for `Inconsistent` profiles until stability improves.
3. Guardrails for unstable behavior:
   - for low win-rate high-frequency accounts, enforce trade-count caps and lower position sizing.

## 7) Visual and Interactive Outputs

Notebook visualizations in `main.ipynb` include:
- Fear/Greed trend + moving average
- daily trades and PnL trend (including cumulative PnL)
- sentiment-vs-outcome scatter plots
- sentiment-class bar comparisons
- top/bottom coin PnL charts
- intraday hour activity + PnL profile

Interactive dashboard in `streamlit_app.py` includes tabs for:
- Overview
- Sentiment
- Coins
- Segment Risk
- Out-of-Sample Validation
- Predictive Baseline

## 8) Reproducibility

### 8.1 Environment Setup
```bash
pip install -r requirements.txt
```

### 8.2 Run Notebook (Primary)
- Open `main.ipynb`
- Run cells top-to-bottom

### 8.3 Run Script Pipeline
```bash
python clean_and_analyze.py
```

### 8.4 Run Dashboard
```bash
streamlit run streamlit_app.py
```

## 9) Repository Contents
- `main.ipynb`: complete assignment workflow + advanced evaluation
- `clean_and_analyze.py`: script pipeline
- `streamlit_app.py`: dashboard
- `checkin.ipynb`: iterative notebook version
- `requirements.txt`: dependencies
- output data artifacts:
  - `fear_greed_index_clean.csv`
  - `historical_data_clean.csv`
  - `daily_trade_metrics.csv`
  - `fear_greed_class_summary.csv`
  - `coin_summary.csv`
  - `account_daily_metrics.csv`
  - `daily_market_metrics.csv`
  - `trader_features_segments.csv`

## 10) Limitations
- Raw historical `Timestamp` is coarse/low-resolution; `Timestamp IST` was used as authoritative time.
- Insights are correlational/descriptive, not causal claims.
- Predictive baseline is intentionally lightweight and should be expanded before production deployment.

## 11) Next High-Impact Improvement
A strong next step is a walk-forward backtest with transaction-cost-aware execution assumptions and segment-specific risk constraints.
# trade-analysis

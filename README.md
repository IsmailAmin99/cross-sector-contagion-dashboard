# Cross-Sector Contagion Dashboard 
#### Authors: Isma'il Amin & Danielson Azumah
#### Course: CSCi-3412 Machine Learning 

## Project Overview
The **Cross-Sector Contagion Dashboard** is a predictive market analytics tool designed to map and visualize how macro-economic & geopolitical stress propgates through different market sectors. This project utilizes Graph Attention Networks (GAT) to quantify the "contagion effect" between 11 major sector ETFs during high-volatility events. 

## Technical Implementation
- **Graph Neural Networks:** We treat market sectors as nodes in a graph. The GAT architecture learns dynamic attention weghts to identify which sectors act as primary "transmitters" of stress during specific event types.
- **Predictive Modeling:** Implements Gradient Boosted Regressors to forecast drawdown depth based on event category & severity.
- **Adversarial Defense:** Includes an input validation layer designed to detect & mitigate data poisoning attacks on the regression models.
- **Backtesting Strategy:** A custom contagion strategy that rebalances portfolios based on predicted sector drawdowns, currently outperforming the S&P 500 benchmark.

## Key Features 
- **Event Impact Explorer:** Select historical shocks (i.e., SVB Collapse, Inflation Surprises) to visualize how stress ripples through the market graph.
- **Risk Metrics:** Interactive correlation heatmaps and drawdown comparisons for real-time risk assessment.
- **Model Transparency:** Visualization of attention weights to provide interpretability for the GAT's predictions.

## Installation & Setup
1. CLone the repository:
```bash
git clone https://github.com/IsmailAmin99/cross-sector-contagion-dashboaard.git
cd cross-sector-contagion-dashboard
```

2. Install requirements:
```bash
pip install streamlit pandas plotly scikit-learn yfinance
```

3. Run the app:
```bash
streamlit run contagion_dashboard.py
```

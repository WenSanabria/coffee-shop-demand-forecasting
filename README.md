# Daily Product Demand Forecasting

A privacy-safe machine learning portfolio project that forecasts **daily menu-item demand** for a small coffee shop. The project demonstrates an end-to-end analytics workflow: data quality cleanup, feature engineering, time-aware validation, model comparison, and a business-facing forecasting demo.

## Business problem

Small food businesses must decide how much product to prepare before demand is known. Overproduction creates waste and higher costs, while underproduction can cause stockouts and lost sales. This project estimates next-day demand at the menu-item level to support production and inventory planning.

## Project highlights

- Refactored from a graduate data science project that originally analyzed **286K+ private POS transactions**
- Standardizes messy product text using Unicode repair, canonicalization, and fuzzy matching
- Aggregates raw transactions into daily item-level demand
- Engineers calendar, lag, rolling-average, price, holiday, and weather-proxy features
- Uses an **80/20 time-based train/test split** instead of random shuffling
- Compares a lag-1 baseline, Decision Tree, Random Forest, and Gradient Boosting
- Includes reusable Python modules plus a Jupyter walkthrough
- Provides a leakage-aware **one-day-ahead** forecasting helper

## Original academic results

The original project used private Toast POS data supplied by the business owner. That dataset is **not distributed** in this repository. On the original data, Gradient Boosting produced the best RMSE in the completed model comparison:

| Model | MAE | RMSE |
|---|---:|---:|
| Gradient Boosting | 4.00 | 6.98 |
| Random Forest | 4.37 | 7.70 |
| Decision Tree | 4.44 | 8.08 |
| Baseline (Lag-1) | 4.05 | 8.74 |

These metrics are included as academic-project results. Running this public repository on the synthetic sample will produce different values.

## Repository structure

```text
coffee-shop-demand-forecasting/
├── README.md
├── run_forecast.py
├── requirements.txt
├── data/
│   └── sample_pos_transactions.csv
├── notebooks/
│   └── demand_forecasting_portfolio.ipynb
├── src/
│   └── forecasting.py
├── reports/
│   └── project_summary.md
└── assets/
    ├── model_comparison.png
    └── actual_vs_predicted.png
```

## Run locally

```bash
git clone <your-repository-url>
cd coffee-shop-demand-forecasting
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_forecast.py
```

To explore the analysis interactively:

```bash
jupyter lab
```

Then open `notebooks/demand_forecasting_portfolio.ipynb`.

## Modeling approach

1. **Clean transactions:** remove voids, parse timestamps, and standardize product names.
2. **Create daily demand:** aggregate quantity sold by date and menu item.
3. **Engineer features:** weekday/weekend, month/year, lag-1, lag-7, 7-day rolling mean, average price, holidays, and a deterministic seasonal weather proxy.
4. **Validate through time:** train on the first 80% of dates and test on the later 20%.
5. **Compare models:** evaluate a simple lag baseline against tree-based regressors with MAE, RMSE, and MAPE.
6. **Forecast:** use the selected model for a one-day-ahead prediction from observed history.

## Data privacy

The original business dataset is private and is intentionally excluded. `data/sample_pos_transactions.csv` is synthetic and exists only so the repository can be cloned and run end to end. The `.gitignore` also blocks common names/locations for the original dataset.

## Skills demonstrated

Python · pandas · NumPy · scikit-learn · data cleaning · fuzzy matching · feature engineering · time-series validation · forecasting · model evaluation · reproducible analytics workflows · business problem framing

## About the Author

**Wendy Eidson**  
Wendy Sanabria is a Data Operations Analyst with experience in Python, PySpark, SQL, Databricks, data automation, analytics, and energy-efficiency data. She is completing an M.S. in Data Science at Colorado School of Mines.

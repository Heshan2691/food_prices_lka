# Sri Lanka Market Food Price Forecasting — Ensemble Learning & Hybrid ML Dashboard

An end-to-end Machine Learning & Ensemble Learning project trained on the official **Sri Lanka Food Prices** dataset published by the **UN World Food Programme (WFP) on HDX** (compiled from Sri Lanka's Department of Census & Statistics and HARTI).

## Official Data Source
- **Portal:** [HDX — Sri Lanka Food Prices](https://data.humdata.org/dataset/wfp-food-prices-for-sri-lanka)
- **Reproducible Fetcher:** Run `python fetch_data.py` to download the latest official CSV directly into `data/wfp_food_prices_lka.csv`.

## Project Structure
```text
food_prices_lka/
├── data/
│   └── wfp_food_prices_lka.csv
├── models/
│   └── best_food_price_model.joblib
├── results/
│   ├── model_comparison.csv
│   ├── feature_importance.csv
│   ├── market_summary.csv
│   ├── commodity_history.csv
│   └── *.png
├── .streamlit/
│   └── config.toml
├── fetch_data.py
├── notebook.ipynb          # Full ML pipeline + Day 4 analysis + GenAI declaration
├── app.py                  # Interactive Streamlit analytics dashboard
└── requirements.txt
```

## Models Compared
- **Bagging:** Random Forest Regressor
- **Boosting:** XGBoost, LightGBM, CatBoost
- **Stacking:** Random Forest + XGBoost + LightGBM → Ridge meta-learner
- **Hybrid (`RF + ANN`):** Two-stage out-of-fold Random Forest + Artificial Neural Network (`MLPRegressor`)
- **Hyperparameter Tuning:** 5-fold `RandomizedSearchCV` on LightGBM

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

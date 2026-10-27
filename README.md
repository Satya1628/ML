# Airline Ticket Price Prediction

A compact end-to-end regression project using the supplied Indian flight-price dataset.

## Dataset files supplied
- `Clean_Dataset.csv` (used): 300,153 rows; includes airline, source/destination city, departure/arrival time, stops, class, duration, days left, and price.
- `business.csv` and `economy.csv`: separate subsets with a different schema and without `days_left`/`class` in the same format. They are retained in `data/source_files/` for reference, but the main pipeline uses `Clean_Dataset.csv` for a consistent feature set.

## Model workflow
1. Inspect data quality and distributions.
2. Use a random 60% train / 20% validation / 20% test split (validation is split from the original 80% training portion).
3. Preprocess categorical columns with one-hot encoding; impute/scale numeric features in a pipeline.
4. Compare Ridge Regression with Random Forest on validation MAE.
5. Select the best validation model, refit it on train + validation data, and report final test metrics.
6. Use `app.py` to enter flight details and estimate a fare.

## Final test results
The selected model was Random Forest. On the held-out test set:
- MAE: ₹1,373.82
- RMSE: ₹2,926.31
- R²: 0.9834

These are dataset-specific offline results, not a guarantee of live fare accuracy. The dataset appears to contain repeated structured route/flight patterns, so high R² should be interpreted in that context. A random split can place similar records in both train and test; for stronger generalization testing, a time-based or route/flight-group split would be a useful extension.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Retrain
```bash
python src/train_model.py
```

The training script compares two compact models. It trains Random Forest on a reproducible subset for practical runtime and records all split sizes and metrics in `reports/`.

## Project contents
- `data/flight_price.csv`: cleaned main dataset (generated index removed)
- `data/source_files/`: original uploaded CSV files
- `notebooks/flight_price_prediction.ipynb`: notebook walkthrough
- `src/train_model.py`: reproducible model training script
- `models/best_model.joblib`: selected trained model bundle
- `reports/model_comparison.csv`: validation comparison
- `reports/metrics_summary.json`: final metrics and split sizes
- `reports/test_predictions.csv`: held-out predictions

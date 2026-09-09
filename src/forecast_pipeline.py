"""Leakage-safe training, validation, and recursive forecasting pipeline."""

from pathlib import Path
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import MODEL_DIR, RESULT_DIR, TARGET, DATE_COL, RANDOM_STATE


FEATURES = [
    "store", "item", "year", "month", "day", "day_of_week", "day_of_year",
    "week_of_year", "quarter", "is_weekend", "is_month_start", "is_month_end",
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
    "rolling_std_7", "rolling_std_14", "rolling_std_28",
]
LAGS = [1, 2, 3, 7, 14, 28]
WINDOWS = [7, 14, 28]


def _date_features(date):
    date = pd.Timestamp(date)
    return {
        "year": date.year, "month": date.month, "day": date.day,
        "day_of_week": date.dayofweek, "day_of_year": date.dayofyear,
        "week_of_year": int(date.isocalendar().week), "quarter": date.quarter,
        "is_weekend": int(date.dayofweek >= 5),
        "is_month_start": int(date.is_month_start), "is_month_end": int(date.is_month_end),
    }


def _feature_row(store, item, date, values):
    row = {"store": store, "item": item, **_date_features(date)}
    for lag in LAGS:
        row[f"lag_{lag}"] = values[-lag] if len(values) >= lag else np.nan
    for window in WINDOWS:
        recent = np.asarray(values[-window:], dtype=float)
        row[f"rolling_mean_{window}"] = recent.mean() if len(recent) else np.nan
        row[f"rolling_std_{window}"] = recent.std() if len(recent) > 1 else 0.0
    return row


def make_training_features(df: pd.DataFrame):
    """Build features from known history only; no future rows are consulted."""
    result = df.copy().sort_values(["store", "item", DATE_COL])
    dates = pd.to_datetime(result[DATE_COL])
    result["year"] = dates.dt.year
    result["month"] = dates.dt.month
    result["day"] = dates.dt.day
    result["day_of_week"] = dates.dt.dayofweek
    result["day_of_year"] = dates.dt.dayofyear
    result["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    result["quarter"] = dates.dt.quarter
    result["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
    result["is_month_start"] = dates.dt.is_month_start.astype(int)
    result["is_month_end"] = dates.dt.is_month_end.astype(int)
    grouped = result.groupby(["store", "item"], sort=False)[TARGET]
    for lag in LAGS:
        result[f"lag_{lag}"] = grouped.shift(lag)
    for window in WINDOWS:
        shifted = grouped.shift(1)
        result[f"rolling_mean_{window}"] = shifted.groupby([result["store"], result["item"]], sort=False).transform(lambda x: x.rolling(window, min_periods=1).mean())
        result[f"rolling_std_{window}"] = shifted.groupby([result["store"], result["item"]], sort=False).transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
    return result.dropna(subset=[f"lag_{max(LAGS)}"]).reset_index(drop=True)


def recursive_forecast(model, history: pd.DataFrame, future: pd.DataFrame):
    """Forecast future rows one date at a time using predictions as new history."""
    histories = {
        (store, item): group.sort_values(DATE_COL)[TARGET].astype(float).tolist()
        for (store, item), group in history.groupby(["store", "item"])
    }
    predictions = []
    for row in future.sort_values(DATE_COL).itertuples(index=False):
        key = (row.store, row.item)
        values = histories.setdefault(key, [])
        features = pd.DataFrame([_feature_row(row.store, row.item, row.date, values)])
        features = features.reindex(columns=FEATURES).fillna(0)
        prediction = max(0.0, float(model.predict(features)[0]))
        predictions.append((row.id if hasattr(row, "id") else len(predictions), row.date, row.store, row.item, prediction))
        values.append(prediction)
    return pd.DataFrame(predictions, columns=["id", DATE_COL, "store", "item", "prediction"])


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    nonzero = actual != 0
    return {
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "MAPE": float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100) if nonzero.any() else 0.0,
    }


def run_forecast(train: pd.DataFrame, test: pd.DataFrame, split_date="2017-10-01"):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    train = train.copy()
    test = test.copy()
    train[DATE_COL] = pd.to_datetime(train[DATE_COL])
    test[DATE_COL] = pd.to_datetime(test[DATE_COL])
    cutoff = pd.Timestamp(split_date)
    fit_history = train[train[DATE_COL] < cutoff].sort_values(DATE_COL)
    validation = train[train[DATE_COL] >= cutoff].sort_values(DATE_COL)
    fit_features = make_training_features(fit_history)
    model = lgb.LGBMRegressor(
        objective="regression", n_estimators=700, learning_rate=0.05,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
    )
    model.fit(fit_features[FEATURES], fit_features[TARGET])
    validation_with_ids = validation.assign(id=validation.index)
    validation_predictions = recursive_forecast(model, fit_history, validation_with_ids)
    actuals = validation_with_ids[["id", TARGET]].rename(columns={TARGET: "actual"})
    validation_predictions = validation_predictions.merge(actuals, on="id", how="left")
    validation_predictions["error"] = validation_predictions["actual"] - validation_predictions["prediction"]
    validation_metrics = metrics(validation_predictions["actual"], validation_predictions["prediction"])

    # Retrain on all labelled observations before producing the competition/test forecast.
    all_features = make_training_features(train)
    final_model = lgb.LGBMRegressor(
        objective="regression", n_estimators=700, learning_rate=0.05,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
    )
    final_model.fit(all_features[FEATURES], all_features[TARGET])
    test_predictions = recursive_forecast(final_model, train, test)
    test_predictions["store_name"] = test_predictions["store"].map(lambda value: f"Store {int(value)}")
    test_predictions["item_name"] = test_predictions["item"].map(lambda value: f"Item {int(value)}")
    validation_predictions["store_name"] = validation_predictions["store"].map(lambda value: f"Store {int(value)}")
    validation_predictions["item_name"] = validation_predictions["item"].map(lambda value: f"Item {int(value)}")
    test_predictions.to_csv(RESULT_DIR / "test_predictions.csv", index=False)
    validation_predictions.to_csv(RESULT_DIR / "validation_predictions.csv", index=False)
    pd.DataFrame([validation_metrics]).to_csv(RESULT_DIR / "forecast_metrics.csv", index=False)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_DIR / "lightgbm_forecaster.joblib")
    with open(RESULT_DIR / "run_summary.json", "w", encoding="utf-8") as file:
        json.dump({"split_date": split_date, "validation_rows": len(validation), "test_rows": len(test), "metrics": validation_metrics}, file, indent=2)
    return validation_metrics, validation_predictions, test_predictions

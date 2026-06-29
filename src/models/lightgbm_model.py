import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))
from config import RANDOM_STATE

import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


def train_lgb(X_train, y_train, X_val=None, y_val=None):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": RANDOM_STATE,
        "n_jobs": -1,
    }

    train_data = lgb.Dataset(X_train, y_train)
    valid_sets = [train_data]
    valid_names = ["train"]

    if X_val is not None and y_val is not None:
        val_data = lgb.Dataset(X_val, y_val, reference=train_data)
        valid_sets.append(val_data)
        valid_names.append("valid")

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )
    return model


def evaluate_lgb(model, X_val, y_val):
    preds = model.predict(X_val, num_iteration=model.best_iteration)
    rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
    mae = float(mean_absolute_error(y_val, preds))
    mape = float(np.mean(np.abs((y_val - preds) / (y_val + 1e-8)) * 100))
    return {"rmse": rmse, "mae": mae, "mape": mape}, preds

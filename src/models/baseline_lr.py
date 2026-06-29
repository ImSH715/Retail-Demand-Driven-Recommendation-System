import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))
from config import RANDOM_STATE

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


def train_lr(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_lr(model, X_val, y_val):
    preds = model.predict(X_val)
    rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
    mae = float(mean_absolute_error(y_val, preds))
    mape = float(np.mean(np.abs((y_val - preds) / (y_val + 1e-8)) * 100))
    return {"rmse": rmse, "mae": mae, "mape": mape}, preds

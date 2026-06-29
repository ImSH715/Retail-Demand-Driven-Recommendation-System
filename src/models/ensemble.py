import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def blend_predictions(preds_dict: dict[str, np.ndarray], weights: dict[str, float] = None) -> np.ndarray:
    if weights is None:
        n = len(preds_dict)
        weights = {k: 1.0 / n for k in preds_dict}

    weighted_sum = np.zeros_like(list(preds_dict.values())[0], dtype=float)
    for name, pred in preds_dict.items():
        weighted_sum += weights.get(name, 1.0 / len(preds_dict)) * pred

    return weighted_sum


def evaluate_ensemble(preds, y_true):
    rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
    mae = float(mean_absolute_error(y_true, preds))
    mape = float(np.mean(np.abs((y_true - preds) / (y_true + 1e-8)) * 100))
    return {"rmse": rmse, "mae": mae, "mape": mape}

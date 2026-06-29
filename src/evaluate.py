import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.append(str(Path(__file__).parents[1]))
from config import TARGET


def calc_metrics(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8)) * 100)),
    }


def print_comparison(results: dict[str, dict]):
    rows = []
    for name, metrics in results.items():
        rows.append({"Model": name, **metrics})
    df = pd.DataFrame(rows).set_index("Model")
    print("\n" + "=" * 60)
    print("Model Performance Comparison")
    print("=" * 60)
    print(df.to_string())
    print("=" * 60)
    return df

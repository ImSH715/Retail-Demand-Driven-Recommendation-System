import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))
from config import FIGURE_DIR, DATE_COL, TARGET

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    from prophet import Prophet
except ImportError:
    Prophet = None


def train_prophet(train: pd.DataFrame, forecast_days: int = 90):
    if Prophet is None:
        print("Prophet not installed. Skipping...")
        return None, None

    daily = train.groupby(DATE_COL)[TARGET].sum().reset_index()
    daily.columns = ["ds", "y"]

    split_idx = int(len(daily) * 0.8)
    train_df = daily.iloc[:split_idx]
    test_df = daily.iloc[split_idx:]

    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    model.fit(train_df)

    future = model.make_future_dataframe(periods=len(test_df))
    forecast = model.predict(future)
    preds = forecast.iloc[-len(test_df):]["yhat"].values
    actuals = test_df["y"].values

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig = model.plot(forecast)
    fig.savefig(FIGURE_DIR / "prophet_forecast.png", dpi=150, bbox_inches="tight")

    fig2 = model.plot_components(forecast)
    fig2.savefig(FIGURE_DIR / "prophet_components.png", dpi=150, bbox_inches="tight")

    rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
    mae = float(mean_absolute_error(actuals, preds))
    mape = float(np.mean(np.abs((actuals - preds) / (actuals + 1e-8)) * 100))

    return {"rmse": rmse, "mae": mae, "mape": mape}, preds

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).parents[1]))
from config import DATE_COL, STORE_COL, ITEM_COL, TARGET


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.to_datetime(df[DATE_COL])
    df["year"] = d.dt.year
    df["month"] = d.dt.month
    df["day"] = d.dt.day
    df["day_of_week"] = d.dt.dayofweek  # 0=Mon
    df["day_of_year"] = d.dt.dayofyear
    df["week_of_year"] = d.dt.isocalendar().week.astype(int)
    df["quarter"] = d.dt.quarter
    df["is_weekend"] = (d.dt.dayofweek >= 5).astype(int)
    df["is_month_start"] = d.dt.is_month_start.astype(int)
    df["is_month_end"] = d.dt.is_month_end.astype(int)
    return df


def add_lag_features(
    df: pd.DataFrame,
    group_cols: list,
    lags: list[int],
    target: str = TARGET,
) -> pd.DataFrame:
    df = df.copy().sort_values([*group_cols, DATE_COL])
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(group_cols)[target].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    group_cols: list,
    windows: list[int],
    target: str = TARGET,
) -> pd.DataFrame:
    df = df.copy().sort_values([*group_cols, DATE_COL])
    for w in windows:
        grouped = df.groupby(group_cols)[target]
        df[f"rolling_mean_{w}"] = grouped.transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"rolling_std_{w}"] = grouped.transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).std().fillna(0)
        )
        df[f"rolling_max_{w}"] = grouped.transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).max()
        )
        df[f"rolling_min_{w}"] = grouped.transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).min()
        )
    return df


def add_expanding_features(
    df: pd.DataFrame,
    group_cols: list,
    target: str = TARGET,
) -> pd.DataFrame:
    df = df.copy().sort_values([*group_cols, DATE_COL])
    grouped = df.groupby(group_cols)[target]
    df["expanding_mean"] = grouped.transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    return df


def add_store_item_stats(
    df: pd.DataFrame,
    target: str = TARGET,
) -> pd.DataFrame:
    stats = df.groupby([STORE_COL, ITEM_COL])[target].agg(
        store_item_mean="mean",
        store_item_std="std",
        store_item_max="max",
        store_item_min="min",
    ).reset_index()
    df = df.merge(stats, on=[STORE_COL, ITEM_COL], how="left")
    return df


def build_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    df = add_date_features(df)

    group_cols = [STORE_COL, ITEM_COL]

    if is_train:
        df = add_lag_features(df, group_cols, lags=[1, 2, 3, 7, 14, 28])
        df = add_rolling_features(df, group_cols, windows=[7, 14, 28])
        df = add_expanding_features(df, group_cols)
        df = df.dropna()
    else:
        df = add_lag_features(df, group_cols, lags=[1, 2, 3, 7, 14, 28])
        df = add_rolling_features(df, group_cols, windows=[7, 14, 28])
        df = add_expanding_features(df, group_cols)
        # For test, fill remaining NaN with 0
        df = df.fillna(0)

    return df.reset_index(drop=True)

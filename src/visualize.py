import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
from config import FIGURE_DIR, TARGET, DATE_COL, STORE_COL, ITEM_COL

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10


def plot_time_series(train: pd.DataFrame, store: int = 1, item: int = 1):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    subset = train[(train[STORE_COL] == store) & (train[ITEM_COL] == item)]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(subset[DATE_COL], subset[TARGET], linewidth=0.5)
    ax.set_title(f"Store {store} - Item {item} Daily Sales")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"ts_store{store}_item{item}.png", bbox_inches="tight")
    plt.close()


def plot_pred_vs_actual(y_true, y_pred, model_name: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=1)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{model_name}: Predicted vs Actual")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"pred_vs_actual_{model_name}.png", bbox_inches="tight")
    plt.close()


def plot_residuals(y_true, y_pred, model_name: str):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(residuals, bins=50, edgecolor="black")
    axes[0].set_title(f"{model_name}: Residual Distribution")
    axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Frequency")
    axes[1].scatter(y_pred, residuals, alpha=0.3, s=1)
    axes[1].axhline(y=0, color="r", linestyle="--")
    axes[1].set_title(f"{model_name}: Residuals vs Predicted")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Residual")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"residuals_{model_name}.png", bbox_inches="tight")
    plt.close()

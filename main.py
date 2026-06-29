"""
Store Item Demand Forecasting — End-to-End ML Pipeline

Models:
  1. Linear Regression (baseline)
  2. LightGBM (main)
  3. Prophet (aggregate trend/seasonality)
  4. Ensemble (weighted blend)
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from config import (
    RAW_DIR, PROCESSED_DIR, MODEL_DIR, FIGURE_DIR,
    TARGET, DATE_COL, STORE_COL, ITEM_COL,
    RANDOM_STATE, TEST_SIZE,
)
from src.download_data import download_raw
from src.preprocess import preprocess, load_raw, clean_data
from src.features import build_features
from src.models.baseline_lr import train_lr, evaluate_lr
from src.models.lightgbm_model import train_lgb, evaluate_lgb
from src.models.prophet_model import train_prophet
from src.models.ensemble import blend_predictions, evaluate_ensemble
from src.evaluate import calc_metrics, print_comparison
from src.shap_analysis import analyze_shap
from src.visualize import plot_time_series, plot_pred_vs_actual, plot_residuals

import warnings
warnings.filterwarnings("ignore")


def main():
    print("=" * 60)
    print("Store Item Demand Forecasting Pipeline")
    print("=" * 60)

    # 1. Download
    print("\n[1/5] Downloading data...")
    download_raw()
    train, test = preprocess()

    # 2. Feature engineering
    print("\n[2/5] Building features...")
    train_feat = build_features(train, is_train=True)
    test_feat = build_features(test, is_train=False)

    feature_cols = [c for c in train_feat.columns
                    if c not in [TARGET, DATE_COL, "id"]]

    X = train_feat[feature_cols]
    y = train_feat[TARGET]

    print(f"Features ({len(feature_cols)}): {feature_cols[:8]}...")

    # 3. Train/validation split (time-based)
    print("\n[3/5] Splitting data...")
    dates = train_feat[DATE_COL].unique()
    split_date = pd.to_datetime("2017-10-01")
    train_mask = train_feat[DATE_COL] < split_date
    val_mask = train_feat[DATE_COL] >= split_date

    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # 4. Train models
    print("\n[4/5] Training models...")
    results = {}
    predictions = {}

    # --- Linear Regression (baseline) ---
    print("  >> Linear Regression...")
    lr_model = train_lr(X_train, y_train)
    lr_metrics, lr_preds = evaluate_lr(lr_model, X_val, y_val)
    results["Linear Regression"] = lr_metrics
    predictions["Linear Regression"] = lr_preds
    print(f"     RMSE: {lr_metrics['rmse']:.2f}, MAE: {lr_metrics['mae']:.2f}")

    # --- LightGBM ---
    print("  >> LightGBM...")
    lgb_model = train_lgb(X_train, y_train, X_val, y_val)
    lgb_metrics, lgb_preds = evaluate_lgb(lgb_model, X_val, y_val)
    results["LightGBM"] = lgb_metrics
    predictions["LightGBM"] = lgb_preds
    print(f"     RMSE: {lgb_metrics['rmse']:.2f}, MAE: {lgb_metrics['mae']:.2f}")

    # --- Prophet (aggregate) ---
    print("  >> Prophet (aggregate daily sales)...")
    prophet_result, _ = train_prophet(train)
    if prophet_result:
        daily_avg = train.groupby("date")["sales"].sum().mean()
        results["Prophet (Aggregate)"] = prophet_result
        print(f"     RMSE: {prophet_result['rmse']:.2f}  (MAPE: {prophet_result['rmse']/daily_avg*100:.1f}%)")
        print(f"     Note: Prophet predicts total daily sales (avg={daily_avg:.0f}), others predict per-item sales")

    # --- Ensemble ---
    print("  >> Ensemble (LGBM + LR)...")
    ensemble_preds = blend_predictions(
        {"LightGBM": lgb_preds, "LR": lr_preds},
        weights={"LightGBM": 0.7, "LR": 0.3},
    )
    ensemble_metrics = evaluate_ensemble(ensemble_preds, y_val)
    results["Ensemble (LGBM+LR)"] = ensemble_metrics
    predictions["Ensemble"] = ensemble_preds
    print(f"     RMSE: {ensemble_metrics['rmse']:.2f}, MAE: {ensemble_metrics['mae']:.2f}")

    # 5. Evaluate & compare
    print("\n[5/5] Evaluation & Visualization...")
    comparison_df = print_comparison(results)

    # --- Visualizations ---
    print("\nSaving visualizations...")
    plot_time_series(train, store=1, item=1)
    plot_pred_vs_actual(y_val, lgb_preds, "LightGBM")
    plot_residuals(y_val, lgb_preds, "LightGBM")

    # --- SHAP ---
    print("Running SHAP analysis...")
    X_sample = X_val.sample(min(500, len(X_val)), random_state=RANDOM_STATE)
    analyze_shap(lgb_model, X_sample, feature_cols, output_prefix="lgbm")

    # --- Submission prediction ---
    print("\nGenerating test predictions...")
    X_test = test_feat[feature_cols]
    test_preds = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)

    submission = pd.DataFrame({
        "id": test["id"] if "id" in test.columns else range(len(test_preds)),
        "sales": np.maximum(0, test_preds),
    })
    sub_path = Path("outputs") / "submission.csv"
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved: {sub_path}")

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print(f"Figures saved in: {FIGURE_DIR}")
    print(f"Models saved in: {MODEL_DIR}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()

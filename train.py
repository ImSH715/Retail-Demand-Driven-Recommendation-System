"""Run all locally available model training and offline evaluation."""

import pandas as pd

from config import TRAIN_FILE, TEST_FILE
from src.forecast_pipeline import run_forecast
from src.recommender import evaluate_recommender


def main():
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    print("Training leakage-safe demand forecast...")
    metrics, _, _ = run_forecast(train, test)
    print("Demand validation:", metrics)
    print("Evaluating Instacart recommender...")
    print("Recommendation validation:", evaluate_recommender())


if __name__ == "__main__":
    main()

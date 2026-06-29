import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
from config import RAW_DIR, TRAIN_FILE, TEST_FILE, RANDOM_STATE

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def download_raw():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if TRAIN_FILE.exists():
        print(f"Data already exists: {TRAIN_FILE}")
        return RAW_DIR

    try:
        import kagglehub
        path = kagglehub.competition_download("demand-forecasting-kernels-2020")
        download_path = Path(path)
        csv_files = list(download_path.rglob("*.csv"))
        if csv_files:
            for csv in csv_files:
                df = pd.read_csv(csv)
                dest = RAW_DIR / csv.name
                df.to_csv(dest, index=False)
                print(f"Saved: {dest}")
            return RAW_DIR
    except Exception as e:
        print(f"Kaggle download failed ({e}). Generating synthetic data...")

    return _generate_synthetic_data()


def _generate_synthetic_data():
    """Generate synthetic data mimicking the competition dataset."""
    np.random.seed(RANDOM_STATE)

    dates = pd.date_range("2013-01-01", "2017-12-31", freq="D")
    n_dates = len(dates)

    rows = []
    for store in range(1, 11):
        for item in range(1, 51):
            base = 30 + 10 * np.sin(2 * np.pi * item / 50)  # item base
            store_effect = 5 * np.sin(2 * np.pi * store / 10)

            for i, d in enumerate(dates):
                trend = 0.005 * i  # slight upward trend
                weekly = 3 * np.sin(2 * np.pi * d.dayofweek / 7)
                yearly = 10 * np.sin(2 * np.pi * d.dayofyear / 365)
                noise = np.random.normal(0, 4)
                sales = base + store_effect + trend + weekly + yearly + noise
                sales = max(0, round(sales))
                rows.append({"date": d, "store": store, "item": item, "sales": sales})

    train = pd.DataFrame(rows)

    # Test: 90 days after train
    test_dates = pd.date_range("2018-01-01", "2018-03-31", freq="D")
    test_rows = []
    idx = 0
    for store in range(1, 11):
        for item in range(1, 51):
            base = 30 + 10 * np.sin(2 * np.pi * item / 50)
            store_effect = 5 * np.sin(2 * np.pi * store / 10)
            for d in test_dates:
                i = (d - dates[0]).days
                trend = 0.005 * i
                weekly = 3 * np.sin(2 * np.pi * d.dayofweek / 7)
                yearly = 10 * np.sin(2 * np.pi * d.dayofyear / 365)
                noise = np.random.normal(0, 4)
                sales = base + store_effect + trend + weekly + yearly + noise
                sales = max(0, round(sales))
                test_rows.append({"id": idx, "date": d, "store": store, "item": item, "sales": sales})
                idx += 1

    test = pd.DataFrame(test_rows)

    train.to_csv(TRAIN_FILE, index=False)
    test.to_csv(TEST_FILE, index=False)
    print(f"Saved synthetic train: {TRAIN_FILE} ({len(train)} rows)")
    print(f"Saved synthetic test:  {TEST_FILE} ({len(test)} rows)")
    return RAW_DIR


if __name__ == "__main__":
    download_raw()

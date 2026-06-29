import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
from config import TRAIN_FILE, TEST_FILE, PROCESSED_DIR, DATE_COL


def load_raw():
    train = pd.read_csv(TRAIN_FILE, parse_dates=[DATE_COL])
    test = pd.read_csv(TEST_FILE, parse_dates=[DATE_COL])
    return train, test


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["sales"] if "sales" in df.columns else ["id"])
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    # sales must be non-negative
    if "sales" in df.columns:
        df = df[df["sales"] >= 0]
    return df.reset_index(drop=True)


def preprocess():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train, test = load_raw()
    train = clean_data(train)
    test = clean_data(test)

    train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)

    print(f"Train: {train.shape}, {train[DATE_COL].min()} ~ {train[DATE_COL].max()}")
    print(f"Test:  {test.shape}, {test[DATE_COL].min()} ~ {test[DATE_COL].max()}")
    return train, test


if __name__ == "__main__":
    preprocess()

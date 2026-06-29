from pathlib import Path

ROOT = Path(__file__).parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
FIGURE_DIR = OUTPUT_DIR / "figures"

TRAIN_FILE = RAW_DIR / "train.csv"
TEST_FILE = RAW_DIR / "test.csv"

TARGET = "sales"
DATE_COL = "date"
STORE_COL = "store"
ITEM_COL = "item"

# Train: 2013-01-01 ~ 2017-12-31
# Test:  2018-01-01 ~ 2018-03-31 (91 days)
TRAIN_START = "2013-01-01"
TRAIN_END = "2017-12-31"
TEST_START = "2018-01-01"
TEST_END = "2018-03-31"

RANDOM_STATE = 42
TEST_SIZE = 0.2

KAGGLE_DATASET = "demand-forecasting-kernels-2020"

"""Idempotent CSV to PostgreSQL loader for the Instacart data."""

import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

# Allow both `python database/etl.py` and `python -m database.etl`.
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import INSTACART_RAW_DIR


def load_data(database_url=None, raw_dir=INSTACART_RAW_DIR, chunksize=100_000):
    database_url = database_url or os.getenv("DATABASE_URL", "postgresql+psycopg2://recsys:recsys123@localhost:5432/retail_db")
    engine = create_engine(database_url)
    schema = (Path(__file__).with_name("schema.sql")).read_text(encoding="utf-8")
    with engine.begin() as connection:
        for statement in schema.split(";"):
            if statement.strip():
                connection.execute(text(statement))
        for table in ["order_products", "orders", "users", "products", "departments", "aisles"]:
            connection.execute(text(f"TRUNCATE TABLE {table} CASCADE"))

    for name in ["aisles", "departments", "products"]:
        pd.read_csv(raw_dir / f"{name}.csv").to_sql(name, engine, if_exists="append", index=False, method="multi", chunksize=chunksize)
    orders = pd.read_csv(raw_dir / "orders.csv")
    orders[["user_id"]].drop_duplicates().to_sql("users", engine, if_exists="append", index=False, method="multi", chunksize=chunksize)
    orders.to_sql("orders", engine, if_exists="append", index=False, method="multi", chunksize=chunksize)
    for filename in ["order_products__prior.csv", "order_products__train.csv"]:
        pd.read_csv(raw_dir / filename).to_sql("order_products", engine, if_exists="append", index=False, method="multi", chunksize=chunksize)
    return engine


if __name__ == "__main__":
    load_data()

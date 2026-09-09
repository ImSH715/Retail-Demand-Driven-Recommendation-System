"""Train both neural recommender architectures on a reproducible sample."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import INSTACART_RAW_DIR, MODEL_DIR, RESULT_DIR
from src.models.neural_recommenders import NeuralCollaborativeFiltering, TwoTowerModel, train_model, save_checkpoint


def main(max_positive=150_000, negative_ratio=1):
    orders = pd.read_csv(INSTACART_RAW_DIR / "orders.csv", usecols=["order_id", "user_id", "eval_set"])
    interactions = pd.read_csv(INSTACART_RAW_DIR / "order_products__train.csv", usecols=["order_id", "product_id"])
    interactions = interactions.merge(orders[orders.eval_set == "train"], on="order_id")[["user_id", "product_id"]].drop_duplicates()
    interactions = interactions.sample(min(max_positive, len(interactions)), random_state=42)
    users = {str(value): index for index, value in enumerate(interactions.user_id.unique())}
    items = {str(value): index for index, value in enumerate(interactions.product_id.unique())}
    positive_users = interactions.user_id.map(lambda x: users[str(x)]).to_numpy()
    positive_items = interactions.product_id.map(lambda x: items[str(x)]).to_numpy()
    rng = np.random.default_rng(42)
    negative_users = positive_users.copy()
    negative_items = rng.integers(0, len(items), len(positive_items))
    train_users = np.concatenate([positive_users, negative_users])
    train_items = np.concatenate([positive_items, negative_items])
    labels = np.concatenate([np.ones(len(positive_users)), np.zeros(len(negative_users))])
    results = {}
    for name, model_cls, filename in [("ncf", NeuralCollaborativeFiltering, "ncf.pt"), ("two_tower", TwoTowerModel, "two_tower.pt")]:
        model = train_model(model_cls(len(users), len(items)), train_users, train_items, labels)
        save_checkpoint(model, MODEL_DIR / filename, users, items, name)
        with torch.no_grad():
            score = model(torch.tensor(train_users), torch.tensor(train_items)).numpy()
        results[name] = {"samples": len(labels), "positive_rate": float(labels.mean()), "training_auc_proxy": float(((score[labels == 1] > score[labels == 0][:len(score[labels == 1])]).mean()))}
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "neural_model_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

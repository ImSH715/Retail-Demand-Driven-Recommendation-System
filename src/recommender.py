"""Lightweight Instacart recommender and offline evaluation."""

from pathlib import Path
import json
import pandas as pd

from config import INSTACART_RAW_DIR, RESULT_DIR


def build_recommendations(user_id: int, top_k: int = 10):
    orders_path = INSTACART_RAW_DIR / "orders.csv"
    prior_path = INSTACART_RAW_DIR / "order_products__prior.csv"
    products_path = INSTACART_RAW_DIR / "products.csv"
    orders = pd.read_csv(orders_path, usecols=["order_id", "user_id", "eval_set"])
    user_orders = orders[(orders.user_id == user_id) & (orders.eval_set == "prior")]
    if user_orders.empty:
        return pd.DataFrame(columns=["product_id", "product_name", "score"])
    prior = pd.read_csv(prior_path, usecols=["order_id", "product_id", "reordered"])
    history = prior[prior.order_id.isin(user_orders.order_id)]
    scores = history.groupby("product_id").agg(score=("reordered", "sum"), purchases=("product_id", "size")).reset_index()
    scores["score"] = scores["score"] + scores["purchases"] * 0.25
    products = pd.read_csv(products_path, usecols=["product_id", "product_name"])
    return scores.merge(products, on="product_id").sort_values(["score", "purchases"], ascending=False).head(top_k)


def evaluate_recommender(sample_size=1000, top_k=10):
    orders = pd.read_csv(INSTACART_RAW_DIR / "orders.csv")
    labels = orders[orders.eval_set == "train"][["order_id", "user_id"]]
    prior_orders = orders[orders.eval_set == "prior"][["order_id", "user_id"]]
    products = pd.read_csv(INSTACART_RAW_DIR / "order_products__train.csv", usecols=["order_id", "product_id"])
    labels = labels.merge(products, on="order_id")
    users = labels.user_id.drop_duplicates().head(sample_size)
    label_map = labels[labels.user_id.isin(users)].groupby("user_id").product_id.apply(set)
    history = prior_orders[prior_orders.user_id.isin(users)]
    prior = pd.read_csv(INSTACART_RAW_DIR / "order_products__prior.csv", usecols=["order_id", "product_id"])
    history = history.merge(prior, on="order_id")
    counts = history.groupby(["user_id", "product_id"]).size().reset_index(name="count")
    hits = []
    for user_id, group in counts.groupby("user_id"):
        if user_id not in label_map:
            continue
        recommended = set(group.sort_values("count", ascending=False).head(top_k).product_id)
        truth = label_map[user_id]
        hits.append({"user_id": int(user_id), "precision_at_k": len(recommended & truth) / top_k, "recall_at_k": len(recommended & truth) / len(truth)})
    result = pd.DataFrame(hits)
    summary = {"users_evaluated": len(result), "precision_at_k": float(result.precision_at_k.mean()), "recall_at_k": float(result.recall_at_k.mean()), "k": top_k}
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(RESULT_DIR / "recommendation_user_metrics.csv", index=False)
    with open(RESULT_DIR / "recommendation_metrics.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return summary

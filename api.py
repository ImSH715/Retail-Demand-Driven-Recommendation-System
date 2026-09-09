"""FastAPI serving layer for recommendations, forecasts, and SQL analytics."""

import json
import os
from pathlib import Path

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from sqlalchemy import create_engine, text

from config import MODEL_DIR, RESULT_DIR, INSTACART_RAW_DIR
from src.models.neural_recommenders import NeuralCollaborativeFiltering, TwoTowerModel

app = FastAPI(title="Retail Intelligence API", version="1.0.0")
products_path = INSTACART_RAW_DIR / "products.csv"
PRODUCTS = pd.read_csv(products_path, usecols=["product_id", "product_name"]) if products_path.exists() else pd.DataFrame(columns=["product_id", "product_name"])
FORECASTS = pd.read_csv(RESULT_DIR / "test_predictions.csv") if (RESULT_DIR / "test_predictions.csv").exists() else pd.DataFrame()


@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!doctype html>
    <html lang="ko"><head><meta charset="utf-8"><title>Retail Intelligence</title>
    <style>body{font-family:Arial,sans-serif;max-width:820px;margin:48px auto;padding:0 24px;color:#17202a;background:#f6f8fb}
    .card{background:white;border-radius:16px;padding:28px;margin:16px 0;box-shadow:0 4px 18px #17202a14}
    a{color:#1769aa;text-decoration:none}a:hover{text-decoration:underline}code{background:#eef2f6;padding:3px 6px;border-radius:5px}
    li{margin:12px 0}</style></head><body>
    <h1>Retail Intelligence API</h1>
    <p>수요 예측과 개인화 상품 추천을 확인하는 서비스입니다.</p>
    <div class="card"><h2>처음 사용한다면</h2><ol>
    <li><a href="/docs">Swagger 문서 열기</a>에서 API 목록을 확인합니다.</li>
    <li><a href="/health">시스템 상태</a>에서 모델이 준비됐는지 확인합니다.</li>
    <li><a href="/recommend/1?model=two_tower&top_k=5">사용자 1번 추천</a>을 눌러 상품 추천을 확인합니다.</li>
    <li><a href="/forecast?store=1&item=1&limit=20">매장 1·상품 1 수요 예측</a>을 확인합니다.</li>
    </ol></div>
    <div class="card"><h2>화면으로 보기</h2><p>시각화 대시보드는 별도 Streamlit 서버에서 실행합니다: <code>streamlit run app.py</code></p>
    <p>기본 주소: <a href="http://localhost:8501">http://localhost:8501</a></p></div>
    </body></html>
    """


def load_neural(name):
    filename = "ncf.pt" if name == "ncf" else "two_tower.pt"
    checkpoint_path = MODEL_DIR / filename
    metadata_path = checkpoint_path.with_suffix(".json")
    if not checkpoint_path.exists() or not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cls = NeuralCollaborativeFiltering if name == "ncf" else TwoTowerModel
    model = cls(checkpoint_path and len(metadata["user_map"]), len(metadata["item_map"]))
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True)["state_dict"])
    model.eval()
    return model, metadata


@app.get("/health")
def health():
    return {"status": "ok", "ncf_available": load_neural("ncf") is not None, "two_tower_available": load_neural("two_tower") is not None}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, model: str = Query("two_tower", pattern="^(ncf|two_tower)$"), top_k: int = Query(10, ge=1, le=100)):
    loaded = load_neural(model)
    if loaded is None:
        raise HTTPException(status_code=503, detail="Train the neural recommenders first: python train_recommenders.py")
    network, metadata = loaded
    user_key = str(user_id)
    if user_key not in metadata["user_map"]:
        raise HTTPException(status_code=404, detail="User is not present in the trained interaction sample")
    user_index = metadata["user_map"][user_key]
    item_ids = list(metadata["item_map"].keys())
    item_indices = torch.arange(len(item_ids))
    users = torch.full((len(item_ids),), user_index, dtype=torch.long)
    with torch.no_grad():
        scores = network(users, item_indices).numpy()
    best = scores.argsort()[-top_k:][::-1]
    result = pd.DataFrame({"product_id": [int(item_ids[i]) for i in best], "score": [float(scores[i]) for i in best]})
    return {"user_id": user_id, "model": model, "recommendations": result.merge(PRODUCTS, on="product_id").to_dict("records")}


@app.get("/forecast")
def forecast(store: int | None = None, item: int | None = None, limit: int = Query(100, ge=1, le=1000)):
    if FORECASTS.empty:
        raise HTTPException(status_code=404, detail="Forecast results are not available")
    result = FORECASTS.copy()
    if store is not None:
        result = result[result.store == store]
    if item is not None:
        result = result[result.item == item]
    result["store_name"] = result["store"].map(lambda value: f"Store {int(value)}")
    result["item_name"] = result["item"].map(lambda value: f"Item {int(value)}")
    return {"rows": result.head(limit).to_dict("records")}


@app.get("/analytics/top-products")
def top_products(limit: int = Query(10, ge=1, le=100)):
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise HTTPException(status_code=503, detail="DATABASE_URL is not configured")
    engine = create_engine(database_url)
    query = text("""
        SELECT p.product_id, p.product_name, COUNT(*) AS purchase_count,
               RANK() OVER (ORDER BY COUNT(*) DESC) AS overall_rank
        FROM order_products op JOIN products p ON p.product_id = op.product_id
        GROUP BY p.product_id, p.product_name
        ORDER BY purchase_count DESC LIMIT :limit
    """)
    with engine.connect() as connection:
        return {"rows": [dict(row._mapping) for row in connection.execute(query, {"limit": limit})]}

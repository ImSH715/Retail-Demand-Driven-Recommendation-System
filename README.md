# Retail Intelligence Lab

Instacart 구매 이력과 매장·상품 판매량 데이터를 이용한 수요 예측 및 상품 추천 프로젝트입니다.

## 현재 구현된 기능

- 과거 데이터만 사용하는 누수 방지형 수요 예측
- LightGBM 모델 학습 및 시간순 validation
- RMSE, MAE, MAPE 평가와 예측 결과 CSV 저장
- Instacart prior 주문 기반 사용자별 상품 추천
- Precision@10, Recall@10 오프라인 추천 평가
- Streamlit 대시보드에서 예측·실제값·추천 결과 시각화
- Neural Collaborative Filtering 및 Two-Tower 추천 모델
- FastAPI 추천·예측·SQL 분석 API
- PostgreSQL 스키마, ETL, SQL Window Function 분석
- Docker Compose 기반 PostgreSQL/API 실행 환경
- 가격 이력이 없는 현재 데이터의 한계를 화면에 명시

## 실행 방법

프로젝트 폴더에서 다음 명령을 실행합니다.

```bash
pip install -r requirements.txt
python train.py
python train_recommenders.py
streamlit run app.py
uvicorn api:app --reload
```

처음 사용하는 경우에는 다음 순서로 확인합니다.

1. API를 실행하고 `http://localhost:8000/`에 접속합니다.
2. 안내 페이지에서 `Swagger 문서`를 열어 API를 직접 실행합니다.
3. `http://localhost:8000/health`에서 모델 상태를 확인합니다.
4. `http://localhost:8501`에서 그래프와 평가 결과를 확인합니다.

브라우저에서 표시되는 Streamlit 주소로 접속합니다.

## 결과 파일

- `outputs/results/forecast_metrics.csv`: 수요 예측 validation 지표
- `outputs/results/validation_predictions.csv`: 실제값·예측값·오차
- `outputs/results/test_predictions.csv`: 정답이 없는 test 데이터 예측
- `outputs/results/recommendation_metrics.json`: 추천 평가 지표
- `outputs/models/lightgbm_forecaster.joblib`: 최종 학습 모델
- `outputs/models/ncf.pt`: Neural Collaborative Filtering 모델
- `outputs/models/two_tower.pt`: Two-Tower 모델
- `outputs/results/neural_model_metrics.json`: 신경망 모델 학습 결과

## API

- `GET /health`
- `GET /recommend/{user_id}?model=ncf&top_k=10`
- `GET /forecast?store=1&item=1`
- `GET /analytics/top-products?limit=10`

PostgreSQL 적재는 다음처럼 실행합니다.

```bash
python database/etl.py
```

SQL Window Function 분석은 `database/analytics.sql`에 있습니다.

Docker 환경에서는 다음 명령으로 PostgreSQL과 API를 시작합니다.

```bash
docker compose up --build
```

Docker Compose를 사용하면 다음 주소가 제공됩니다.

- API 안내: `http://localhost:8000/`
- API 문서: `http://localhost:8000/docs`
- 시각화 대시보드: `http://localhost:8501`

## 검증 방식

수요 예측은 2017-10-01 이전 데이터로 학습하고 이후 기간을 미래처럼 예측합니다. test 데이터의 정답이 공개되지 않은 경우에는 test 성능을 계산하지 않고 예측 파일만 생성합니다.

추천 모델은 Instacart `prior` 주문으로 추천한 뒤, `train` 주문의 실제 상품을 정답으로 비교합니다.

## 가격 예측

현재 Instacart 파일에는 날짜별 가격 이력이 없기 때문에 가격 상승·하락 모델은 아직 학습하지 않습니다. 가격 예측을 추가하려면 상품별 날짜, 가격, 할인율, 매장, 프로모션 이력이 필요합니다.

수요 예측용 현재 데이터셋은 상품명이 없는 store/item 시계열 데이터이므로 화면에는 `Store 1`, `Item 1`처럼 표시됩니다. 실제 상품명을 보여주려면 상품 ID와 상품 카탈로그를 연결해야 하며, Instacart 추천 결과에는 실제 `product_name`이 표시됩니다.

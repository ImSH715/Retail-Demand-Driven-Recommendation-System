"""Interactive dashboard for model verification and recommendations."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
RESULT_DIR = ROOT / "outputs" / "results"

st.set_page_config(page_title="Retail Intelligence Lab", page_icon="", layout="wide")
st.title("Retail Intelligence Lab")
st.caption("수요 예측 검증과 Instacart 상품 추천을 한 화면에서 확인합니다.")

metrics_file = RESULT_DIR / "forecast_metrics.csv"
pred_file = RESULT_DIR / "validation_predictions.csv"
test_file = RESULT_DIR / "test_predictions.csv"
rec_file = RESULT_DIR / "recommendation_metrics.json"

if not metrics_file.exists() or not pred_file.exists():
    st.warning("학습 결과가 없습니다. 프로젝트 폴더에서 `python train.py`를 먼저 실행하세요.")
    st.stop()

forecast_metrics = pd.read_csv(metrics_file).iloc[0].to_dict()
validation = pd.read_csv(pred_file, parse_dates=["date"])
test_predictions = pd.read_csv(test_file, parse_dates=["date"]) if test_file.exists() else pd.DataFrame()

tab_forecast, tab_recommend, tab_price, tab_about = st.tabs(["수요 예측 검증", "상품 추천", "가격 예측", "실행 정보"])

with tab_forecast:
    st.subheader("미래 데이터를 흉내 낸 시간순 검증")
    st.caption("검증 구간은 학습에 사용하지 않고, 과거 판매량만으로 한 날짜씩 예측했습니다.")
    cols = st.columns(3)
    cols[0].metric("RMSE", f"{forecast_metrics['RMSE']:.2f}")
    cols[1].metric("MAE", f"{forecast_metrics['MAE']:.2f}")
    cols[2].metric("MAPE", f"{forecast_metrics['MAPE']:.2f}%")
    st.line_chart(validation.groupby("date")[["actual", "prediction"]].mean())
    st.subheader("매장·상품별 확인")
    stores = sorted(validation.store.unique())
    items = sorted(validation.item.unique())
    selected_store = st.selectbox("매장", stores, format_func=lambda value: f"Store {value}")
    selected_item = st.selectbox("상품", items, format_func=lambda value: f"Item {value}")
    selected = validation[(validation.store == selected_store) & (validation.item == selected_item)].set_index("date")
    st.line_chart(selected[["actual", "prediction"]])
    display_columns = [column for column in ["date", "store_name", "item_name", "actual", "prediction", "error"] if column in selected.reset_index().columns]
    st.dataframe(selected.reset_index()[display_columns].tail(30), use_container_width=True)
    st.download_button("검증 결과 CSV 다운로드", validation.to_csv(index=False), "validation_predictions.csv", "text/csv")

with tab_recommend:
    st.subheader("사용자별 상품 추천")
    st.caption("Instacart prior 주문 이력으로 추천하고, train 주문을 정답으로 Precision@K와 Recall@K를 계산합니다.")
    if rec_file.exists():
        rec_metrics = json.loads(rec_file.read_text(encoding="utf-8"))
        cols = st.columns(3)
        cols[0].metric("평가 사용자", f"{rec_metrics['users_evaluated']:,}")
        cols[1].metric("Precision@10", f"{rec_metrics['precision_at_k']:.3f}")
        cols[2].metric("Recall@10", f"{rec_metrics['recall_at_k']:.3f}")
    user_id = st.number_input("사용자 ID", min_value=1, value=1, step=1)
    if st.button("추천 조회"):
        from src.recommender import build_recommendations
        recommendations = build_recommendations(int(user_id), 10)
        if recommendations.empty:
            st.info("해당 사용자의 prior 구매 이력을 찾지 못했습니다.")
        else:
            st.dataframe(recommendations, use_container_width=True, hide_index=True)

with tab_price:
    st.subheader("가격 변동 예측")
    st.info("현재 보유한 Instacart 데이터에는 날짜별 실제 가격 이력이 없습니다. 가격 데이터가 추가되면 동일한 시간순 검증 구조로 가격 상승·하락과 예상 가격을 학습할 수 있습니다.")
    st.write("필요 데이터: 상품 ID, 날짜, 가격, 할인율, 매장, 프로모션, 재고 상태")

with tab_about:
    st.subheader("검증 결과의 의미")
    st.write("수요 예측 test.csv는 제출용 정답이 공개되지 않은 경우가 있어, train의 마지막 기간을 별도 validation으로 분리해 성능을 측정합니다.")
    st.write("test 예측은 실제 정답이 공개될 때까지 성능을 확정할 수 없으며, 현재 화면에서는 생성된 미래 예측값과 다운로드 파일을 확인할 수 있습니다.")
    if not test_predictions.empty:
        st.metric("생성된 test 예측 행", f"{len(test_predictions):,}")
        st.dataframe(test_predictions.head(20), use_container_width=True, hide_index=True)

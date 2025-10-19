 # Streamlit 메인 (대시보드/업로드/리포트/챗봇)
import os, sys
import time
import streamlit as st
import pandas as pd
from utils.io_schemas import CARD_COLS, MANUAL_COLS, TRAVEL_TITLES_COLS
from utils.settings import NAVER_CLIENT_ID
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
from services.reconcile import reconcile, ReconcileConfig
from services.rag import build_index, rag_answer
from services.auto_fill import auto_fill_card_rows, auto_defaults_for_non_corp, auto_title
from services.travel_report_check import mark_missing_reports
from services.naver_maps import geocode, static_map, route_summary


st.set_page_config(page_title="출장비용 자동기입/검증", layout="wide")

if "applicant_name" not in st.session_state:
    st.session_state.applicant_name = None

if not st.session_state.applicant_name:
    st.title("🔐 로그인 / 사용자 등록")
    name_in = st.text_input("신청자 이름을 입력하세요 (예: 김도희)")
    if st.button("확인", key="login_confirm"):
        if name_in.strip():
            st.session_state.applicant_name = name_in.strip()
            st.success(f"안녕하세요, {st.session_state.applicant_name}님! 👋")
            st.rerun()
        else:
            st.warning("이름을 입력하세요.")
    st.stop()

APPLICANT_NAME = st.session_state.applicant_name

st.title("출장비용지급품의 보조 도구 (MVP)")

with st.sidebar:
    st.markdown(f"**신청자:** {APPLICANT_NAME}")
    st.caption("※ CSV는 UTF-8 권장 / 컬럼명 표준은 README 참고")

tab1, tab2, tab3 = st.tabs([
    "① 데이터 업로드·자동기입·검증",
    "② 경로검색(네이버지도)",
    "③ 카드/결의 대사 결과"       # ← 추가
])

# ---------------- Tab1 ----------------
with tab1:
    st.subheader("1) CSV 업로드")
    c1, c2, c3 = st.columns(3)
    with c1:
        f_card = st.file_uploader("카드내역 CSV(법인카드 포함)", type=["csv"], key="card")
    with c2:
        f_manual = st.file_uploader("수기 입력 CSV(비법인/개인카드 등)", type=["csv"], key="manual")
    with c3:
        f_titles = st.file_uploader("기존 국내출장 보고서 제목 CSV(title 한 컬럼)", type=["csv"], key="titles")

    if st.button("자동기입 실행", type="primary"):
        if not f_card and not f_manual:
            st.warning("최소 한 개의 거래 CSV가 필요합니다.")
        else:
            dfs = []
            if f_card:
                dfc = pd.read_csv(f_card)
                missing_cols = set(CARD_COLS) - set(dfc.columns)
                if missing_cols:
                    st.error(f"카드 CSV 컬럼 부족: {missing_cols}")
                    st.stop()
                dfc = auto_fill_card_rows(dfc)
                dfs.append(dfc)

            if f_manual:
                dfm = pd.read_csv(f_manual)
                missing_cols = set(MANUAL_COLS) - set(dfm.columns)
                if missing_cols:
                    st.error(f"수기 CSV 컬럼 부족: {missing_cols}")
                    st.stop()
                dfm = auto_defaults_for_non_corp(dfm)
                dfs.append(dfm)

            df_all = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

            # 제목 자동 생성
            if not df_all.empty:
                month = str(df_all['report_month'].iloc[0])
                auto_t = auto_title(month, APPLICANT_NAME)
                st.success(f"제목 자동 생성: **{auto_t}**")

            # 보고서 누락 체크
            if f_titles and not df_all.empty:
                titles_df = pd.read_csv(f_titles)
                if 'title' not in titles_df.columns: st.error("제목 CSV에는 'title' 컬럼이 필요합니다."); st.stop()
                df_all = mark_missing_reports(df_all, titles_df)
                miss_n = int(df_all['missing_report'].sum())
                if miss_n > 0:
                    st.toast("해당출장지출에 대한 출장보고서가 누락되었습니다!", icon="⚠️")
                    st.info(f"보고서 누락 건수: {miss_n}건 (표의 'missing_report' 열 확인)")
            st.divider()
            st.subheader("자동기입 결과 미리보기")
            st.dataframe(df_all, use_container_width=True, height=480)
            st.download_button("CSV 다운로드", df_all.to_csv(index=False).encode("utf-8-sig"),
                               file_name="auto_filled_expenses.csv")

# ---------------- Tab2 ----------------
with tab2:
    st.subheader("경로 검색(네이버 지도)")
    if not NAVER_CLIENT_ID:
        st.warning("네이버 지도 API 키(NAVER_CLIENT_ID/SECRET)가 필요합니다. .env 설정 후 재시작하세요.")
    colA, colB = st.columns(2)
    with colA:
        origin = st.text_input("출발지", placeholder="예: 서울역")
    with colB:
        dest = st.text_input("도착지", placeholder="예: 대전역")
    if st.button("검색", key="search_route"):
        if not (origin and dest):
            st.error("출발지/도착지를 모두 입력하세요.")
        else:
            o = geocode(origin); d = geocode(dest)
            if not (o and d):
                st.error("지오코딩 실패(장소명을 바꿔보세요).")
            else:
                summ = route_summary(o, d)
                img = static_map([o, d])
                if summ:
                    st.success(f"예상 거리: {summ['distance_km']} km / 예상 소요: {summ['duration_min']} 분")
                if img:
                    st.image(img, caption="네이버 정적지도(핀: 출발=red, 도착=blue)", use_column_width=True)
                    st.download_button("이미지 저장", data=img.getvalue(), file_name="route_map.png", mime="image/png")
                else:
                    st.info("이미지 생성은 키가 필요하거나 일시적 오류일 수 있어요.")

with tab3:
    st.subheader("카드 vs 결의서 대사 결과")

    from services.reconcile import reconcile, ReconcileConfig
    from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

    # ✅ 업로드/대사 실행 부분
    c1, c2 = st.columns(2)
    with c1:
        f_card = st.file_uploader("법인카드 내역 CSV", type=["csv"], key="recon_card")
    with c2:
        f_expense = st.file_uploader("지출결의서 CSV", type=["csv"], key="recon_expense")

    if st.button("대사 실행", type="primary", key="reconcile_run"):
        if not (f_card and f_expense):
            st.warning("두 CSV 모두 업로드해야 합니다.")
            st.stop()

        dfc = pd.read_csv(f_card)
        dfm = pd.read_csv(f_expense)
        cfg = ReconcileConfig(amount_tol=100, date_tol_days=0, merchant_threshold=70)
        res = reconcile(dfc, dfm, cfg)

        st.json(res["summary"])

        # ✅ 바로 아래에 show_grid() 함수 붙이기
        def show_grid(df, title, highlight_condition_js=None, height=300):
            st.markdown(f"**{title}**")
            if df is None or df.empty:
                st.info("데이터가 없습니다.")
                return
            gb = GridOptionsBuilder.from_dataframe(df)
            gb.configure_default_column(resizable=True, filter=True, sortable=True)
            gb.configure_side_bar()
            if highlight_condition_js:
                getRowStyle = JsCode(f"""
                function(params) {{
                    {highlight_condition_js}
                }}
                """)
                gb.configure_grid_options(getRowStyle=getRowStyle)
            grid_options = gb.build()
            AgGrid(
                df,
                gridOptions=grid_options,
                height=height,
                theme="balham",
                fit_columns_on_grid_load=True,
            )

        # ✅ 대사 결과 3종 표시
        show_grid(res["matches"], "매칭 성공 내역", height=260)
        show_grid(
            res["unmatched_card"],
            "카드에는 있고 결의서에는 없는 내역 (미매칭)",
            highlight_condition_js="""
                if (params.data.reason && params.data.reason.length > 0) {
                    return { background: '#ffe6e6', color: '#7a0000', fontWeight: '600' };
                }
                return null;
            """,
            height=260
        )
        show_grid(
            res["unmatched_expense"],
            "결의서에는 있고 카드에는 없는 내역 (미매칭)",
            highlight_condition_js="""
                if (params.data.reason && params.data.reason.length > 0) {
                    return { background: '#ffe6e6', color: '#7a0000', fontWeight: '600' };
                }
                return null;
            """,
            height=260
        )

#RAG Expander
with st.expander("FAQ / 규정 검색 (RAG)"):
    q = st.text_input("사내 규정/FAQ 질문", key="rag_q")
    if st.button("검색", key="search_rag"):
        if not q.strip():
            st.warning("질문을 입력하세요.")
        else:
            try: build_index()
            except Exception as e: st.info(f"인덱스 준비 중: {e}")
            a, refs = rag_answer(q)
            st.markdown(a)
            if refs: st.caption("참고 소스: " + ", ".join(m.get("path","") for m in refs if isinstance(m,dict)))

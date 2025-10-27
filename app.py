 # Streamlit 메인 (대시보드/업로드/리포트/챗봇)
import os, sys
import time
import streamlit as st
import pandas as pd
from services.autofill.autofill_service import process_html
from utils.io_schemas import CARD_COLS, MANUAL_COLS, TRAVEL_TITLES_COLS
from utils.settings import NAVER_CLIENT_ID
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
from services.reconcile import reconcile, ReconcileConfig
from services.rag import build_index, rag_answer
from services.auto_fill import auto_fill_card_rows, auto_defaults_for_non_corp, auto_title
from services.travel_report_check import mark_missing_reports
from services.naver_maps import geocode, static_map, route_summary
from services.autofill.ocr_attendance import extract_times_from_image
from services.autofill.autofill_service import process_html



st.set_page_config(page_title="출장비용 자동기입/검증", layout="wide")

# [S] 초기 사용자이름 입력 페이지===========
# if "applicant_name" not in st.session_state:
#     st.session_state.applicant_name = None

# if not st.session_state.applicant_name:
#     st.title("🔐 로그인 / 사용자 등록")
#     name_in = st.text_input("신청자 이름을 입력하세요 (예: 김슈어)")
#     if st.button("확인", key="login_confirm"):
#         if name_in.strip():
#             st.session_state.applicant_name = name_in.strip()
#             st.success(f"안녕하세요, {st.session_state.applicant_name}님! 👋")
#             st.rerun()
#         else:
#             st.warning("이름을 입력하세요.")
#     st.stop()

# APPLICANT_NAME = st.session_state.applicant_name

# [E] 초기 사용자이름 입력 페이지===========
st.title("법인카드 사용 내역 자동점검 및 레포팅 시스템")


### 🅱️메인화면 사이드 바-----
with st.sidebar:
    #st.markdown(f"**신청자:** {APPLICANT_NAME}")      #초기 사용자이름 입력
    st.caption("※ CSV는 UTF-8 권장 / 컬럼명 표준은 README 참고")



### 🦖메인화면 탭-----
tab1, tab2, tab3, tab4 = st.tabs([
    "① 근태 스크린샷 → 상세내용/공급가액 자동기입",
    "② 경로검색(네이버지도)",
    "③ 카드/결의 대사 결과",
    "④ 사용기록 대시보드",
])


# === 근태 스크린샷 → 지출결의서 HTML 자동반영 ===
st.divider()
st.subheader("1) 근태 스크린샷 → 상세내용/공급가액 자동기입 (HTML에 반영)")

# 기본 HTML 양식 설정 (세션에 저장)
with st.expander("📄 기본 HTML 양식 설정 (최초 1회)", expanded=("base_html" not in st.session_state)):
    base_html_upload = st.file_uploader("기본 지출결의서 HTML 템플릿 업로드", type=["html","htm"], key="base_html_upload")
    if base_html_upload:
        st.session_state.base_html = base_html_upload.read()
        st.success(f"✅ 기본 양식 저장 완료 ({len(st.session_state.base_html)} bytes)")

    if "base_html" in st.session_state:
        st.info(f"현재 기본 양식: {len(st.session_state.base_html)} bytes 저장됨")
        if st.button("기본 양식 초기화", key="reset_base_html"):
            del st.session_state.base_html
            st.rerun()

# 근태 스크린샷 업로드
att_img = st.file_uploader("근태현황 스크린샷 업로드 (PNG/JPG)", type=["png","jpg","jpeg"], key="att_img")

debug_view = st.checkbox("🔎 디버그 보기(ROI/색마스크/라인)", value=True)

if st.button("이미지 인식 → HTML 자동반영", key="run_img2html", type="primary"):
    if not att_img:
        st.warning("근태 이미지를 업로드하세요.")
        st.stop()

    if "base_html" not in st.session_state:
        st.error("먼저 기본 HTML 양식을 설정해주세요.")
        st.stop()

    try:
        # 1 이미지 OCR로 시간 추출
        img_bytes = att_img.read()
        if debug_view:
            times, dbg = extract_times_from_image(img_bytes, return_debug=True)
            st.success(f"인식 결과 → 총:{times['total_hm']} / 선택:{times['select_hm']} / 초과:{times['extend_hm']}")

            c1, c2 = st.columns(2)

            # 편의 헬퍼 (버전 호환)
            def show(img, caption):
                st.image(img, caption=caption, use_column_width=True)

            with c1:
                show(dbg["orig"], "원본")
                # 마스크는 0~255 그레이스케일(np.uint8)
                show(dbg["green_mask"], "녹색 마스크(선택근무 점)")

            with c2:
                show(dbg["overlay"], "오버레이(ROI/라인/읽은박스)")
                show(dbg["red_mask"], "빨강 마스크(초과근무 점)")
        else:
            times = extract_times_from_image(img_bytes)

        # 2 HTML에 반영 (상세내용·공급가액)
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_in:
            tmp_in.write(st.session_state.base_html)  # ← 여기!
            tmp_in.flush()
            in_path = tmp_in.name


        out_fd, out_path = tempfile.mkstemp(suffix=".html")
        os.close(out_fd)

        result = process_html(in_path, out_path, override=times)
        st.json(result)  # {'총근무','선택근무','초과근무','공급가액','상세내용'}

        # 3 수정된 HTML 다운로드 제공
        with open(out_path, "rb") as f:
            updated = f.read()
        st.download_button(
            "수정된 지출결의서 HTML 다운로드",
            data=updated, file_name="updated_expenditure.html", mime="text/html"
        )

        with st.expander("수정된 HTML 일부 미리보기"):
            st.code(updated.decode("utf-8", errors="ignore")[:4000], language="html")

    except Exception as e:
        st.error(f"자동 반영 중 오류: {e}")
    finally:
        try: os.remove(in_path)
        except: pass
        try: os.remove(out_path)
        except: pass








# --Tab2 2.경로검색(네이버지도) ----------------
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





# --Tab3. 카드 vs 결의서 대사 결과) ----------------
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



# ---------------- Tab1: 지출결의서 HTML 자동기입 ----------------
st.divider()
st.subheader("2) 지출결의서 HTML 자동기입 (상세내용·공급가액)")

html_file = st.file_uploader("지출결의서 HTML 업로드 (export된 .html)", type=["html", "htm"], key="expense_html2")

col_run1, col_run2 = st.columns([1, 3])
with col_run1:
    run_html_btn = st.button("HTML 자동기입 실행", key="run_autofill_html", type="primary")

if run_html_btn:
    if not html_file:
        st.warning("지출결의서 HTML 파일을 업로드하세요.")
        st.stop()

    # 업로드 파일을 임시 경로에 저장 후 처리
    import tempfile, os

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_in:
        tmp_in.write(html_file.read())
        tmp_in.flush()
        in_path = tmp_in.name

    out_fd, out_path = tempfile.mkstemp(suffix=".html")
    os.close(out_fd)

    try:
        result = process_html(in_path, out_path)  # ✅ 상세내용/공급가액 자동 반영
        st.success("HTML 자동기입이 완료되었습니다.")
        st.json(result)  # {'총근무', '선택근무', '초과근무', '공급가액', '상세내용'}

        # 수정된 HTML 읽어서 다운로드 버튼 제공
        with open(out_path, "rb") as f:
            updated_bytes = f.read()

        st.download_button(
            "수정된 지출결의서 HTML 다운로드",
            data=updated_bytes,
            file_name="updated_expenditure.html",
            mime="text/html",
        )

        # 간단 미리보기(선택)
        with st.expander("수정된 HTML 텍스트 미리보기"):
            st.code(updated_bytes.decode("utf-8", errors="ignore")[:5000], language="html")

    except Exception as e:
        st.error(f"자동기입 중 오류가 발생했습니다: {e}")
    finally:
        # 임시파일 정리
        try:
            os.remove(in_path)
        except Exception:
            pass
        try:
            os.remove(out_path)
        except Exception:
            pass
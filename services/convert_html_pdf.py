# -*- coding: utf-8 -*-
"""
convert_html_pdf.py
- Streamlit app: convert 지출결의서 HTML -> CSV
- Saves CSV to data/vectorstore/converted as: "{사용자}_지출결의서_converted.csv"
"""

import io
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

# ---------------------------
# Config
# ---------------------------
OPTDIR = Path("data/vectorstore/converted")
OPTDIR.mkdir(parents=True, exist_ok=True)

CSV_HEADERS = [
    "기본적요", "증빙유형", "증빙일자", "승인시간", "사용카드",
    "프로젝트", "거래처", "업무용차량", "공급가액", "부가세", "합계", "상세내용",
    "지급요청일", "결의금액", "부서", "사용자",
]


# ---------------------------
# Helpers
# ---------------------------
def _txt(el) -> str:
    return el.get_text(strip=True) if el else ""


def decode_bytes(b: bytes) -> str:
    """Try UTF-8 first, fall back to CP949."""
    for enc in ("utf-8", "cp949", "euc-kr"):
        try:
            return b.decode(enc)
        except Exception:
            continue
    # last resort
    return b.decode("utf-8", errors="ignore")


def parse_general_info(soup: BeautifulSoup) -> Dict[str, str]:
    """
    Extract 공통 정보: 사용자(user), 부서(dept), 결의금액(amount), 지급요청일(req_date)
    The layout may vary slightly, so we try by class names first then by labels.
    """
    info = {"사용자": "", "부서": "", "결의금액": "", "지급요청일": ""}

    # Try by classes seen in examples
    # 사용자
    el = soup.select_one("td.userName.BCel, td.userName")
    if el:
        info["사용자"] = _txt(el)
    # 부서
    el = soup.select_one("td.useDeptName.BCel, td.useDeptName")
    if el:
        info["부서"] = _txt(el)
    # 결의금액
    el = soup.select_one("td.amtTot.BCel, td.amtTot")
    if el:
        info["결의금액"] = _txt(el)
    else:
        # sometimes span with id or currency widget
        el = soup.select_one("#apprLineRuleAmount, span[data-id='apprLineRuleAmount']")
        if el:
            info["결의금액"] = _txt(el)
    # 지급요청일
    el = soup.select_one("td.payReqDate.BCel, td.payReqDate")
    if el:
        info["지급요청일"] = _txt(el)

    # If any missing, try reading the summary table by labels (회 사 / 사용부서 / 사용자 / 결의금액 / 지급요청일)
    if not all(info.values()):
        # Find the detailSection that contains those labels
        for table in soup.find_all("table", class_="detailSection"):
            text = table.get_text(" ", strip=True)
            if any(lbl in text for lbl in ("회 사", "사용부서", "사용자", "결의금액", "지급요청일")):
                # grab by nearest BCel cells order used commonly:
                bcels = table.select("td.BCel")
                vals = [_txt(td) for td in bcels]
                # naive fallbacks by typical positions
                if not info["부서"] and len(vals) >= 2:
                    info["부서"] = vals[1]
                if not info["사용자"] and len(vals) >= 3:
                    info["사용자"] = vals[2]
                if not info["결의금액"] and len(vals) >= 7:
                    info["결의금액"] = vals[6]
                if not info["지급요청일"] and len(vals) >= 8:
                    info["지급요청일"] = vals[7]
                break

    # Final fallbacks: if any remain blank, leave as ""
    return info


def parse_line_items(soup: BeautifulSoup) -> List[Dict[str, str]]:
    """
    Extract per-line fields from #slipBplTable. Rows come in groups of 3:
      - debitTr   : 기본적요, 증빙유형, 증빙일자, 승인시간, 사용카드
      - debitTr2  : 프로젝트, 거래처, 업무용차량, 공급가액, 부가세, 합계
      - debitTr3  : 상세내용
    """
    items: List[Dict[str, str]] = []
    table = soup.find("table", {"id": "slipBplTable"})
    if not table:
        return items

    trs = table.find_all("tr", class_=["debitTr", "debitTr2", "debitTr3"])
    for i in range(0, len(trs), 3):
        group = trs[i : i + 3]
        if len(group) < 3:
            continue
        tr1, tr2, tr3 = group

        row = {
            "기본적요": _txt(tr1.select_one("td.debitRmk")),
            "증빙유형": _txt(tr1.select_one("td.taxInvestigationDivision")),
            "증빙일자": _txt(tr1.select_one("td.taxDate")),
            "승인시간": _txt(tr1.select_one("td.apprTime")),
            "사용카드": _txt(tr1.select_one("td.cardUser")),
            "프로젝트": _txt(tr2.select_one("td.dzproject")),
            "거래처": _txt(tr2.select_one("td.dztrade")),
            "업무용차량": _txt(tr2.select_one("td.workcar")),
            "공급가액": _txt(tr2.select_one("td.valueOfSupply")),
            "부가세": _txt(tr2.select_one("td.valueOfVAT")),
            "합계": _txt(tr2.select_one("td.valueOfTot")),
            "상세내용": _txt(tr3.select_one("td.summary")),
        }
        items.append(row)
    return items


def build_dataframe(items: List[Dict[str, str]], common: Dict[str, str]) -> pd.DataFrame:
    """Attach 공통값 to each row and return ordered DataFrame."""
    rows = []
    for it in items:
        row = {**it}
        row["지급요청일"] = common.get("지급요청일", "")
        row["결의금액"] = common.get("결의금액", "")
        row["부서"] = common.get("부서", "")
        row["사용자"] = common.get("사용자", "")
        rows.append(row)
    df = pd.DataFrame(rows, columns=CSV_HEADERS)
    return df


def save_csv(df: pd.DataFrame, user_name: str) -> Path:
    """Save CSV under OPTDIR with required file name pattern."""
    safe_user = (user_name or "unknown_user").strip() or "unknown_user"
    filename = f"{safe_user}_지출결의서_converted.csv"
    out_path = OPTDIR / filename
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="지출결의서 HTML → CSV 변환기", page_icon="📄", layout="centered")
st.title("📄 지출결의서 HTML → CSV 변환기")
st.caption("• HTML을 드래그 앤 드랍으로 업로드하고, **변환** 버튼을 누르면 CSV가 생성됩니다.\n• 저장 위치: `data/vectorstore/converted/`")

uploaded = st.file_uploader(
    "지출결의서 HTML 파일을 업로드하세요 (drag & drop 지원)",
    type=["html", "htm"],
    accept_multiple_files=False,
    help="지출결의서.html 파일을 선택하거나 끌어다 놓으세요.",
)

convert_clicked = st.button("변환")

if convert_clicked:
    if not uploaded:
        st.error("파일을 업로드해주세요.")
        st.stop()

    try:
        html_text = decode_bytes(uploaded.getvalue())
        soup = BeautifulSoup(html_text, "html.parser")

        common = parse_general_info(soup)
        items = parse_line_items(soup)

        if not items:
            st.warning("항목을 찾지 못했습니다. HTML 구조를 확인해주세요. (#slipBplTable이 필요)")
            st.json(common)
            st.stop()

        df = build_dataframe(items, common)
        out_path = save_csv(df, common.get("사용자", ""))

        st.success(f"CSV 생성 완료: {out_path}")
        st.download_button(
            label="CSV 다운로드",
            data=df.to_csv(index=False, encoding="utf-8-sig"),
            file_name=out_path.name,
            mime="text/csv",
        )

        with st.expander("미리보기", expanded=True):
            st.dataframe(df, use_container_width=True)

        with st.expander("추출된 공통값"):
            st.json(common)

    except Exception as e:
        st.exception(e)
        st.error("변환 중 오류가 발생했습니다. 파일 형식을 확인해주세요.")

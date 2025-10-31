# services/autofill/autofill_service.py
from bs4 import BeautifulSoup
import re

MINUTES_PER_BLOCK = 120
AMOUNT_PER_BLOCK = 10000

def hm_to_minutes(hm: str) -> int:
    m = re.match(r"(\d+)\s*h\s*(\d+)\s*m", hm)
    return int(m.group(1))*60 + int(m.group(2)) if m else 0

def minutes_to_hm(minutes: int) -> str:
    return f"{minutes // 60}h {minutes % 60}m"

def extract_hours_from_html_text(text: str):
    total  = re.search(r"총근무[:：]?\s*(\d+\s*h\s*\d+\s*m)", text)
    select = re.search(r"선택근무[:：]?\s*(\d+\s*h\s*\d+\s*m)", text)
    extend = re.search(r"초과근무[:：]?\s*(\d+\s*h\s*\d+\s*m)", text)
    total_hm  = total.group(1)  if total else ""
    select_hm = select.group(1) if select else ""
    extend_hm = extend.group(1) if extend else ""
    if not extend_hm and total_hm and select_hm:
        extend_hm = minutes_to_hm(max(0, hm_to_minutes(total_hm)-hm_to_minutes(select_hm)))
    return total_hm, select_hm, extend_hm

def process_html(path: str, output_path: str | None = None,
                 override: dict | None = None):
    """override={'total_hm':..., 'select_hm':..., 'extend_hm':...}"""
    with open(path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml")

    if override and any(override.get(k) for k in ("total_hm","select_hm","extend_hm")):
        total_hm  = override.get("total_hm","")
        select_hm = override.get("select_hm","")
        extend_hm = override.get("extend_hm","")
        if not extend_hm and total_hm and select_hm:
            extend_hm = minutes_to_hm(max(0, hm_to_minutes(total_hm)-hm_to_minutes(select_hm)))
    else:
        text = soup.get_text(" ", strip=True)
        total_hm, select_hm, extend_hm = extract_hours_from_html_text(text)

    extend_min = hm_to_minutes(extend_hm)
    blocks = extend_min // MINUTES_PER_BLOCK
    amount = blocks * AMOUNT_PER_BLOCK

    detail_text = f"초과근무: {extend_hm}(총근무:{total_hm}/선택근무:{select_hm})"

    rows = soup.select("#slipBplTable tbody tr.debitTr")
    for r in rows:
        if "야근교통비" in r.get_text():
            r2 = r.find_next_sibling("tr", class_="debitTr2")
            r3 = r.find_next_sibling("tr", class_="debitTr3")
            # 공급가액
            supply = r2.select_one("td.valueOfSupply span.editor_currency") if r2 else None
            if supply:
                supply.string = f"{amount:,}"
            # 상세내용
            detail = r3.select_one('td.summary span[data-id="editorForm_25"]') if r3 else None
            if detail:
                detail.string = detail_text
            break

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(str(soup))

    return {
        "총근무": total_hm,
        "선택근무": select_hm,
        "초과근무": extend_hm,
        "공급가액": amount,
        "상세내용": detail_text,
    }

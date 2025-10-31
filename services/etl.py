# services/reconcile.py 상단 또는 services/etl.py에 넣고 공용으로 사용
import re, unicodedata
import pandas as pd

BRANCH_WORDS = r"(본점|지점|분점|센터|영업소|대리점)"
CORP_WORDS   = r"(주식회사|유한회사|합자회사|합명회사|재단법인|사단법인|협동조합)"
CORP_MARKERS = r"[㈜\(\)\[\]{}\/\-\_·•·\-]"

def normalize_merchant_text(name: str) -> str:
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    s = str(name)

    # 1) 유니코드 정규화 (전각/반각, 조합문자 통일)
    s = unicodedata.normalize("NFKC", s)

    # 2) 괄호 안 내용 제거 (지점/호수/층 정보 등) 例: "스타벅스(판교점)" → "스타벅스"
    s = re.sub(r"[（(].*?[)）]", " ", s)

    # 3) 법인 표기 제거 (㈜, (주), 주식회사 등)
    s = re.sub(CORP_MARKERS, " ", s)
    s = re.sub(CORP_WORDS, " ", s, flags=re.IGNORECASE)

    # 4) 지점/본점 등 지점 관련 단어 제거 (단, 실제 상호에 포함된 경우는 남길지 정책 결정)
    s = re.sub(BRANCH_WORDS, " ", s)

    # 5) 한글/영문/숫자/공백만 남기고 나머지 기호 제거
    s = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", s)

    # 6) 영문 소문자화 + 다중 공백 축소 + 트림
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()

    return s

import re
import pandas as pd

_MMDD = re.compile(r"\b(\d{1,2})/(\d{1,2})\b")

def extract_mmdd_set(titles: list[str]) -> set[str]:
    s = set()
    for t in titles:
        m = _MMDD.search(str(t))
        if m:
            mm = m.group(1).zfill(2); dd = m.group(2).zfill(2)
            s.add(f"{mm}/{dd}")
    return s

def mark_missing_reports(df_tx: pd.DataFrame, titles_df: pd.DataFrame) -> pd.DataFrame:
    """
    tx df의 date(YYYY-MM-DD)를 mm/dd로 만들고, title 목록에 없으면 missing=True
    """
    if df_tx.empty: 
        df_tx['missing_report'] = False
        return df_tx
    have = extract_mmdd_set(list(titles_df['title'].astype(str)))
    df = df_tx.copy()
    df['mmdd'] = pd.to_datetime(df['date']).dt.strftime("%m/%d")
    df['missing_report'] = ~df['mmdd'].isin(have)
    return df

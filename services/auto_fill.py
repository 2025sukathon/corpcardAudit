import datetime
import pandas as pd
from typing import Optional
from utils.settings import APPLICANT_NAME

def _parse_time(t: str) -> Optional[datetime.time]:
    if not t: return None
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.datetime.strptime(t.strip(), fmt).time()
        except Exception:
            continue
    return None

def reason_from_time(tstr: str) -> str:
    """
    규칙:
    06:00~10:30  -> 아침식대
    10:30~15:30  -> 점심식대
    15:01~24:00  -> 저녁식대
    나머지/미지정 -> 식대
    """
    t = _parse_time(tstr)
    if not t: return "식대"
    if t >= datetime.strptime("06:00","%H:%M").time() and t <= datetime.strptime("10:30","%H:%M").time():
        return "아침식대"
    if t > datetime.strptime("10:30","%H:%M").time() and t <= datetime.strptime("15:30","%H:%M").time():
        return "점심식대"
    if t > datetime.strptime("15:01","%H:%M").time():
        return "저녁식대"
    return "식대"

def auto_fill_card_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    - proof_type이 '법인카드'인 행에 대해 상세내용 자동 채움:
      같은 날짜끼리 묶어 빠른 날짜부터 '출장{n}_{출장지}_{사유}'
    - destination이 비면 merchant 사용
    """
    if df.empty: return df
    df = df.copy()

    # 날짜별 시퀀스 번호 부여 (오름차순)
    df['_date_rank'] = df.groupby('date')['approval_time'].rank(method='first').astype(int)
    df['_seq'] = 0
    # 전체 날짜 기준으로 빠른 날짜부터 글로벌 시퀀스
    # (요구가 '같은 날짜끼리 묶어서 빠른날짜부터'이므로 날짜별 counter)
    date_order = {d:i+1 for i,d in enumerate(sorted(df['date'].unique()))}
    df['_seq'] = df['date'].map(date_order)

    # 기본 목적지
    dest = df['destination'].fillna("")
    dest = dest.mask(dest.str.strip()=="", df['merchant'].fillna(""))
    dest = dest.replace("", "미지정")

    # 상세내용 없는 곳만 채우기
    mask = (df['proof_type'].str.contains("법인카드", na=False)) & (df['description'].isna() | (df['description'].str.strip()==""))
    reasons = df['approval_time'].map(reason_from_time)
    df.loc[mask, 'description'] = "출장{seq}_{dest}_{reason}".format(
        seq="{seq}", dest="{dest}", reason="{reason}"
    )  # 자리표시
    df.loc[mask, 'description'] = df.loc[mask].apply(
        lambda r: f"출장{int(date_order.get(r['date'],1))}_{(r['destination'] or r['merchant'] or '미지정')}_{reason_from_time(r['approval_time'])}",
        axis=1
    )

    # 정리
    df.drop(columns=['_date_rank','_seq'], errors='ignore', inplace=True)
    return df

def auto_defaults_for_non_corp(df: pd.DataFrame) -> pd.DataFrame:
    """
    - 법인카드가 아닌 행(혹은 미기재)에 대해:
      proof_type='개인카드', merchant=APPLICANT_NAME 기본값 부여
    """
    if df.empty: return df
    df = df.copy()
    mask = ~df['proof_type'].str.contains("법인카드", na=False)
    df.loc[mask & (df['proof_type'].isna() | (df['proof_type'].str.strip()=="")), 'proof_type'] = "개인카드"
    df.loc[mask & (df['merchant'].isna() | (df['merchant'].str.strip()=="")), 'merchant'] = APPLICANT_NAME
    return df

def auto_title(report_month: str, applicant: str | None = None) -> str:
    """
    'YYYY-MM' -> 'M월 출장비용지급품의-이름'
    """
    month = report_month.split("-")[-1].lstrip("0")
    return f"{month}월 출장비용지급품의-{applicant or APPLICANT_NAME}"

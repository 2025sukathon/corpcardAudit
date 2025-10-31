# services/reconcile.py
from __future__ import annotations
import pandas as pd
import numpy as np
import re, unicodedata
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
from datetime import timedelta
from rapidfuzz import fuzz, process



def normalize_merchant_text(name: str) -> str:
    """괄호·지점명·특수문자 제거 및 통일"""
    if not isinstance(name, str):
        name = str(name) if not pd.isna(name) else ""
    s = unicodedata.normalize("NFKC", name)
    s = re.sub(r"[（(].*?[)）]", " ", s)  # 괄호 안 제거
    s = re.sub(r"[㈜\(\)\[\]{}\/\-\_·•·\-]", " ", s)  # 특수문자
    s = re.sub(r"(주식회사|유한회사|본점|지점|분점|센터|영업소|대리점)", " ", s)
    s = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", s)  # 기호 제거
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ------------------ 설정값 (필요시 app에서 넘길 수 있음) ------------------ #
AMOUNT_TOLERANCE = 0          # 금액 허용 오차(원) : 카드 단위면 0~100 정도 권장
DATE_TOLERANCE_DAYS = 0         # 날짜 허용 범위(±일) : 엣지케이스면 1로
MERCHANT_SIM_THRESHOLD = 70     # 가맹점 유사도 기준 (0~100)

# ------------------ 유틸 ------------------ #
def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date

def _to_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)

def _norm_str(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

def _merchant_sim(a: str, b: str) -> int:
    if not a or not b: return 0
    return int(fuzz.token_set_ratio(a, b))

# ------------------ 데이터 표준화 ------------------ #
def normalize_card_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["date"] = _to_date(d["date"])
    d["amount"] = _to_int(d["amount"])
    d["merchant"] = _norm_str(d["merchant"])
    # ✅ 가맹점 정규화 컬럼 추가
    d["merchant_norm"] = d["merchant"].apply(normalize_merchant_text)
    d["employee_name"] = _norm_str(d.get("employee_name", ""))
    d["proof_type"] = _norm_str(d.get("proof_type", ""))
    return d

def normalize_expense_df(df: pd.DataFrame) -> pd.DataFrame:
    e = df.copy()
    e["date"] = _to_date(e["date"])
    e["amount"] = _to_int(e["amount"])
    e["merchant"] = _norm_str(e.get("merchant", ""))
    # ✅ 결의서도 동일하게 정규화
    e["merchant_norm"] = e["merchant"].apply(normalize_merchant_text)
    e["employee_name"] = _norm_str(e.get("employee_name", ""))
    e["proof_type"] = _norm_str(e.get("proof_type", ""))
    return e

# ------------------ 코어 매칭 ------------------ #
@dataclass
class ReconcileConfig:
    amount_tol: int = AMOUNT_TOLERANCE
    date_tol_days: int = DATE_TOLERANCE_DAYS
    merchant_threshold: int = MERCHANT_SIM_THRESHOLD

def _date_close(a: pd.Timestamp, b: pd.Timestamp, days: int) -> bool:
    if pd.isna(a) or pd.isna(b): return False
    return abs(pd.to_datetime(a) - pd.to_datetime(b)) <= pd.Timedelta(days=days)

def reconcile(card_df: pd.DataFrame,
              expense_df: pd.DataFrame,
              cfg: Optional[ReconcileConfig] = None) -> Dict[str, Any]:
    """
    카드(tx) ↔ 결의(expense) 행 단위 매칭
    return {
      'matches': DataFrame,     # 두 소스 매칭 결과
      'unmatched_card': DataFrame,
      'unmatched_expense': DataFrame,
      'summary': dict
    }
    """
    cfg = cfg or ReconcileConfig()
    c = normalize_card_df(card_df)
    e = normalize_expense_df(expense_df)

    # 인덱스 보존용 id 부여
    c = c.reset_index().rename(columns={"index": "card_id"})
    e = e.reset_index().rename(columns={"index": "exp_id"})

    # 1) 금액으로 1차 후보 좁히기 (± tol)
    #   금액 버킷을 만들어서 매칭 탐색을 줄인다.
    e_buckets = {}
    for _, row in e.iterrows():
        a = row["amount"]
        for delta in range(-cfg.amount_tol, cfg.amount_tol + 1):
            e_buckets.setdefault(a + delta, []).append(row)

    matched_rows: List[Dict[str, Any]] = []
    used_exp = set()

    for _, rc in c.iterrows():
        amount = rc["amount"]
        date_c = pd.to_datetime(rc["date"])
        cand_exps = e_buckets.get(amount, [])  # 같은/유사 금액 후보

        best = None
        best_score = -1

        for rexp in cand_exps:
            if rexp["exp_id"] in used_exp:
                continue
            date_e = pd.to_datetime(rexp["date"])
            # 날짜 필터
            if not _date_close(date_c, date_e, cfg.date_tol_days):
                continue
            # 가맹점 유사도
            sim = _merchant_sim(
                rc.get("merchant_norm") or rc.get("merchant",""),
                rexp.get("merchant_norm") or rexp.get("merchant","")
            )
            if sim < cfg.merchant_threshold:
                continue
            # 스코어: (유사도, 날짜가 같으면 +보너스)
            score = sim + (10 if date_c.date() == date_e.date() else 0)
            if score > best_score:
                best_score = score
                best = rexp

        if best is not None:
            used_exp.add(best["exp_id"])
            matched_rows.append({
                "card_id": rc["card_id"],
                "exp_id": best["exp_id"],
                "date_card": rc["date"],
                "date_exp": best["date"],
                "amount_card": rc["amount"],
                "amount_exp": best["amount"],
                "merchant_card": rc.get("merchant",""),
                "merchant_exp": best.get("merchant",""),
                "merchant_sim": best_score,
                "match_status": "matched"
            })

    # 2) 매칭/미매칭 분리
    m_df = pd.DataFrame(matched_rows)
    matched_card_ids = set(m_df["card_id"]) if not m_df.empty else set()
    matched_exp_ids  = set(m_df["exp_id"]) if not m_df.empty else set()

    unmatched_card = c[~c["card_id"].isin(matched_card_ids)].copy()
    unmatched_exp  = e[~e["exp_id"].isin(matched_exp_ids)].copy()

    # 3) 추가 진단: 왜 매칭 안 됐는지 라벨링 (간단 규칙)
    def _diagnose_unmatched_card(row):
        # 같은 금액인데 날짜나 상호 때문에 실패?
        same_amt = e[e["amount"].between(row["amount"]-cfg.amount_tol, row["amount"]+cfg.amount_tol)]
        if same_amt.empty:
            return "no_amount_match"
        # 날짜는 맞는데 상호가 낮은 유사도?
        same_date = same_amt[_to_date(same_amt["date"]) == row["date"]]
        if not same_date.empty:
            sims = same_date["merchant"].apply(lambda x: _merchant_sim(row["merchant"], x))
            if sims.max() < cfg.merchant_threshold:
                return "low_merchant_similarity"
        # 날짜도 약간 다른가?
        near_date = same_amt[_to_date(same_amt["date"]).apply(lambda d: _date_close(pd.to_datetime(d), pd.to_datetime(row["date"]), cfg.date_tol_days))]
        if not near_date.empty:
            return "date_tolerance_miss"
        return "unknown"

    if not unmatched_card.empty:
        unmatched_card["reason"] = unmatched_card.apply(_diagnose_unmatched_card, axis=1)

    def _diagnose_unmatched_exp(row):
        # 결의에는 있는데 카드가 없는 경우
        same_amt = c[c["amount"].between(row["amount"]-cfg.amount_tol, row["amount"]+cfg.amount_tol)]
        if same_amt.empty:
            return "no_amount_match"
        same_date = same_amt[_to_date(same_amt["date"]) == row["date"]]
        if not same_date.empty:
            sims = same_date["merchant"].apply(lambda x: _merchant_sim(row["merchant"], x))
            if sims.max() < cfg.merchant_threshold:
                return "low_merchant_similarity"
        near_date = same_amt[_to_date(same_amt["date"]).apply(lambda d: _date_close(pd.to_datetime(d), pd.to_datetime(row["date"]), cfg.date_tol_days))]
        if not near_date.empty:
            return "date_tolerance_miss"
        return "unknown"

    if not unmatched_exp.empty:
        unmatched_exp["reason"] = unmatched_exp.apply(_diagnose_unmatched_exp, axis=1)

    # 4) 요약
    summary = {
        "total_card": int(len(c)),
        "total_expense": int(len(e)),
        "matched": int(len(m_df)),
        "unmatched_card": int(len(unmatched_card)),
        "unmatched_expense": int(len(unmatched_exp)),
        "params": {
            "amount_tolerance": cfg.amount_tol,
            "date_tolerance_days": cfg.date_tol_days,
            "merchant_sim_threshold": cfg.merchant_threshold
        }
    }

    return {
        "matches": m_df.sort_values(["date_card","amount_card"], ignore_index=True),
        "unmatched_card": unmatched_card.sort_values(["date","amount"], ignore_index=True),
        "unmatched_expense": unmatched_exp.sort_values(["date","amount"], ignore_index=True),
        "summary": summary
    }

# ------------------ 편의 함수: CSV에서 바로 대사 ------------------ #
def reconcile_from_csv(card_csv: str, expense_csv: str,
                       amount_tol: int = AMOUNT_TOLERANCE,
                       date_tol_days: int = DATE_TOLERANCE_DAYS,
                       merchant_threshold: int = MERCHANT_SIM_THRESHOLD) -> Dict[str, Any]:
    card = pd.read_csv(card_csv)
    exp = pd.read_csv(expense_csv)
    cfg = ReconcileConfig(amount_tol, date_tol_days, merchant_threshold)
    return reconcile(card, exp, cfg)

# services/ingest/adapters.py
import pandas as pd

OUR_COLS = ["report_month","employee_name","date","approval_time",
            "amount","merchant","proof_type","description","destination"]

def parse_kb_corp_csv(path:str, employee_name:str) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8")
    # ▼ 카드사 CSV 컬럼명 예시는 실제 파일에 맞게 조정
    df = pd.DataFrame({
        "report_month": pd.to_datetime(raw["거래일자"]).dt.strftime("%Y-%m"),
        "employee_name": employee_name,
        "date": pd.to_datetime(raw["거래일자"]).dt.date,
        "approval_time": raw.get("승인시간", ""),  # 없으면 빈값
        "amount": raw["이용금액"].astype(int),
        "merchant": raw["가맹점명"].astype(str),
        "proof_type": "법인카드",
        "description": "",
        "destination": ""
    })
    df = df[OUR_COLS]
    return df

def parse_personal_csv(path:str, employee_name:str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    df = pd.DataFrame({
        "report_month": pd.to_datetime(raw["date"]).dt.strftime("%Y-%m"),
        "employee_name": employee_name,
        "date": pd.to_datetime(raw["date"]).dt.date,
        "approval_time": raw.get("time",""),
        "amount": raw["amount"].astype(int),
        "merchant": raw["merchant"].astype(str),
        "proof_type": "개인카드",  # 자동 지정
        "description": "",
        "destination": ""
    })
    return df[OUR_COLS]

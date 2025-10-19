# 카드 내역 CSV(전사 시스템에서 내려받는 형식에 맞게 조정)
CARD_COLS = [
    "report_month",     # 2025-09
    "employee_name",    # 김도희
    "date",             # 2025-09-24
    "approval_time",    # 12:54:46
    "amount",           # 10000
    "merchant",         # 황솔숯불대 (또는 장소)
    "proof_type",       # 법인카드 / (비어있을 수도)
    "description",      # 상세내용(비어있으면 자동기입)
    "destination"       # 출장지(없으면 비워도 됨)
]

# 수기 입력(비법인카드/현금 등) CSV
MANUAL_COLS = [
    "report_month","employee_name","date","approval_time","amount",
    "merchant","proof_type","description","destination"
]

# 기존 출장보고서 제목 리스트 CSV (한 컬럼: title)
TRAVEL_TITLES_COLS = ["title"]

import os
from dotenv import load_dotenv
load_dotenv()

APPLICANT_NAME = os.getenv("APPLICANT_NAME", "신청자")
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID", "")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET", "")
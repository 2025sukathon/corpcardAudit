# services/alerts.py
import os
import requests

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

def send_slack_alert(message: str, level: str = "info"):
    """
    Slack으로 간단한 텍스트 알림을 보내는 함수
    level: info | warning | error
    """
    url = os.getenv("SLACK_WEBHOOK_URL")
    
    if not url:
        print("[WARN] SLACK_WEBHOOK_URL이 설정되지 않았습니다.")
        return

    emoji = {
        "info": "ℹ️",
        "warning": "⚠️",
        "error": "🚨"
    }.get(level, "💬")

    payload = {
        "text": f"{emoji} {message}"
    }

    try:
        res = requests.post(url, json=payload)
        if res.status_code != 200:
            print(f"[Slack Error] {res.status_code}: {res.text}")
    except Exception as e:
        print(f"[Slack Exception] {e}")

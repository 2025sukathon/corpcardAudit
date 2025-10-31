from services.rag import build_index, rag_answer
from __future__ import annotations
import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

# 프로젝트 내부 모듈
from utils.settings import NAVER_CLIENT_ID, NAVER_CLIENT_SECRET
from services.naver_maps import geocode, route_summary, static_map
from services.auto_fill import reason_from_time, auto_title
# OpenAI가 없으면 폴백으로 규칙형 답변만 사용
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

# ---------- 간단 의도(인텐트) 탐지 ----------


_ROUTE_PATTERNS = [
    r"(출발)\s*[:：]\s*(?P<origin>[^,\n]+)\s*[,，]\s*(도착)\s*[:：]\s*(?P<dest>[^\n]+)",
    r"(?:경로|길찾기).*(출발)\s*[:：]\s*(?P<origin>[^,\n]+).*(도착)\s*[:：]\s*(?P<dest>[^\n]+)",
    r"(?P<origin>.+?)\s*에서\s*(?P<dest>.+?)\s*(?:까지|로)\s*(?:경로|거리|시간)"
]

_MEAL_PATTERNS = [
    r"(아침|점심|저녁)\s*식대\s*규칙",
    r"(식대|사유).*규칙",
    r"사유.*자동.*입력",
]

_TITLE_PATTERNS = [
    r"(제목|타이틀).*자동(완성|생성)",
]

@dataclass
class ChatResult:
    text: str
    # 정적 지도 이미지가 있으면 바이트를 함께 반환 (Streamlit에서 표시/다운로드)
    map_image_bytes: Optional[bytes] = None
    route_info: Optional[Dict[str, Any]] = None

# ---------- 핵심 챗봇 ----------

class ExpenseChatbot:
    """
    경량 챗봇:
      - 경로 요약: '출발: A, 도착: B' → 거리/시간 + 정적 지도 이미지
      - 규칙 FAQ: 시간대→사유, 제목 자동 생성 등
      - (선택) OpenAI가 있으면 자연어 답변 강화
    """

    def __init__(self):
        self.use_llm = False
        self.client = None
        api_key = os.getenv("OPENAI_API_KEY")
        if OpenAI and api_key:
            try:
                self.client = OpenAI()
                self.use_llm = True
            except Exception:
                self.client = None
                self.use_llm = False

    # ----- 공개 메서드 -----
    def chat(self, user_text: str, context: Optional[Dict[str, Any]] = None) -> ChatResult:
        """
        user_text: 사용자가 입력한 질문/요청
        context: 필요시 전달 (예: report_month="2025-09", applicant_name="김도희")
        """
        context = context or {}
        # 1) 경로 의도?
        route_args = self._parse_route(user_text)
        if route_args:
            return self._handle_route(*route_args)

        # 2) 규칙/FAQ?
        if self._is_meal_rule(user_text):
            return ChatResult(text=self._answer_meal_rule())

        if self._is_title_rule(user_text):
            month = str(context.get("report_month", "2025-09"))
            name = str(context.get("applicant_name", "신청자"))
            return ChatResult(text=self._answer_title_rule(month, name))

        # 3) 🔍 RAG 시도 (규정 관련 키워드 있을 때만)
        if any(k in user_text for k in ["규정", "정책", "결의", "출장비", "비용", "증빙", "FAQ"]):
            try:
                ans, metas = rag_answer(user_text)
                if ans.strip():
                    return ChatResult(text=ans)
            except Exception as e:
                print(f"[RAG Error] {e}")
        # 4) LLM 응답 (폴백 포함)
        return ChatResult(text=self._llm_or_fallback(user_text, context))

    # ----- 라우팅/의도 파싱 -----
    def _parse_route(self, text: str) -> Optional[Tuple[str, str]]:
        t = self._normalize(text)
        for p in _ROUTE_PATTERNS:
            m = re.search(p, t)
            if m:
                origin = m.group("origin").strip()
                dest = m.group("dest").strip()
                if origin and dest:
                    return origin, dest
        # 간단 키워드
        if ("출발" in t and "도착" in t) or ("경로" in t and ("에서" in t and "까지" in t)):
            # 너무 느슨하면 오탐이 많으니 명시 포맷을 권장
            return None
        return None

    def _is_meal_rule(self, text: str) -> bool:
        t = self._normalize(text)
        return any(re.search(p, t) for p in _MEAL_PATTERNS)

    def _is_title_rule(self, text: str) -> bool:
        t = self._normalize(text)
        return any(re.search(p, t) for p in _TITLE_PATTERNS)

    @staticmethod
    def _normalize(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    # ----- 경로 처리 -----
    def _handle_route(self, origin: str, dest: str) -> ChatResult:
        if not (NAVER_CLIENT_ID and NAVER_CLIENT_SECRET):
            return ChatResult(
                text="네이버 지도 API 키가 설정되지 않았어요. .env의 NAVER_CLIENT_ID / NAVER_CLIENT_SECRET를 넣어주세요.\n"
                     f"요청하신 경로: 출발='{origin}', 도착='{dest}'"
            )
        o = geocode(origin)
        d = geocode(dest)
        if not (o and d):
            return ChatResult(text=f"지오코딩에 실패했어요. 다른 표현으로 검색해볼까요?\n출발='{origin}', 도착='{dest}'")
        summ = route_summary(o, d)
        img = static_map([o, d])
        if not summ:
            txt = f"경로 요약을 가져오지 못했어요. (출발: {origin}, 도착: {dest})"
        else:
            txt = f"출발 **{origin}** → 도착 **{dest}**\n- 예상 거리: **{summ['distance_km']} km**\n- 예상 소요: **{summ['duration_min']} 분**"
        return ChatResult(text=txt, map_image_bytes=(img.getvalue() if img else None), route_info=summ or {})

    # ----- 규칙형 답변 -----
    def _answer_meal_rule(self) -> str:
        return (
            "식대 사유 자동 입력 규칙은 다음과 같아요:\n"
            "- **06:00 ~ 10:30** → `아침식대`\n"
            "- **10:30 ~ 15:30** → `점심식대`\n"
            "- **15:01 이후** → `저녁식대`\n"
            "승인 시간이 비어 있거나 해석이 안 되면 기본값은 `식대`로 처리합니다.\n"
            "예: 12:10 → `점심식대`, 18:20 → `저녁식대`"
        )

    def _answer_title_rule(self, report_month: str, applicant_name: str) -> str:
        title = auto_title(report_month, applicant_name)
        m = report_month.split("-")[-1].lstrip("0")
        return (
            f"제목 자동 생성 규칙은 `'<월>월 출장비용지급품의-<신청자>'` 입니다.\n"
            f"- 입력된 report_month = {report_month} → **{m}월**\n"
            f"- 신청자 = **{applicant_name}**\n"
            f"→ 자동 제목: **{title}**"
        )

    # ----- LLM or Fallback -----
    def _llm_or_fallback(self, user_text: str, context: Dict[str, Any]) -> str:
        """
        OpenAI API가 있으면 LLM으로 친절한 답변을 생성하고,
        없으면 핵심 규칙/가이드 중심의 폴백 답변을 제공.
        """
        # 폴백: 자주 묻는 질문 키워드 몇 개를 커버
        t = self._normalize(user_text)
        if "증빙유형" in t or "개인카드" in t:
            return (
                "비법인(수기) 내역의 기본값은 다음과 같이 자동 적용됩니다:\n"
                "- `proof_type`: **개인카드**\n"
                "- `거래처(merchant)`: **신청자 이름**\n"
                "필요하면 항목별로 UI에서 수정 가능합니다."
            )
        if "보고서" in t and ("누락" in t or "경고" in t):
            return (
                "보고서 누락 검사는 결의 내역의 `date`를 `mm/dd`로 변환해, 기존 보고서 제목 리스트의 `mm/dd`와 대조해요.\n"
                "일치하는 날짜가 없으면 `missing_report=True`로 표시하고, 토스트 알림을 5초 노출합니다."
            )

        # LLM 사용
        if self.use_llm and self.client:
            sys = (
                "너는 '법인카드 사용 내역 자동 점검 및 리포팅 시스템'의 챗봇 어시스턴트야. "
                "간결하고 한국어로 답하고, 숫자/규칙/절차를 단계적으로 설명해. "
                "가능하면 사용자가 바로 실행할 수 있는 버튼/메뉴 위치나 입력 포맷 예시를 제공해."
            )
            msgs = [
                {"role": "system", "content": sys},
                {"role": "user", "content": user_text},
            ]
            try:
                resp = self.client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=msgs,
                    temperature=0.2,
                )
                return resp.choices[0].message.content or "답변을 생성했지만 빈 내용이 반환되었어요."
            except Exception as e:
                return f"(LLM 호출 실패로 폴백) 핵심 규칙 안내: {self._answer_meal_rule()}\n\n에러: {e}"

        # 최종 폴백
        return (
            "다음 항목을 도와줄 수 있어요:\n"
            "1) 경로 질의: `출발: 판교역, 도착: 대전역`\n"
            "2) 식대 사유 자동 규칙 안내\n"
            "3) 제목 자동 생성 규칙 안내\n"
            "4) 비법인카드 기본값/보고서 누락 검증 방식 설명\n"
            "OpenAI API 키를 설정하면 더 자연스러운 대화가 가능합니다."
        )

# --------- 편의 함수 (Streamlit 등에서 바로 사용) ---------

_singleton_bot: Optional[ExpenseChatbot] = None

def get_bot() -> ExpenseChatbot:
    global _singleton_bot
    if _singleton_bot is None:
        _singleton_bot = ExpenseChatbot()
    return _singleton_bot

def chat_reply(user_text: str, context: Optional[Dict[str, Any]] = None) -> ChatResult:
    bot = get_bot()
    return bot.chat(user_text, context)



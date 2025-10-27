# services/rag.py
from __future__ import annotations
import os, glob, csv, re, calendar
from datetime import date, timedelta
from typing import Optional, List, Tuple, Dict

from chromadb import PersistentClient
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# ---------------- OpenAI 클라이언트 ----------------
try:
    from openai import OpenAI as _OpenAI
    import httpx
except Exception:
    _OpenAI = None
    httpx = None

def _make_openai_client() -> Optional["_OpenAI"]:
    if not _OpenAI:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    http_proxy  = os.getenv("HTTP_PROXY")  or os.getenv("http_proxy")
    https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
    if http_proxy or https_proxy:
        # httpx 0.28+ uses 'proxy' instead of 'proxies'
        proxy = https_proxy or http_proxy
        return _OpenAI(api_key=api_key, http_client=httpx.Client(proxy=proxy, timeout=30))
    return _OpenAI(api_key=api_key)

_OPENAI_CLIENT = _make_openai_client()
_OPENAI_EMB_MODEL = os.getenv("EMB_MODEL", "text-embedding-3-small")

# ---------------- 임베딩 함수 (충돌 해결 버전) ----------------
class OpenAIEmbedder:
    """chromadb용 임베더: .name 속성과 __call__ 제공"""
    def __init__(self, client, model_name: str):
        self.client = client
        self.model_name = model_name
        self._name = f"openai:{model_name}"
        
    @property
    def name(self):
        return self._name

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Chroma 0.4.16+는 input 키워드 인자를 받음"""
        resp = self.client.embeddings.create(model=self.model_name, input=input)
        return [d.embedding for d in resp.data]

def _embedding_function():
    if _OPENAI_CLIENT:
        return OpenAIEmbedder(_OPENAI_CLIENT, os.getenv("EMB_MODEL", "text-embedding-3-small"))
    else:
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

# ---------------- 경로 설정 ----------------
DBDIR  = "data/vectorstore/chroma"
FAQDIR = "data/vectorstore"

def _chroma_client():
    os.makedirs(DBDIR, exist_ok=True)
    return PersistentClient(path=DBDIR)

# ---------------- 파일 리더 ----------------
def _read(path: str) -> str:
    if path.lower().endswith((".md", ".txt")):
        return open(path, "r", encoding="utf-8").read()
    try:
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    except Exception:
        return ""

def _read_qa_csv(path: str) -> list[tuple[str, str]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            q = (r.get("question") or "").strip()
            a = (r.get("answer") or "").strip()
            if q and a:
                rows.append((q, a))
    return rows

def _read_qa_md(path: str) -> list[tuple[str, str]]:
    # 매우 단순: "### Q:" / "A:" 포맷
    txt = open(path, "r", encoding="utf-8").read()
    pairs = []
    for block in txt.split("### Q:")[1:]:
        q, _, rest = block.partition("\n")
        a_tag = "A:"
        a = rest.split("\n", 1)[0] if a_tag not in rest else rest.split(a_tag, 1)[1].strip()
        q, a = q.strip(), a.strip()
        if q and a:
            pairs.append((q, a))
    return pairs

# ---------------- 인덱스 빌드 ----------------
def build_index() -> None:
    os.makedirs(DBDIR, exist_ok=True)
    cli = _chroma_client()
    col = cli.get_or_create_collection("faqs", embedding_function=_embedding_function())

    # 1) PDF/MD/TXT 문서 색인
    files = glob.glob(f"{FAQDIR}/*.md") + glob.glob(f"{FAQDIR}/*.txt") + glob.glob(f"{FAQDIR}/*.pdf")
    ids, docs, metas = [], [], []
    for f in files:
        txt = _read(f)
        if not txt.strip():
            continue
        ids.append(os.path.basename(f))
        docs.append(txt[:10000])
        metas.append({"path": f})
    if ids:
        col.upsert(ids=ids, documents=docs, metadatas=metas)

    # 2) CSV/MD Q&A 색인
    qa_files = glob.glob(f"{FAQDIR}/*.csv") + glob.glob(f"{FAQDIR}/faq_qa.md") + glob.glob(f"{FAQDIR}/*_qa.md")
    q_ids, q_docs, q_metas = [], [], []
    for f in qa_files:
        pairs = _read_qa_csv(f) if f.endswith(".csv") else _read_qa_md(f)
        for idx, (q, a) in enumerate(pairs):
            q_ids.append(f"{os.path.basename(f)}::Q{idx+1}")
            q_docs.append(f"[Q] {q}\n[A] {a}")
            q_metas.append({"path": f, "type": "qa"})
    if q_ids:
        col.upsert(ids=q_ids, documents=q_docs, metadatas=q_metas)

    cli.persist()

# ---------------- 마감일 규칙 & 영업일 보정 ----------------
ADJUST_TO_BUSINESS_DAY = os.getenv("RAG_DEADLINE_BUSINESS_DAY", "1") in ("1", "true", "True")
HOLIDAYS_FILE = os.getenv("RAG_HOLIDAYS_FILE", "data/vectorstore/holidays_kr.csv")

_HOLIDAYS: set[date] = set()
def _load_holidays() -> set[date]:
    days: set[date] = set()
    try:
        if os.path.exists(HOLIDAYS_FILE):
            with open(HOLIDAYS_FILE, encoding="utf-8") as f:
                for line in f:
                    s = line.strip().split(",")[0]
                    if not s:
                        continue
                    try:
                        y, m, d = s.split("-")
                        days.add(date(int(y), int(m), int(d)))
                    except:
                        pass
    except:
        pass
    return days
_HOLIDAYS = _load_holidays()

def _adjust_to_business_day(d: date, prefer: str = "previous") -> date:
    step = -1 if prefer == "previous" else +1
    cur = d
    while cur.weekday() >= 5 or cur in _HOLIDAYS:  # 5=토, 6=일
        cur = cur + timedelta(days=step)
    return cur

_DEADLINE_ANY = re.compile(r"(마감일|마감|말일)", re.I)
_THIS_MONTH   = re.compile(r"(이번\s*달|이번달)", re.I)
_NEXT_MONTH   = re.compile(r"(다음\s*달|익월)", re.I)
_PREV_MONTH   = re.compile(r"(지난\s*달|전월)", re.I)
_YEAR_MONTH   = re.compile(r"(?:(?P<y>\d{4})\s*[./-]?\s*(년)?\s*)?(?P<m>1[0-2]|0?[1-9])\s*(월)?")

def _shift_month(dt: date, delta: int) -> date:
    y, m = dt.year, dt.month + delta
    while m < 1:
        y -= 1; m += 12
    while m > 12:
        y += 1; m -= 12
    return date(y, m, 1)

def _end_of_month(dt: date) -> date:
    last = calendar.monthrange(dt.year, dt.month)[1]
    return date(dt.year, dt.month, last)

def _parse_target_month_from_text(q: str, today: date) -> Optional[date]:
    t = q.strip()
    if _THIS_MONTH.search(t): return date(today.year, today.month, 1)
    if _NEXT_MONTH.search(t): return _shift_month(today, +1)
    if _PREV_MONTH.search(t): return _shift_month(today, -1)
    m = _YEAR_MONTH.search(t)
    if m:
        yy = m.group("y")
        mm = int(m.group("m"))
        yyyy = int(yy) if yy else today.year
        return date(yyyy, mm, 1)
    return None

# ---------------- 질의 응답 ----------------
def rag_answer(question: str, k: int = 3) -> tuple[str, list[dict]]:
    # 0) 마감일 계열 특수처리 (RAG 전에 결정론)
    if _DEADLINE_ANY.search(question):
        today = date.today()
        target = _parse_target_month_from_text(question, today) or date(today.year, today.month, 1)
        eom = _end_of_month(target)
        eom_adj = _adjust_to_business_day(eom, prefer=os.getenv("RAG_DEADLINE_PREFER", "previous")) \
                  if ADJUST_TO_BUSINESS_DAY else eom
        y, m = target.year, target.month
        msg = (
            "지출결의서 마감일은 일반적으로 매월 말일입니다.\n"
            f"요청하신 기간: **{y}년 {m}월**\n"
            f"기본 말일: {eom.strftime('%Y-%m-%d')}\n"
            f"영업일 보정: **{eom_adj.strftime('%Y년 %m월 %d일')}**"
        )
        return msg, []

    # 1) 문서/QA RAG
    cli = Client(Settings(persist_directory=DBDIR, is_persistent=True))
    col = cli.get_or_create_collection("faqs", embedding_function=_embedding_function())
    r = col.query(query_texts=[question], n_results=k)
    ctx = "\n\n".join(r["documents"][0]) if r and r.get("documents") else ""

    # 2) LLM 생성 (있으면)
    if _OPENAI_CLIENT is not None:
        msgs = [
            {"role": "system", "content": "사내 규정/FAQ를 바탕으로 간결히 한국어로 답하세요. 인용은 따옴표로 표시."},
            {"role": "user",   "content": f"[컨텍스트]\n{ctx}\n\n[질문]\n{question}"},
        ]
        try:
            a = _OPENAI_CLIENT.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=msgs, temperature=0.1,
            )
            return a.choices[0].message.content, (r["metadatas"][0] if r else [])
        except Exception:
            pass

    # 3) 키가 없거나 LLM 실패 시 스니펫 반환
    return (f"상위 스니펫:\n{ctx[:1200]}\n\n(OPENAI_API_KEY 없거나 오류 시 스니펫만 표시)",
            r["metadatas"][0] if r else [])

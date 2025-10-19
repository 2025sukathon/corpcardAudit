# services/rag.py
from __future__ import annotations
import os, glob
from typing import List, Callable, Optional

from chromadb import Client
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# --- OpenAI 1.x 클라이언트 준비 (프록시 자동 인식) ---
try:
    from openai import OpenAI as _OpenAI
    import httpx
except Exception:
    _OpenAI = None
    httpx = None

def _make_openai_client() -> Optional["_OpenAI"]:
    """OPENAI_API_KEY가 있고 openai 패키지가 있으면 1.x 클라이언트를 반환."""
    if not _OpenAI:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    http_proxy  = os.getenv("HTTP_PROXY")  or os.getenv("http_proxy")
    https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")

    if http_proxy or https_proxy:
        # 둘 중 하나만 있어도 양쪽에 매핑되도록 처리
        proxies = {
            "http://":  http_proxy  or https_proxy,
            "https://": https_proxy or http_proxy,
        }
        client = _OpenAI(api_key=api_key, http_client=httpx.Client(proxies=proxies, timeout=30))
    else:
        client = _OpenAI(api_key=api_key)
    return client

# --- 임베딩 함수 결정: OpenAI(있으면) / sentence-transformers(없으면) ---
_OPENAI_CLIENT = _make_openai_client()
_OPENAI_EMB_MODEL = os.getenv("EMB_MODEL", "text-embedding-3-small")

def _openai_embed(texts: List[str]) -> List[List[float]]:
    """openai 1.x 임베딩 함수 (list[str] -> list[float])"""
    # _OPENAI_CLIENT은 옵션에 따라 None일 수 있음. 호출 전에 보장되도록 아래에서 분기.
    resp = _OPENAI_CLIENT.embeddings.create(model=_OPENAI_EMB_MODEL, input=texts)
    return [d.embedding for d in resp.data]

# fallback: 로컬 임베딩
_ST_EMB = SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)  # callable(list[str]) -> list[list[float]]

def _embedding_function() -> Callable[[List[str]], List[List[float]]]:
    """Chroma에 넘길 embedding_function 반환."""
    if _OPENAI_CLIENT is not None:
        return _openai_embed                    # 함수 자체를 넘김
    return _ST_EMB                              # SentenceTransformerEmbeddingFunction 인스턴스(호출 가능)

# --- 경로 설정 ---
DBDIR  = "data/vectorstore/chroma"
FAQDIR = "data/vectorstore"   # PDF/MD/TXT를 둘 위치

def _read(path: str) -> str:
    if path.lower().endswith((".md", ".txt")):
        return open(path, "r", encoding="utf-8").read()
    try:
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    except Exception:
        return ""

def build_index() -> None:
    os.makedirs(DBDIR, exist_ok=True)
    cli = Client(Settings(persist_directory=DBDIR, is_persistent=True))
    col = cli.get_or_create_collection(
        "faqs",
        embedding_function=_embedding_function(),
    )

    files = []
    files.extend(glob.glob(f"{FAQDIR}/*.md"))
    files.extend(glob.glob(f"{FAQDIR}/*.txt"))
    files.extend(glob.glob(f"{FAQDIR}/*.pdf"))

    ids, docs, metas = [], [], []
    for f in files:
        txt = _read(f)
        if not txt.strip():
            continue
        # 빠르게 파일 단위로 저장 (필요시 문단 split 로직 추가 가능)
        ids.append(os.path.basename(f))
        docs.append(txt[:10000])
        metas.append({"path": f})

    if ids:
        col.upsert(ids=ids, documents=docs, metadatas=metas)
    cli.persist()

def rag_answer(question: str, k: int = 3) -> tuple[str, list[dict]]:
    """질문에 대한 RAG 답변과 메타데이터(출처)를 반환."""
    cli = Client(Settings(persist_directory=DBDIR, is_persistent=True))
    col = cli.get_or_create_collection(
        "faqs",
        embedding_function=_embedding_function(),
    )

    r = col.query(query_texts=[question], n_results=k)
    ctx = "\n\n".join(r["documents"][0]) if r and r.get("documents") else ""

    # OpenAI 1.x로 답변 (키 없으면 컨텍스트만 반환)
    if _OPENAI_CLIENT is not None:
        msg = [
            {"role": "system", "content": "사내 규정/FAQ를 바탕으로 간결히 한국어로 답하세요. 인용은 따옴표로 표시."},
            {"role": "user",   "content": f"[컨텍스트]\n{ctx}\n\n[질문]\n{question}"},
        ]
        try:
            a = _OPENAI_CLIENT.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=msg,
                temperature=0.1,
            )
            return a.choices[0].message.content, (r["metadatas"][0] if r else [])
        except Exception:
            # 실패 시 컨텍스트 스니펫만 반환
            pass

    return (f"상위 스니펫:\n{ctx[:1200]}\n\n(OPENAI_API_KEY 없거나 오류 시 스니펫만 표시)", r["metadatas"][0] if r else [])

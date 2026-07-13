"""Central configuration for the monopoly RAG pipeline."""

from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT_DIR / "GIAO TRINH KINH TE CHINH TRI MAC - LENIN - BỘ GIÁO DỤC VÀ ĐÀO TẠO.pdf"
OUTPUT_DIR = ROOT_DIR / "output" / "rag"
STRUCTURED_PATH = OUTPUT_DIR / "structured.json"
CHUNKS_PATH = OUTPUT_DIR / "chunks.jsonl"
PARENTS_PATH = OUTPUT_DIR / "parents.json"
CHROMA_DIR = OUTPUT_DIR / "chroma"
EMBEDDINGS_PATH = OUTPUT_DIR / "embeddings.npy"
IDS_PATH = OUTPUT_DIR / "ids.json"
BM25_PATH = OUTPUT_DIR / "bm25.pkl"
SYNTHESIS_PATH = OUTPUT_DIR / "monopoly_synthesis_vi.md"
SOURCES_PATH = OUTPUT_DIR / "sources.md"
SUMMARY_CACHE_PATH = OUTPUT_DIR / "summary_cache.json"
FLASHCARDS_CACHE_PATH = OUTPUT_DIR / "flashcards_cache.json"


def load_env_file(path: Path | None = None) -> None:
    """Load simple KEY=VALUE pairs from .env without adding a dependency."""

    env_path = path or (ROOT_DIR / ".env")
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


# Load .env BEFORE reading any RAG_* constant below; otherwise values set in
# .env (RERANKER, TOP_K, models, chunk sizes...) are ignored at import time.
load_env_file()


def _csv_env(name: str, default: str = "") -> list[str]:
    """Parse a comma-separated environment variable into clean values."""

    return [item.strip().rstrip("/") for item in os.getenv(name, default).split(",") if item.strip()]


# Set CORS_ALLOW_ORIGINS on Railway to the public URL(s) of the frontend,
# separated by commas. Keeping production origins explicit is safer than '*'.
CORS_ALLOW_ORIGINS = _csv_env(
    "CORS_ALLOW_ORIGINS",
    "http://localhost:8899,http://127.0.0.1:8899,http://localhost:5173,"
    "http://127.0.0.1:5173,https://mln121-xi.vercel.app",
)

EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIMENSION = 3072
GENERATION_MODEL = os.getenv("RAG_GENERATION_MODEL", "gpt-4o-mini")
RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", GENERATION_MODEL)

CHROMA_COLLECTION = os.getenv("RAG_CHROMA_COLLECTION", "mln121_monopoly")
EMBED_BATCH_SIZE = int(os.getenv("RAG_EMBED_BATCH_SIZE", "32"))

DENSE_TOP_N = int(os.getenv("RAG_DENSE_TOP_N", "25"))
SPARSE_TOP_N = int(os.getenv("RAG_SPARSE_TOP_N", "25"))
RRF_K = int(os.getenv("RAG_RRF_K", "60"))
RERANKER = os.getenv("RAG_RERANKER", "off").lower()
RERANK_CANDIDATES = int(os.getenv("RAG_RERANK_CANDIDATES", "12"))
TOP_K = int(os.getenv("RAG_TOP_K", "6"))
# Cap synthesis chunks in the final context so the (broad, on-topic) summary
# does not crowd out primary textbook pages and policy sources.
SYNTHESIS_MAX = int(os.getenv("RAG_SYNTHESIS_MAX", "2"))

# --- Abstention / input guarding (behavioural safety) ---
# The corpus is small, so retrieval ALWAYS returns top_k chunks even for an
# off-corpus or nonsense query. Without a gate the model turns those "least bad"
# chunks into a confident answer. RELEVANCE_MIN is a floor on the top dense
# cosine similarity to the corpus: below it, /api/chat refuses instead of
# answering. Calibrated on eval/style_cases.jsonl (positive vs out-of-corpus
# groups). ABSTAIN_ENABLED / TRIAGE_ENABLED let us toggle the guards for A/B.
ABSTAIN_ENABLED = os.getenv("RAG_ABSTAIN_ENABLED", "1") == "1"
RELEVANCE_MIN = float(os.getenv("RAG_RELEVANCE_MIN", "0.34"))
TRIAGE_ENABLED = os.getenv("RAG_TRIAGE_ENABLED", "1") == "1"
# Gray zone [RELEVANCE_MIN, RELEVANCE_MAX): cosine can't separate same-domain but
# off-corpus questions (e.g. surplus value) from in-corpus ones. For queries that
# land here, an LLM scope-classifier reads the retrieved passages and decides
# whether they actually answer the question; below MIN we hard-abstain, above MAX
# we always answer (no extra call). Fail-open: classifier errors default to answer.
RELEVANCE_MAX = float(os.getenv("RAG_RELEVANCE_MAX", "0.62"))
SCOPE_CLASSIFIER_ENABLED = os.getenv("RAG_SCOPE_CLASSIFIER", "1") == "1"
# Total budget for all context passed to the LLM.
CONTEXT_TOKEN_CAP = int(os.getenv("RAG_CONTEXT_TOKEN_CAP", "6000"))
# Max size of a single parent section to expand to (small-to-big). If a parent
# is larger, retrieval falls back to a sentence window around the matched chunk
# instead of dropping straight to the lone small chunk.
PARENT_MAX_TOKENS = int(os.getenv("RAG_PARENT_MAX_TOKENS", "2000"))

CHUNK_MIN_TOKENS = int(os.getenv("RAG_CHUNK_MIN_TOKENS", "450"))
CHUNK_MAX_TOKENS = int(os.getenv("RAG_CHUNK_MAX_TOKENS", "600"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("RAG_CHUNK_OVERLAP_TOKENS", "90"))

INCLUDE_CHAPTER5_PART3 = os.getenv("RAG_INCLUDE_CH5_PART3", "0") == "1"
QUIZ_VERIFY = os.getenv("RAG_QUIZ_VERIFY", "1") == "1"
QUIZ_VERIFY_RETRIES = int(os.getenv("RAG_QUIZ_VERIFY_RETRIES", "1"))

# Errors that should never be retried (bad key, bad request, etc.).
_NON_RETRYABLE = {"AuthenticationError", "PermissionDeniedError", "BadRequestError", "NotFoundError"}


def with_backoff(fn, *, retries: int = 5, base_delay: float = 1.0, max_delay: float = 20.0):
    """Call fn() with exponential backoff on transient OpenAI/network errors."""

    import time

    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - retry transient, re-raise the rest
            if type(exc).__name__ in _NON_RETRYABLE or attempt == retries - 1:
                raise
            delay = min(max_delay, base_delay * (2 ** attempt))
            print(f"[retry {attempt + 1}/{retries}] {type(exc).__name__}: {exc} -> sleep {delay:.1f}s")
            time.sleep(delay)


def get_openai_client():
    """Return a process-wide cached OpenAI client."""

    from functools import lru_cache

    @lru_cache(maxsize=1)
    def _client():
        from openai import OpenAI

        return OpenAI(api_key=require_openai_api_key())

    return _client()


def require_openai_api_key() -> str:
    load_env_file()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Add it to .env or export it in the shell.")
    return api_key


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

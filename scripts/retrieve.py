"""Hybrid retrieval for the monopoly RAG corpus."""

from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import tiktoken

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts import config
from scripts.ingest.embed_index import tokenize_vi


try:
    ENCODING = tiktoken.encoding_for_model(config.GENERATION_MODEL)
except KeyError:
    ENCODING = tiktoken.get_encoding("cl100k_base")


@dataclass
class Candidate:
    chunk_id: str
    score: float
    source: str
    chunk: dict[str, Any]
    # Raw dense cosine similarity to the query, kept separate from `score` so
    # that rrf_fuse/rerank (which overwrite `score`) don't destroy the signal we
    # use for the abstention gate. 0.0 for candidates that came only from BM25.
    dense_cosine: float = 0.0


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


@lru_cache(maxsize=1)
def load_chunks(path: Path = config.CHUNKS_PATH) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"Missing chunks file: {path}. Run `python scripts/ingest/build.py` first.")
    return {row["id"]: row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}


@lru_cache(maxsize=1)
def load_parents(path: Path = config.PARENTS_PATH) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"Missing parents file: {path}. Run `python scripts/ingest/build.py` first.")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_bm25() -> dict[str, Any]:
    if not config.BM25_PATH.exists():
        raise RuntimeError(f"Missing BM25 index: {config.BM25_PATH}. Run `python scripts/ingest/build.py` first.")
    with config.BM25_PATH.open("rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def load_embedding_matrix() -> tuple[list[str], np.ndarray]:
    """Load cached chunk embeddings for dense search.

    Chroma's native backend has crashed on some Windows/Python combinations in
    this project. The corpus is tiny, so a direct NumPy cosine search is simpler
    and avoids a runtime dependency on Chroma for serving.
    """

    if not config.EMBEDDINGS_PATH.exists() or not config.IDS_PATH.exists():
        raise RuntimeError(
            f"Missing embedding cache: {config.EMBEDDINGS_PATH} / {config.IDS_PATH}. "
            "Run `python scripts/ingest/build.py` first."
        )

    payload = json.loads(config.IDS_PATH.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if payload.get("model") != config.EMBEDDING_MODEL:
            raise RuntimeError(
                f"Embedding cache model is {payload.get('model')}, expected {config.EMBEDDING_MODEL}. "
                "Rebuild the index."
            )
        items = payload.get("items", [])
    else:
        items = payload

    ids = [item["id"] if isinstance(item, dict) else str(item) for item in items]
    embeddings = np.load(config.EMBEDDINGS_PATH).astype(np.float32)
    if len(ids) != len(embeddings):
        raise RuntimeError("Embedding cache mismatch: ids.json and embeddings.npy have different lengths.")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return ids, embeddings / norms


def embed_query(query: str) -> list[float]:
    client = config.get_openai_client()
    response = config.with_backoff(
        lambda: client.embeddings.create(model=config.EMBEDDING_MODEL, input=query)
    )
    return response.data[0].embedding


def dense_search(
    query: str,
    chunks_by_id: dict[str, dict[str, Any]],
    n: int = config.DENSE_TOP_N,
    where: dict[str, Any] | None = None,
) -> list[Candidate]:
    ids, embeddings = load_embedding_matrix()
    query_embedding = np.array(embed_query(query), dtype=np.float32)
    query_norm = np.linalg.norm(query_embedding)
    if query_norm == 0:
        return []
    scores = embeddings @ (query_embedding / query_norm)

    candidates: list[Candidate] = []
    for idx in np.argsort(scores)[::-1]:
        chunk_id = ids[int(idx)]
        chunk = chunks_by_id.get(chunk_id)
        if not chunk:
            continue
        if where and any(chunk.get(key) != value for key, value in where.items()):
            continue
        cosine = float(scores[int(idx)])
        candidates.append(
            Candidate(chunk_id=chunk_id, score=cosine, source="dense", chunk=chunk, dense_cosine=cosine)
        )
        if len(candidates) >= n:
            break
    return candidates


def sparse_search(
    query: str,
    chunks_by_id: dict[str, dict[str, Any]],
    n: int = config.SPARSE_TOP_N,
) -> list[Candidate]:
    payload = load_bm25()
    ids = payload["ids"]
    scores = payload["bm25"].get_scores(tokenize_vi(query))
    if len(scores) == 0:
        return []

    top_indices = np.argsort(scores)[::-1][:n]
    candidates: list[Candidate] = []
    for idx in top_indices:
        score = float(scores[idx])
        if score <= 0:
            continue
        chunk_id = ids[idx]
        chunk = chunks_by_id.get(chunk_id)
        if chunk:
            candidates.append(Candidate(chunk_id=chunk_id, score=score, source="sparse", chunk=chunk))
    return candidates


def rrf_fuse(dense: list[Candidate], sparse: list[Candidate], k: int = config.RRF_K) -> list[Candidate]:
    fused: dict[str, Candidate] = {}
    scores: dict[str, float] = {}

    for ranking in (dense, sparse):
        for rank, candidate in enumerate(ranking, start=1):
            scores[candidate.chunk_id] = scores.get(candidate.chunk_id, 0.0) + 1.0 / (k + rank)
            if candidate.chunk_id not in fused:
                fused[candidate.chunk_id] = candidate

    ordered = sorted(fused.values(), key=lambda item: scores[item.chunk_id], reverse=True)
    for candidate in ordered:
        candidate.score = scores[candidate.chunk_id]
        candidate.source = "rrf"
    return ordered


def rerank_openai(query: str, candidates: list[Candidate], top_k: int) -> list[Candidate]:
    if not candidates:
        return []

    shortlist = candidates[: config.RERANK_CANDIDATES]
    items = []
    for idx, candidate in enumerate(shortlist, start=1):
        text = candidate.chunk["embedding_text"]
        if len(text) > 1800:
            text = text[:1800] + "..."
        items.append({"rank": idx, "id": candidate.chunk_id, "text": text})

    prompt = {
        "query": query,
        "instruction": (
            "Rerank các đoạn theo mức liên quan để trả lời câu hỏi. "
            "Ưu tiên đoạn có định nghĩa, nguyên nhân, vai trò Nhà nước, điện/nước Việt Nam nếu câu hỏi cần. "
            "Trả JSON object dạng {\"items\":[{\"id\":\"...\",\"score\":0-100}]}."
        ),
        "candidates": items,
    }
    client = config.get_openai_client()
    response = config.with_backoff(
        lambda: client.chat.completions.create(
            model=config.RERANK_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Bạn là reranker JSON-only cho hệ thống RAG tiếng Việt."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
        )
    )
    content = response.choices[0].message.content or "{}"
    try:
        data = json.loads(content)
        scored = {item["id"]: float(item.get("score", 0)) for item in data.get("items", [])}
    except Exception:
        return candidates[:top_k]

    by_id = {candidate.chunk_id: candidate for candidate in shortlist}
    reranked = []
    for chunk_id, score in sorted(scored.items(), key=lambda pair: pair[1], reverse=True):
        candidate = by_id.get(chunk_id)
        if candidate:
            candidate.score = score
            candidate.source = "openai_rerank"
            reranked.append(candidate)

    seen = {candidate.chunk_id for candidate in reranked}
    reranked.extend(candidate for candidate in candidates if candidate.chunk_id not in seen)
    return reranked


def rerank(query: str, candidates: list[Candidate], top_k: int = config.TOP_K) -> list[Candidate]:
    """Return candidates re-ordered by relevance (full list, no truncation)."""

    if config.RERANKER == "openai":
        return rerank_openai(query, candidates, top_k=top_k)
    return candidates


def diversify(
    candidates: list[Candidate],
    top_k: int,
    synthesis_max: int = config.SYNTHESIS_MAX,
) -> list[Candidate]:
    """Pick top_k in order while capping synthesis chunks for source diversity.

    The synthesis doc is broad and on-topic, so it tends to fill every slot.
    Capping it leaves room for the primary textbook pages and policy sources
    that a complete answer needs.
    """

    selected: list[Candidate] = []
    synthesis_used = 0
    for candidate in candidates:
        if candidate.chunk.get("source_type") == "synthesis":
            if synthesis_used >= synthesis_max:
                continue
            synthesis_used += 1
        selected.append(candidate)
        if len(selected) >= top_k:
            return selected

    # Backfill if the cap left us short (e.g. corpus is mostly synthesis).
    chosen = {c.chunk_id for c in selected}
    for candidate in candidates:
        if candidate.chunk_id not in chosen:
            selected.append(candidate)
            if len(selected) >= top_k:
                break
    return selected


def sibling_chunks(parent_id: str, chunks_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """All chunks belonging to the same parent, in reading order."""

    siblings = [c for c in chunks_by_id.values() if c.get("parent_id") == parent_id]
    siblings.sort(key=lambda c: c["id"])
    return siblings


def build_window(
    candidate: Candidate,
    chunks_by_id: dict[str, dict[str, Any]],
    parent: dict[str, Any],
    max_tokens: int,
) -> tuple[str, set[str]]:
    """Grow a window of neighbouring sibling chunks around the matched chunk.

    Used when the parent section is too large to include whole. Returns the
    window text (breadcrumb + joined siblings in order) and the set of chunk ids
    it covers (for dedup).
    """

    siblings = sibling_chunks(candidate.chunk["parent_id"], chunks_by_id)
    ids = [c["id"] for c in siblings]
    try:
        center = ids.index(candidate.chunk_id)
    except ValueError:
        center = 0

    selected = {center}
    total = count_tokens(siblings[center]["text"])
    lo, hi = center - 1, center + 1
    while lo >= 0 or hi < len(siblings):
        for idx in (lo, hi):
            if 0 <= idx < len(siblings) and idx not in selected:
                extra = count_tokens(siblings[idx]["text"])
                if total + extra <= max_tokens:
                    selected.add(idx)
                    total += extra
        lo -= 1
        hi += 1

    ordered = [siblings[i] for i in sorted(selected)]
    breadcrumb = parent["text"].split("\n\n", 1)[0]
    body = " ".join(chunk["text"] for chunk in ordered)
    return f"{breadcrumb}\n\n{body}".strip(), {chunk["id"] for chunk in ordered}


def expand_small_to_big(
    candidates: list[Candidate],
    parents: dict[str, dict[str, Any]],
    chunks_by_id: dict[str, dict[str, Any]],
    token_cap: int = config.CONTEXT_TOKEN_CAP,
    parent_max_tokens: int = config.PARENT_MAX_TOKENS,
) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    covered_chunks: set[str] = set()
    total_tokens = 0

    for candidate in candidates:
        if candidate.chunk_id in covered_chunks:
            continue

        parent_id = candidate.chunk.get("parent_id")
        parent = parents.get(parent_id) if parent_id else None
        meta = parent or candidate.chunk

        if parent and count_tokens(parent["text"]) <= parent_max_tokens:
            # Whole parent fits -> use it (the original small-to-big behaviour).
            key = parent_id
            text = parent["text"]
            covered = {c["id"] for c in sibling_chunks(parent_id, chunks_by_id)}
        elif parent:
            # Parent too big -> window of neighbouring sentences in the section.
            text, covered = build_window(candidate, chunks_by_id, parent, parent_max_tokens)
            key = f"{parent_id}:{min(covered)}-{max(covered)}"
        else:
            # No parent (e.g. policy/synthesis stored flat).
            key = candidate.chunk_id
            text = candidate.chunk.get("text", candidate.chunk.get("embedding_text", ""))
            covered = {candidate.chunk_id}

        if key in seen_keys:
            continue
        tokens = count_tokens(text)
        if contexts and total_tokens + tokens > token_cap:
            continue

        seen_keys.add(key)
        covered_chunks |= covered
        total_tokens += tokens
        contexts.append(
            {
                "chunk_id": candidate.chunk_id,
                "parent_id": parent_id,
                "score": candidate.score,
                "retrieval_source": candidate.source,
                "source_type": meta.get("source_type"),
                "source_title": meta.get("source_title"),
                "source_url": meta.get("source_url"),
                "citation": meta.get("citation"),
                "chapter": meta.get("chapter"),
                "section_path": meta.get("section_path"),
                "page_start": meta.get("page_start"),
                "page_end": meta.get("page_end"),
                "token_count": tokens,
                "text": text,
            }
        )
    return contexts


def retrieve_with_relevance(
    query: str,
    top_k: int = config.TOP_K,
    where: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """Run the hybrid pipeline and also return a relevance signal.

    `relevance` is the highest dense cosine similarity between the query and any
    corpus chunk (computed before RRF fusion, which would otherwise overwrite the
    score). It answers "does the corpus contain anything close to this query?" and
    drives the abstention gate in the chat endpoint. It is NOT the RRF score.
    """

    chunks_by_id = load_chunks()
    parents = load_parents()
    dense = dense_search(query, chunks_by_id, where=where)
    sparse = sparse_search(query, chunks_by_id)
    relevance = max((c.dense_cosine for c in dense), default=0.0)
    fused = rrf_fuse(dense, sparse)
    ranked = rerank(query, fused, top_k=top_k)
    selected = diversify(ranked, top_k=top_k)
    contexts = expand_small_to_big(selected, parents, chunks_by_id)
    return contexts, relevance


def hybrid_retrieve(
    query: str,
    top_k: int = config.TOP_K,
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    contexts, _ = retrieve_with_relevance(query, top_k=top_k, where=where)
    return contexts


def format_source(context: dict[str, Any]) -> str:
    if context.get("source_type") == "textbook":
        page = context.get("page_start")
        page_end = context.get("page_end")
        page_text = f"tr.{page}" if page == page_end else f"tr.{page}-{page_end}"
        return f"Giáo trình, {context.get('chapter')}, {page_text}, {context.get('section_path')}"
    if context.get("citation"):
        return str(context["citation"])
    return str(context.get("source_title") or context.get("source_type") or "Nguồn")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    args = parser.parse_args()

    contexts = hybrid_retrieve(args.query, top_k=args.top_k)
    for idx, context in enumerate(contexts, start=1):
        print(f"\n[{idx}] score={context['score']:.4f} {format_source(context)}")
        print(context["text"][:1000].replace("\n", " "))


if __name__ == "__main__":
    main()

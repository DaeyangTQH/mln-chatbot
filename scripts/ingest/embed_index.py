"""Embed chunks and build dense Chroma plus sparse BM25 indexes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

try:
    import chromadb
    from chromadb.config import Settings
except Exception:  # pragma: no cover - Chroma is optional for serving
    chromadb = None
    Settings = None

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts import config


def load_chunks(path: Path = config.CHUNKS_PATH) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_embedding_cache() -> dict[str, np.ndarray]:
    if not config.EMBEDDINGS_PATH.exists() or not config.IDS_PATH.exists():
        return {}

    data = json.loads(config.IDS_PATH.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if data.get("model") != config.EMBEDDING_MODEL:
            return {}
        items = data.get("items", [])
    else:
        items = data

    embeddings = np.load(config.EMBEDDINGS_PATH)
    cache: dict[str, np.ndarray] = {}
    for item, vector in zip(items, embeddings):
        if isinstance(item, dict) and item.get("hash"):
            cache[item["hash"]] = vector
    return cache


def save_embedding_cache(chunks: list[dict[str, Any]], embeddings: np.ndarray) -> None:
    items = [
        {
            "id": chunk["id"],
            "hash": text_hash(chunk["embedding_text"]),
        }
        for chunk in chunks
    ]
    np.save(config.EMBEDDINGS_PATH, embeddings)
    config.IDS_PATH.write_text(
        json.dumps(
            {
                "model": config.EMBEDDING_MODEL,
                "dimension": config.EMBEDDING_DIMENSION,
                "items": items,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def embed_missing(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    client = config.get_openai_client()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), config.EMBED_BATCH_SIZE):
        batch = texts[start : start + config.EMBED_BATCH_SIZE]
        response = config.with_backoff(
            lambda: client.embeddings.create(model=config.EMBEDDING_MODEL, input=batch)
        )
        vectors.extend([item.embedding for item in response.data])
        print(f"embedded {min(start + len(batch), len(texts))}/{len(texts)}")
    return vectors


def build_embeddings(chunks: list[dict[str, Any]]) -> np.ndarray:
    cache = load_embedding_cache()
    result: list[np.ndarray | None] = []
    missing_texts: list[str] = []
    missing_positions: list[int] = []

    for idx, chunk in enumerate(chunks):
        key = text_hash(chunk["embedding_text"])
        cached = cache.get(key)
        if cached is not None:
            result.append(cached)
        else:
            result.append(None)
            missing_positions.append(idx)
            missing_texts.append(chunk["embedding_text"])

    new_vectors = embed_missing(missing_texts)
    for position, vector in zip(missing_positions, new_vectors):
        result[position] = np.array(vector, dtype=np.float32)

    if any(vector is None for vector in result):
        raise RuntimeError("Embedding cache build failed: some vectors are missing.")

    embeddings = np.vstack([vector for vector in result if vector is not None]).astype(np.float32)
    save_embedding_cache(chunks, embeddings)
    return embeddings


def chroma_metadata(chunk: dict[str, Any]) -> dict[str, str | int | float | bool]:
    return {
        "parent_id": chunk.get("parent_id") or "",
        "source_type": chunk.get("source_type") or "",
        "source_title": chunk.get("source_title") or "",
        "source_file": chunk.get("source_file") or "",
        "source_url": chunk.get("source_url") or "",
        "citation": chunk.get("citation") or "",
        "chapter": chunk.get("chapter") or "",
        "section_path": chunk.get("section_path") or "",
        "page_start": chunk.get("page_start") if chunk.get("page_start") is not None else -1,
        "page_end": chunk.get("page_end") if chunk.get("page_end") is not None else -1,
        "topic": chunk.get("topic") or "",
        "token_count": int(chunk.get("token_count") or 0),
    }


def build_chroma(chunks: list[dict[str, Any]], embeddings: np.ndarray, reset: bool = True) -> None:
    if os.getenv("RAG_BUILD_CHROMA", "0") != "1":
        print("skip chroma: RAG_BUILD_CHROMA is not 1; NumPy dense search uses embeddings.npy")
        return
    if chromadb is None or Settings is None:
        raise RuntimeError("chromadb is not installed. Install it or leave RAG_BUILD_CHROMA unset.")

    client = chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    if reset:
        try:
            client.delete_collection(config.CHROMA_COLLECTION)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=config.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["embedding_text"] for chunk in chunks]
    metadatas = [chroma_metadata(chunk) for chunk in chunks]

    for start in range(0, len(chunks), 1000):
        end = start + 1000
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end].tolist(),
        )


def tokenize_vi(text: str) -> list[str]:
    text = text.lower()
    try:
        from pyvi.ViTokenizer import tokenize

        return tokenize(text).split()
    except Exception:
        return re_fallback_tokenize(text)


def re_fallback_tokenize(text: str) -> list[str]:
    import re

    return re.findall(r"[\wÀ-ỹ]+", text.lower())


def build_bm25(chunks: list[dict[str, Any]]) -> None:
    tokenized = [tokenize_vi(chunk["embedding_text"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    payload = {
        "ids": [chunk["id"] for chunk in chunks],
        "tokenized_corpus": tokenized,
        "bm25": bm25,
    }
    with config.BM25_PATH.open("wb") as f:
        pickle.dump(payload, f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, default=config.CHUNKS_PATH)
    parser.add_argument("--no-reset-chroma", action="store_true")
    args = parser.parse_args()

    config.ensure_output_dir()
    chunks = load_chunks(args.chunks)
    embeddings = build_embeddings(chunks)
    build_chroma(chunks, embeddings, reset=not args.no_reset_chroma)
    build_bm25(chunks)
    stats = {
        "chunks": len(chunks),
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_shape": list(embeddings.shape),
        "chroma_dir": str(config.CHROMA_DIR),
        "bm25_path": str(config.BM25_PATH),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

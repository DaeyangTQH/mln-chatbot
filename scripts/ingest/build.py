"""Run the full offline ingest pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts import config
from scripts.ingest.chunk import build_chunks, write_jsonl
from scripts.ingest.embed_index import build_bm25, build_chroma, build_embeddings
from scripts.ingest.extract_structure import build_structured


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-ch5-part3", action="store_true", default=config.INCLUDE_CHAPTER5_PART3)
    parser.add_argument("--keep-chroma", action="store_true", help="Do not remove the existing Chroma directory first.")
    args = parser.parse_args()

    config.ensure_output_dir()
    if config.CHROMA_DIR.exists() and not args.keep_chroma:
        shutil.rmtree(config.CHROMA_DIR)

    structured = build_structured(include_ch5_part3=args.include_ch5_part3)
    config.STRUCTURED_PATH.write_text(
        json.dumps(structured, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"structured: {structured['stats']}")

    chunks, parents = build_chunks(config.STRUCTURED_PATH)
    write_jsonl(config.CHUNKS_PATH, chunks)
    config.PARENTS_PATH.write_text(json.dumps(parents, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"chunks: {len(chunks)}, parents: {len(parents)}")

    embeddings = build_embeddings(chunks)
    build_chroma(chunks, embeddings, reset=True)
    build_bm25(chunks)
    stats = {
        "structured": structured["stats"],
        "chunks": len(chunks),
        "parents": len(parents),
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_shape": list(embeddings.shape),
        "chroma_collection": config.CHROMA_COLLECTION,
        "reranker_default": config.RERANKER,
    }
    (config.OUTPUT_DIR / "pipeline_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

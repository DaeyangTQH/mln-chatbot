"""Small retrieval evaluation set for the monopoly RAG corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.retrieve import format_source, hybrid_retrieve


# Each case lists the components that MUST all be retrieved (AND semantics).
# A component is either a textbook page range or a non-textbook source_type.
# Paraphrases probe real recall, not just the canonical wording.
GOLDEN = [
    {
        "query": "Độc quyền là gì?",
        "need": [{"type": "textbook", "pages": [(112, 121)]}],
    },
    {
        "query": "Thế nào được gọi là độc quyền trong nền kinh tế thị trường?",  # paraphrase
        "need": [{"type": "textbook", "pages": [(112, 121)]}],
    },
    {
        "query": "Nguyên nhân hình thành độc quyền?",
        "need": [{"type": "textbook", "pages": [(112, 121)]}],
    },
    {
        "query": "Vì sao lại xuất hiện các tổ chức độc quyền?",  # paraphrase
        "need": [{"type": "textbook", "pages": [(112, 121)]}],
    },
    {
        "query": "Độc quyền nhà nước là gì?",
        "need": [{"type": "textbook", "pages": [(116, 119), (130, 134)]}],
    },
    {
        # Multi-source: needs BOTH the Vietnam-application textbook pages AND a
        # policy source. Retrieving only a broad synthesis chunk no longer counts.
        "query": "Vì sao Việt Nam không tư nhân hóa điện và nước?",
        "need": [
            {"type": "textbook", "pages": [(57, 58), (153, 176)]},
            {"type": "policy"},
        ],
    },
    {
        "query": "Tại sao Nhà nước vẫn giữ vai trò chi phối ngành điện và nước sạch ở Việt Nam?",  # paraphrase
        "need": [{"type": "policy"}],
    },
]


def overlaps(page_start: int | None, page_end: int | None, expected: tuple[int, int]) -> bool:
    if page_start is None or page_end is None:
        return False
    return max(page_start, expected[0]) <= min(page_end, expected[1])


def covers(context: dict[str, Any], need: dict[str, Any]) -> bool:
    if context.get("source_type") != need["type"]:
        return False
    if "pages" in need:
        return any(overlaps(context.get("page_start"), context.get("page_end"), pr) for pr in need["pages"])
    return True


def describe_need(need: dict[str, Any]) -> str:
    if "pages" in need:
        return f"{need['type']} {need['pages']}"
    return need["type"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    hits = 0
    for idx, case in enumerate(GOLDEN, start=1):
        contexts = hybrid_retrieve(case["query"], top_k=args.top_k)
        satisfied = {
            describe_need(need): any(covers(context, need) for context in contexts)
            for need in case["need"]
        }
        case_hit = all(satisfied.values())
        hits += int(case_hit)
        print(f"\nCASE {idx}: {case['query']}")
        print(f"hit@{args.top_k}: {'YES' if case_hit else 'NO'}")
        for need_desc, ok in satisfied.items():
            print(f"    need[{need_desc}]: {'OK' if ok else 'MISS'}")
        for rank, context in enumerate(contexts, start=1):
            marker = "*" if any(covers(context, need) for need in case["need"]) else " "
            print(f"{marker} [{rank}] score={context['score']:.4f} {format_source(context)}")
            preview = context["text"].replace("\n", " ")[:200]
            print(f"    {preview}")

    total = len(GOLDEN)
    print(f"\nSUMMARY hit@{args.top_k}: {hits}/{total} = {hits / total:.2%}")


if __name__ == "__main__":
    main()

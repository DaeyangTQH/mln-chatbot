"""Map-reduce summariser over the structured textbook tree.

Two levels of "reduce":
  - node summary  (map)    : each subsection/section node -> vài bullet.
  - scope summary (reduce) : gộp các node summary theo chương, hoặc gộp các
                              chương thành bản tóm tắt cả bài.

Chỉ bước reduce cuối được stream ra UI; các bước map chạy trước và phát status.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts import config

DEPTHS = {
    "short": "Tóm tắt cô đọng: 4-6 gạch đầu dòng, mỗi ý 1 câu.",
    "detailed": "Tóm tắt chi tiết có cấu trúc: chia theo tiểu mục, mỗi mục 2-4 ý, giữ khái niệm & định nghĩa quan trọng.",
}
SUMMARY_PROMPT_VERSION = "1"
_BULLET_MARKER = re.compile(r"^(?:[-•*]\s+|\d+[.)]\s+)")


@lru_cache(maxsize=1)
def _structured() -> dict[str, Any]:
    if not config.STRUCTURED_PATH.exists():
        raise RuntimeError(
            f"Missing {config.STRUCTURED_PATH}. Run `python scripts/ingest/build.py` first."
        )
    return json.loads(config.STRUCTURED_PATH.read_text(encoding="utf-8"))


def _textbook_nodes() -> list[dict[str, Any]]:
    return [n for n in _structured()["nodes"] if n.get("source_type") == "textbook" and n.get("text")]


def list_scopes() -> list[dict[str, str]]:
    """Chương có trong corpus, theo thứ tự xuất hiện, để đổ vào dropdown UI."""

    scopes: list[dict[str, str]] = [{"id": "all", "label": "Cả bài (toàn corpus)", "chapter": ""}]
    seen: list[str] = []
    for node in _textbook_nodes():
        chapter = node.get("chapter")
        title = node.get("chapter_title") or ""
        if chapter and chapter not in seen:
            seen.append(chapter)
            label = f"{chapter} — {title[:48]}" if title else chapter
            scopes.append({"id": chapter, "label": label, "chapter": chapter})
    return scopes


def _nodes_for_scope(scope: str) -> list[dict[str, Any]]:
    nodes = _textbook_nodes()
    if scope in ("", "all"):
        return nodes
    return [n for n in nodes if n.get("chapter") == scope]


def _label(node: dict[str, Any]) -> str:
    parts = [node.get("chapter"), node.get("section"), node.get("subsection")]
    return " > ".join(p for p in parts if p and p != "None")


def _llm(messages: list[dict[str, str]], stream: bool = False):
    client = config.get_openai_client()
    return config.with_backoff(
        lambda: client.chat.completions.create(
            model=config.GENERATION_MODEL,
            temperature=0.2,
            messages=messages,
            stream=stream,
        )
    )


def _node_hash(node: dict[str, Any]) -> str:
    payload = f"{SUMMARY_PROMPT_VERSION}|{config.GENERATION_MODEL}|{node.get('text', '')}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_summary_cache() -> dict[str, list[str]]:
    try:
        data = json.loads(config.SUMMARY_CACHE_PATH.read_text(encoding="utf-8"))
        items = data.get("items", {})
        return {key: value for key, value in items.items() if isinstance(value, list)}
    except (OSError, json.JSONDecodeError, AttributeError):
        return {}


def save_summary_cache(cache: dict[str, list[str]]) -> None:
    config.ensure_output_dir()
    temp_path = config.SUMMARY_CACHE_PATH.with_suffix(".tmp")
    temp_path.write_text(json.dumps({"items": cache}, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_path, config.SUMMARY_CACHE_PATH)


def _map_node(node: dict[str, Any]) -> list[str]:
    """Rút gọn 1 node thành vài bullet (bước MAP)."""

    text = node.get("text", "")[:6000]
    messages = [
        {
            "role": "system",
            "content": "Bạn tóm tắt giáo trình Kinh tế chính trị Mác - Lênin, trung thực với nội dung gốc, không bịa.",
        },
        {
            "role": "user",
            "content": (
                f"Mục: {_label(node)} (tr.{node.get('page_start')}-{node.get('page_end')})\n\n"
                f"Nội dung:\n{text}\n\n"
                "Rút gọn thành 2-4 gạch đầu dòng ngắn, giữ đúng khái niệm/định nghĩa cốt lõi."
            ),
        },
    ]
    response = _llm(messages, stream=False)
    raw = response.choices[0].message.content or ""
    bullets: list[str] = []
    for line in raw.splitlines():
        # Bỏ ĐÚNG một dấu đầu dòng ("- ", "• ", "* ", "12. ", "12) ") — cần khoảng trắng
        # phía sau nên KHÔNG cắt nhầm "**in đậm**", và xử lý được số nhiều chữ số.
        cleaned = _BULLET_MARKER.sub("", line.strip()).strip()
        if cleaned:
            bullets.append(cleaned)
    return bullets or ([raw.strip()] if raw.strip() else [])


def summarize_stream(scope: str = "all", depth: str = "short") -> Iterator[dict[str, Any]]:
    """Sinh sự kiện dict: {type: status|token|done|error, ...} để API bọc thành SSE."""

    try:
        nodes = _nodes_for_scope(scope)
        if not nodes:
            yield {"type": "error", "message": f"Không có nội dung cho phạm vi '{scope}'."}
            return

        depth_instruction = DEPTHS.get(depth, DEPTHS["short"])
        scope_label = "toàn bộ nội dung độc quyền" if scope in ("", "all") else scope

        # --- MAP ---
        node_summaries: list[str] = []
        cache = load_summary_cache()
        dirty = False
        for idx, node in enumerate(nodes, start=1):
            yield {"type": "status", "message": f"Đang tóm tắt mục {idx}/{len(nodes)}: {_label(node)}"}
            node_key = _node_hash(node)
            bullets = cache.get(node_key)
            if bullets is None:
                bullets = _map_node(node)
                cache[node_key] = bullets
                dirty = True
            yield {
                "type": "node",
                "id": node.get("id"),
                "label": _label(node),
                "chapter": node.get("chapter"),
                "section": node.get("section"),
                "subsection": node.get("subsection"),
                "page_start": node.get("page_start"),
                "page_end": node.get("page_end"),
                "bullets": bullets,
                "excerpt": node.get("text", "")[:600],
            }
            node_summaries.append(
                f"### {_label(node)} (tr.{node.get('page_start')}-{node.get('page_end')})\n"
                + "\n".join(f"- {bullet}" for bullet in bullets)
            )

        if dirty:
            save_summary_cache(cache)

        # --- REDUCE (stream) ---
        yield {"type": "status", "message": "Đang tổng hợp bản tóm tắt..."}
        joined = "\n\n".join(node_summaries)
        messages = [
            {
                "role": "system",
                "content": (
                    "Bạn là trợ giảng tóm tắt bài học Kinh tế chính trị Mác - Lênin. "
                    "Chỉ dùng thông tin trong các bản rút gọn được cung cấp, không thêm kiến thức ngoài. "
                    "Giữ trung lập, phân biệt độc quyền tư nhân / nhà nước / tự nhiên nếu có."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Dưới đây là các bản rút gọn theo từng mục của {scope_label}:\n\n{joined}\n\n"
                    f"Hãy viết một bản tóm tắt mạch lạc bằng tiếng Việt. Yêu cầu độ sâu: {depth_instruction}\n"
                    "Có thể ghi kèm trang (tr.x) khi hữu ích."
                ),
            },
        ]
        for chunk in _llm(messages, stream=True):
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield {"type": "token", "token": delta}

        yield {"type": "done"}
    except Exception as exc:  # noqa: BLE001 - surfaced to UI via SSE
        yield {"type": "error", "message": str(exc)}

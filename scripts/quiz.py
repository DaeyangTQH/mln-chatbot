"""Sinh quiz/flashcard từ cây cấu trúc + chấm câu tự luận.

- generate_quiz(scope, n): sinh câu hỏi từ các node giáo trình trong phạm vi.
  Mỗi câu giữ lại node nguồn (id, trang, text) để chấm & trích dẫn.
- grade_answer(question, user_answer, source_text): chấm câu tự luận bằng cách
  so đáp án người dùng với đoạn nguồn gốc.

Hiện sinh on-demand cho dễ test; có thể pre-generate + cache tĩnh sau (xem CLAUDE.md).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts import config
from scripts.summarize import _label, _nodes_for_scope  # tái dùng helper


def _llm_json(messages: list[dict[str, str]]) -> dict[str, Any]:
    client = config.get_openai_client()
    response = config.with_backoff(
        lambda: client.chat.completions.create(
            model=config.GENERATION_MODEL,
            temperature=0.3,
            response_format={"type": "json_object"},
            messages=messages,
        )
    )
    try:
        return json.loads(response.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        return {}


def _verify_mcq(question: str, options: list[str], answer_index: int, source_text: str) -> dict[str, Any]:
    """Check that an MCQ has one source-grounded answer and plausible distractors."""

    return _llm_json(
        [
            {
                "role": "system",
                "content": (
                    "You verify Vietnamese study MCQs. Check exactly one option is correct, "
                    "the distractors are plausible but false, and the answer follows only from the source. "
                    'Return JSON: {"ok":true|false,"reason":"..."}.'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\nOptions: {json.dumps(options, ensure_ascii=False)}\n"
                    f"Chosen answer index: {answer_index}\n\nSource:\n{source_text[:5000]}"
                ),
            },
        ]
    )


def _cache_load(path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("items", {}) if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _cache_save(path, items: dict[str, Any]) -> None:
    config.ensure_output_dir()
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps({"items": items}, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_path, path)


def _flashcard_node_hash(node: dict[str, Any]) -> str:
    text = node.get("text", "")
    payload = f"flashcards-v1|{config.GENERATION_MODEL}|{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate_quiz(scope: str = "Chương 4", n: int = 5) -> dict[str, Any]:
    """Sinh n câu hỏi (trộn trắc nghiệm + tự luận ngắn) từ phạm vi chọn."""

    nodes = _nodes_for_scope(scope)
    if not nodes:
        return {"questions": [], "error": f"Không có nội dung cho phạm vi '{scope}'."}

    # Chọn tối đa n node (mỗi node -> 1 câu) để bám sát cây và giữ nguồn rõ ràng.
    random.shuffle(nodes)
    picked = nodes[: max(1, min(n, len(nodes)))]

    questions: list[dict[str, Any]] = []
    for idx, node in enumerate(picked):
        kind = "mcq" if idx % 2 == 0 else "short"
        text = node.get("text", "")[:5000]
        if kind == "mcq":
            instruction = (
                'Sinh 1 câu trắc nghiệm 4 lựa chọn dựa CHÍNH XÁC vào đoạn dưới. '
                'Trả JSON: {"question":"...","options":["A","B","C","D"],"answer_index":0,"explanation":"..."}'
            )
        else:
            instruction = (
                'Sinh 1 câu hỏi tự luận ngắn dựa vào đoạn dưới. '
                'Trả JSON: {"question":"...","key_points":["ý1","ý2"],"model_answer":"..."}'
            )
        data = _llm_json(
            [
                {"role": "system", "content": "Bạn ra đề ôn tập Kinh tế chính trị Mác - Lênin, bám sát nguồn, không bịa."},
                {"role": "user", "content": f"Mục: {_label(node)}\n\nĐoạn nguồn:\n{text}\n\n{instruction}"},
            ]
        )
        if not data.get("question"):
            continue
        verified = kind != "mcq" or not config.QUIZ_VERIFY
        if kind == "mcq" and config.QUIZ_VERIFY:
            options = data.get("options") if isinstance(data.get("options"), list) else []
            answer_index = data.get("answer_index")
            if len(options) == 4 and isinstance(answer_index, int) and 0 <= answer_index < len(options):
                check = _verify_mcq(data["question"], options, answer_index, text)
                verified = bool(check.get("ok"))
            else:
                verified = False
            for _ in range(config.QUIZ_VERIFY_RETRIES):
                if verified:
                    break
                reason = str(check.get("reason", "Ensure one source-grounded correct option.")) if "check" in locals() else "Ensure one source-grounded correct option."
                retry = _llm_json(
                    [
                        {"role": "system", "content": "Create one precise Vietnamese MCQ from the source only. Return valid JSON with four options and one answer_index."},
                        {"role": "user", "content": f"Section: {_label(node)}\n\nSource:\n{text}\n\n{instruction}\n\nPrevious issue: {reason}"},
                    ]
                )
                retry_options = retry.get("options") if isinstance(retry.get("options"), list) else []
                retry_answer = retry.get("answer_index")
                if not retry.get("question") or len(retry_options) != 4 or not isinstance(retry_answer, int) or not 0 <= retry_answer < 4:
                    continue
                data = retry
                check = _verify_mcq(data["question"], retry_options, retry_answer, text)
                verified = bool(check.get("ok"))
        questions.append(
            {
                "id": f"q{idx+1}",
                "kind": kind,
                "question": data.get("question"),
                "options": data.get("options"),
                "answer_index": data.get("answer_index"),
                "explanation": data.get("explanation"),
                "key_points": data.get("key_points"),
                "model_answer": data.get("model_answer"),
                "source_label": _label(node),
                "source_node_id": node.get("id"),
                "page_start": node.get("page_start"),
                "page_end": node.get("page_end"),
                "source_text": text,
                "verified": verified,
            }
        )

    return {"scope": scope, "questions": questions}


def generate_flashcards(scope: str = "Chương 4", n: int = 8) -> dict[str, Any]:
    """Generate source-grounded term cards and cache them per textbook node."""

    nodes = _nodes_for_scope(scope)
    if not nodes:
        return {"scope": scope, "cards": [], "error": f"No content for scope '{scope}'."}

    random.shuffle(nodes)
    cache = _cache_load(config.FLASHCARDS_CACHE_PATH)
    dirty = False
    cards: list[dict[str, Any]] = []
    for node in nodes:
        if len(cards) >= n:
            break
        key = _flashcard_node_hash(node)
        node_cards = cache.get(key)
        if not isinstance(node_cards, list):
            text = node.get("text", "")[:5000]
            data = _llm_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "Create concise Vietnamese study flashcards from the supplied textbook text only. "
                            'Return JSON: {"cards":[{"term":"...","definition":"...","example":"..."}]}. '
                            "Definitions must be short and source-grounded."
                        ),
                    },
                    {"role": "user", "content": f"Section: {_label(node)}\n\nSource:\n{text}"},
                ]
            )
            node_cards = data.get("cards") if isinstance(data.get("cards"), list) else []
            cache[key] = node_cards
            dirty = True

        for raw in node_cards:
            if len(cards) >= n or not isinstance(raw, dict):
                break
            term = str(raw.get("term", "")).strip()
            definition = str(raw.get("definition", "")).strip()
            if not term or not definition:
                continue
            term_hash = hashlib.sha1(term.encode("utf-8")).hexdigest()[:8]
            cards.append(
                {
                    "id": f"fc_{node.get('id')}_{term_hash}",
                    "term": term,
                    "definition": definition,
                    "example": str(raw.get("example", "")).strip(),
                    "source_label": _label(node),
                    "source_node_id": node.get("id"),
                    "page_start": node.get("page_start"),
                    "page_end": node.get("page_end"),
                }
            )

    if dirty:
        _cache_save(config.FLASHCARDS_CACHE_PATH, cache)
    return {"scope": scope, "cards": cards}


def grade_answer(question: str, user_answer: str, source_text: str, key_points: list[str] | None = None) -> dict[str, Any]:
    """Chấm câu tự luận: so với đoạn nguồn + các ý chính mong đợi."""

    kp = "\n".join(f"- {p}" for p in (key_points or [])) or "(không có)"
    data = _llm_json(
        [
            {
                "role": "system",
                "content": (
                    "Bạn chấm bài tự luận môn Kinh tế chính trị Mác - Lênin. "
                    "Chỉ căn cứ vào đoạn nguồn và các ý chính; công bằng, có dẫn chứng. "
                    'Trả JSON: {"score":0-10,"verdict":"đúng|một phần|sai","feedback":"...","missing":["..."]}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Câu hỏi: {question}\n\n"
                    f"Đoạn nguồn (chuẩn):\n{source_text[:5000]}\n\n"
                    f"Ý chính cần có:\n{kp}\n\n"
                    f"Bài làm của học viên:\n{user_answer}\n\n"
                    "Hãy chấm điểm 0-10 và nêu điểm thiếu."
                ),
            },
        ]
    )
    return {
        "score": data.get("score"),
        "verdict": data.get("verdict"),
        "feedback": data.get("feedback", ""),
        "missing": data.get("missing", []),
    }

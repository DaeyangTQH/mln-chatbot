"""Khớp câu hỏi người dùng với ngân hàng câu hỏi ôn tập (quiz QA-bank).

Ý tưởng: bộ câu hỏi ôn tập trong `output/rag/quiz_qa.json` là các cặp
(câu hỏi ôn tập -> đáp án chuẩn đã kiểm duyệt). Khi người dùng gõ MỘT CÂU gần
giống đề bài (kể cả diễn đạt lại), ta muốn trả lời dựa trên đúng đáp án chuẩn đó
thay vì để RAG tự sinh (dễ chọn nhầm với câu trắc nghiệm).

Cách khớp: nhúng (embed) phần ĐỀ BÀI của từng câu bằng cùng EMBEDDING_MODEL của
hệ thống, rồi so cosine với câu người dùng. Chỉ khớp khi cosine >= ngưỡng CAO
(config.QUIZ_MATCH_MIN) để tránh nuốt nhầm câu học thông thường.

Embeddings của bank được tính MỘT LẦN rồi cache trong bộ nhớ (@lru_cache) — bank
chỉ vài chục câu nên chỉ tốn một lần gọi embeddings lúc khởi động, KHÔNG cần chạm
tới build.py / ingest.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts import config


@dataclass
class QuizMatch:
    item: dict[str, Any]
    score: float


@lru_cache(maxsize=1)
def load_quiz_bank(path: Path = config.QUIZ_QA_PATH) -> list[dict[str, Any]]:
    """Đọc ngân hàng câu hỏi ôn tập. Trả [] nếu file không tồn tại (tính năng tắt mềm)."""

    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [item for item in data if item.get("question") and item.get("answer")]


@lru_cache(maxsize=1)
def _bank_embeddings() -> tuple[tuple[dict[str, Any], ...], np.ndarray]:
    """Nhúng phần đề bài của cả bank (một lần), trả về (items, ma trận đã chuẩn hóa).

    Ma trận rỗng shape (0, 0) nếu bank trống — caller phải kiểm tra trước khi dùng.
    """

    bank = load_quiz_bank()
    if not bank:
        return tuple(), np.zeros((0, 0), dtype=np.float32)

    client = config.get_openai_client()
    questions = [item["question"] for item in bank]
    response = config.with_backoff(
        lambda: client.embeddings.create(model=config.EMBEDDING_MODEL, input=questions)
    )
    matrix = np.array([row.embedding for row in response.data], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return tuple(bank), matrix / norms


def match(question: str) -> QuizMatch | None:
    """Trả về QuizMatch nếu câu hỏi khớp một câu trong bank ở mức >= QUIZ_MATCH_MIN.

    Fail-open theo hướng AN TOÀN cho pipeline: bất kỳ lỗi nào (bank trống, lỗi
    embeddings...) -> None, tức là để câu hỏi đi tiếp qua RAG như bình thường.
    """

    if not config.QUIZ_MATCH_ENABLED:
        return None
    q = (question or "").strip()
    if not q:
        return None

    try:
        items, matrix = _bank_embeddings()
        if matrix.shape[0] == 0:
            return None

        client = config.get_openai_client()
        response = config.with_backoff(
            lambda: client.embeddings.create(model=config.EMBEDDING_MODEL, input=q)
        )
        query = np.array(response.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm == 0:
            return None
        scores = matrix @ (query / norm)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        if best_score >= config.QUIZ_MATCH_MIN:
            return QuizMatch(item=dict(items[best_idx]), score=best_score)
        return None
    except Exception:  # noqa: BLE001 - fail open: lỗi -> đi tiếp qua RAG
        return None

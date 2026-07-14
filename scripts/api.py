"""FastAPI bridge for the MLN121 RAG chatbot."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts import config, guard, quiz_match
from scripts.chat import (
    build_messages,
    build_quiz_messages,
    call_llm,
    in_scope,
    quiz_answer_prefix,
    sources_for_answer,
    stream_without_sources,
    temperature_for_mode,
)
from scripts.quiz import generate_flashcards, generate_quiz, grade_answer
from scripts.retrieve import load_chunks, load_parents, retrieve_with_relevance
from scripts.summarize import list_scopes, summarize_stream


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[ChatMessage] = Field(default_factory=list)
    mode: str = "default"
    pinned: str | None = Field(default=None, max_length=4000)


class SummarizeRequest(BaseModel):
    scope: str = "all"
    depth: Literal["short", "detailed"] = "short"


class QuizRequest(BaseModel):
    scope: str = "Chương 4"
    n: int = Field(default=5, ge=1, le=10)


class FlashcardRequest(BaseModel):
    scope: str = "Chương 4"
    n: int = Field(default=8, ge=1, le=20)


class GradeRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    user_answer: str = Field(min_length=1, max_length=6000)
    source_text: str = Field(min_length=1, max_length=8000)
    key_points: list[str] = Field(default_factory=list)


app = FastAPI(title="MLN121 RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ALLOW_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def sse(payload: dict, event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/api/health")
def health() -> dict:
    chunks = load_chunks()
    parents = load_parents()
    return {
        "ok": True,
        "chunks": len(chunks),
        "parents": len(parents),
    }


@app.post("/api/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    def event_stream() -> Iterator[str]:
        contexts = []
        try:
            history = [item.model_dump() for item in request.history[-6:]]

            # 1) Chốt chặn đầu vào (injection/rỗng/mơ hồ) trước khi tốn retrieval.
            if config.TRIAGE_ENABLED:
                verdict = guard.triage(request.question, history=history, pinned=request.pinned)
                if verdict:
                    yield sse({"type": "status", "message": "guard"})
                    yield sse({"type": "token", "token": verdict["text"]})
                    yield sse({"type": "done"}, event="done")
                    return

            # 1b) Câu hỏi vui/meta về chatbot & website -> trả lời cố định, bỏ qua RAG.
            meta = guard.meta_response(request.question)
            if meta:
                yield sse({"type": "status", "message": "guard"})
                yield sse({"type": "token", "token": meta["text"]})
                yield sse({"type": "done"}, event="done")
                return

            # 1c) Câu khớp bộ câu hỏi ôn tập -> trả lời dựa trên ĐÁP ÁN CHUẨN, bỏ qua RAG.
            #     Đặt TRƯỚC retrieval/cổng relevance để câu ôn tập không bị chặn nhầm.
            hit = quiz_match.match(request.question)
            if hit:
                yield sse({"type": "status", "message": "quiz"})
                # Dòng đáp án IN ĐẬM (dựng bằng code) phát trước, rồi mới stream giải thích.
                yield sse({"type": "token", "token": quiz_answer_prefix(hit.item) + "\n\n"})
                stream = call_llm(build_quiz_messages(hit.item), stream=True, temperature=0.2)
                for token in stream_without_sources(stream):
                    yield sse({"type": "token", "token": token})
                yield sse({"type": "done"}, event="done")
                return

            yield sse({"type": "status", "message": "retrieving"})
            # Pinned slide text steers retrieval too, so the corpus passages we
            # fetch are about the exact sentence the user highlighted.
            retrieval_query = request.question
            if request.pinned and request.pinned.strip():
                retrieval_query = f"{request.question}\n{request.pinned.strip()}"
            contexts, relevance = retrieve_with_relevance(retrieval_query)

            # 2) Cổng liên quan: corpus nhỏ luôn trả top_k, nên nếu không đoạn nào
            # đủ gần thì từ chối thay vì để model biến chunk ngẫu nhiên thành bài giảng.
            if config.ABSTAIN_ENABLED and relevance < config.RELEVANCE_MIN:
                yield sse({"type": "status", "message": "abstain"})
                yield sse({"type": "token", "token": guard.OUT_OF_CORPUS_TEXT})
                yield sse({"type": "done"}, event="done")
                return

            # 2b) Vùng xám: embedding không tách được câu cùng miền nhưng ngoài chủ đề
            # (vd giá trị thặng dư). Nhờ classifier đọc ngữ cảnh để quyết định (P3).
            if (
                config.ABSTAIN_ENABLED
                and config.SCOPE_CLASSIFIER_ENABLED
                and relevance < config.RELEVANCE_MAX
                and not in_scope(request.question, contexts)
            ):
                yield sse({"type": "status", "message": "abstain"})
                yield sse({"type": "token", "token": guard.OUT_OF_TOPIC_TEXT})
                yield sse({"type": "done"}, event="done")
                return

            messages = build_messages(
                request.question,
                contexts,
                history=history,
                mode=request.mode,
                pinned=request.pinned,
            )
            yield sse({"type": "status", "message": "streaming"})
            chunks: list[str] = []
            # stream_without_sources cắt bỏ mục "Nguồn" mà model có thể tự viết, để chỉ
            # còn duy nhất danh sách nguồn do code dựng ở dưới (tránh hai mục trùng nhau).
            stream = call_llm(messages, stream=True, temperature=temperature_for_mode(request.mode))
            for token in stream_without_sources(stream):
                chunks.append(token)
                yield sse({"type": "token", "token": token})
            # 3) Mục "Nguồn" chỉ gồm nguồn sơ cấp thực; nếu rỗng (câu từ chối hoặc không
            # còn nguồn nào sau khi lọc bản tổng hợp) thì BỎ LUÔN mục Nguồn.
            sources = sources_for_answer("".join(chunks), contexts)
            if sources:
                yield sse({"type": "token", "token": "\n\n" + sources})
            yield sse({"type": "done"}, event="done")
        except Exception as exc:  # noqa: BLE001 - send API errors as SSE so the UI can render them
            yield sse({"type": "error", "message": str(exc)}, event="error")

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/scopes")
def scopes() -> dict:
    return {"scopes": list_scopes()}


@app.post("/api/summarize")
def summarize(request: SummarizeRequest) -> StreamingResponse:
    def event_stream() -> Iterator[str]:
        for event in summarize_stream(scope=request.scope, depth=request.depth):
            evt = event.get("type")
            name = evt if evt in ("done", "error") else None
            yield sse(event, event=name)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/quiz")
def quiz(request: QuizRequest) -> dict:
    return generate_quiz(scope=request.scope, n=request.n)


@app.post("/api/flashcards")
def flashcards(request: FlashcardRequest) -> dict:
    return generate_flashcards(scope=request.scope, n=request.n)


@app.post("/api/grade")
def grade(request: GradeRequest) -> dict:
    return grade_answer(
        question=request.question,
        user_answer=request.user_answer,
        source_text=request.source_text,
        key_points=request.key_points,
    )

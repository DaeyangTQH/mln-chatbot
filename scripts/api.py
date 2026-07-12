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

from scripts import config
from scripts.chat import build_messages, call_llm, temperature_for_mode, verify_citations
from scripts.quiz import generate_flashcards, generate_quiz, grade_answer
from scripts.retrieve import format_source, hybrid_retrieve, load_chunks, load_parents
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
        answer_parts: list[str] = []
        try:
            yield sse({"type": "status", "message": "retrieving"})
            # Pinned slide text steers retrieval too, so the corpus passages we
            # fetch are about the exact sentence the user highlighted.
            retrieval_query = request.question
            if request.pinned and request.pinned.strip():
                retrieval_query = f"{request.question}\n{request.pinned.strip()}"
            contexts = hybrid_retrieve(retrieval_query)
            history = [item.model_dump() for item in request.history[-6:]]
            messages = build_messages(
                request.question,
                contexts,
                history=history,
                mode=request.mode,
                pinned=request.pinned,
            )
            yield sse({"type": "status", "message": "streaming"})
            for token in call_llm(messages, stream=True, temperature=temperature_for_mode(request.mode)):
                answer_parts.append(token)
                yield sse({"type": "token", "token": token})

            answer_text = "".join(answer_parts)
            sources = [
                {
                    "label": format_source(context),
                    "source_type": context.get("source_type"),
                    "source_url": context.get("source_url"),
                    "page_start": context.get("page_start"),
                    "page_end": context.get("page_end"),
                }
                for context in contexts
            ]
            warnings = verify_citations(answer_text, contexts)
            yield sse({"type": "sources", "sources": sources, "warnings": warnings}, event="sources")
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

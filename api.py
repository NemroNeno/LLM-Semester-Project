from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Path, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from db import get_db, init_db, list_chat_messages, list_chats, should_summarize_next_turn
from rag import CHAT_PROVIDER, OLLAMA_CHAT_MODEL, OPENROUTER_MODEL, invoke_graph, message_to_text
from settings import TEMPLATES_DIRECTORY, THREAD_ID

TEMPLATES = Jinja2Templates(directory=TEMPLATES_DIRECTORY)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    chat_id: str | None = None
    provider: str = Field(default=CHAT_PROVIDER)


class ChatWillSummarizeRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    chat_id: str | None = None


class ChatWillSummarizeResponse(BaseModel):
    will_summarize: bool


class ChatResponse(BaseModel):
    reply: str
    context_count: int
    rag_references: list[str] = Field(default_factory=list)
    provider: str
    chat_id: str


app = FastAPI(title="LLM Semester Project")


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/chats")
def chats() -> list[dict[str, str]]:
    return list_chats()


@app.get("/chats/{chat_id}/messages")
def chat_messages(chat_id: str = Path(...)) -> list[dict[str, Any]]:
    return list_chat_messages(chat_id)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return TEMPLATES.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "chat_provider": CHAT_PROVIDER,
            "chat_model": OPENROUTER_MODEL if CHAT_PROVIDER == "openrouter" else OLLAMA_CHAT_MODEL,
            "thread_id": THREAD_ID,
        },
    )


@app.post("/chat/will-summarize", response_model=ChatWillSummarizeResponse)
def chat_will_summarize(payload: ChatWillSummarizeRequest) -> ChatWillSummarizeResponse:
    return ChatWillSummarizeResponse(
        will_summarize=should_summarize_next_turn(payload.chat_id, payload.message)
    )


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    chat_id = payload.chat_id or str(uuid.uuid4())
    conn = get_db()
    if not payload.chat_id:
        title = payload.message[:30] + "..." if len(payload.message) > 30 else payload.message
        conn.execute("INSERT INTO chats (id, title) VALUES (?, ?)", (chat_id, title))
        conn.commit()

    conn.execute(
        "INSERT INTO messages (chat_id, role, content, context_count, rag_references) VALUES (?, ?, ?, ?, ?)",
        (chat_id, "user", payload.message, 0, json.dumps([])),
    )
    conn.commit()

    try:
        state = invoke_graph(payload.message, thread_id=chat_id, provider=payload.provider)
    except RuntimeError as exc:
        conn.close()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Chat invocation failed: {exc}") from exc

    messages = state.get("messages", [])
    if not messages:
        conn.close()
        raise HTTPException(status_code=500, detail="Graph returned no messages.")

    reply = message_to_text(messages[-1])
    rag_references = state.get("context", [])
    context_count = len(rag_references)

    conn.execute(
        "INSERT INTO messages (chat_id, role, content, context_count, rag_references) VALUES (?, ?, ?, ?, ?)",
        (chat_id, "assistant", reply, context_count, json.dumps(rag_references)),
    )
    conn.commit()
    conn.close()

    return ChatResponse(
        reply=reply,
        context_count=context_count,
        rag_references=rag_references,
        provider=payload.provider,
        chat_id=chat_id,
    )
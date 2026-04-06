from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import FastAPI, File, HTTPException, Path, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from db import get_db, init_db, list_chat_messages, list_chats, should_summarize_next_turn
from document_ingestion import delete_document, get_document_status, ingest_uploaded_document, list_documents
from rag import CHAT_PROVIDER, OLLAMA_CHAT_MODEL, OLLAMA_FINETUNED_CHAT_MODEL, OPENROUTER_MODEL, invoke_graph, message_to_text, normalize_provider
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
    kg_references: list[str] = Field(default_factory=list)
    provider: str
    chat_id: str


class DocumentUploadResponse(BaseModel):
    document_id: str
    job_id: str


class DocumentDeleteResponse(BaseModel):
    deleted: bool


class DocumentStatusResponse(BaseModel):
    document: dict[str, Any]
    job: dict[str, Any] | None = None


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
    if CHAT_PROVIDER == "openrouter":
        display_model = OPENROUTER_MODEL
    elif CHAT_PROVIDER == "ollama-finetuned":
        display_model = OLLAMA_FINETUNED_CHAT_MODEL
    else:
        display_model = OLLAMA_CHAT_MODEL

    return TEMPLATES.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "chat_provider": CHAT_PROVIDER,
            "chat_model": display_model,
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
    selected_provider = normalize_provider(payload.provider)
    conn = get_db()
    if not payload.chat_id:
        title = payload.message[:30] + "..." if len(payload.message) > 30 else payload.message
        conn.execute("INSERT INTO chats (id, title) VALUES (?, ?)", (chat_id, title))
        conn.commit()

    conn.execute(
        "INSERT INTO messages (chat_id, role, content, context_count, rag_references, kg_references) VALUES (?, ?, ?, ?, ?, ?)",
        (chat_id, "user", payload.message, 0, json.dumps([]), json.dumps([])),
    )
    conn.commit()

    try:
        state = invoke_graph(payload.message, thread_id=chat_id, provider=selected_provider)
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
    kg_references = state.get("kg_context", [])
    context_count = len(rag_references)

    conn.execute(
        "INSERT INTO messages (chat_id, role, content, context_count, rag_references, kg_references) VALUES (?, ?, ?, ?, ?, ?)",
        (chat_id, "assistant", reply, context_count, json.dumps(rag_references), json.dumps(kg_references)),
    )
    conn.commit()
    conn.close()

    return ChatResponse(
        reply=reply,
        context_count=context_count,
        rag_references=rag_references,
        kg_references=kg_references,
        provider=selected_provider,
        chat_id=chat_id,
    )


@app.get("/admin/documents")
def admin_list_documents() -> list[dict[str, Any]]:
    return list_documents()


@app.get("/admin/documents/{document_id}", response_model=DocumentStatusResponse)
def admin_get_document(document_id: str) -> DocumentStatusResponse:
    status = get_document_status(document_id)
    return DocumentStatusResponse(document=status["document"], job=status["job"])


@app.post("/admin/documents", response_model=DocumentUploadResponse)
def admin_upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    result = ingest_uploaded_document(file)
    return DocumentUploadResponse(document_id=result.document_id, job_id=result.job_id)


@app.delete("/admin/documents/{document_id}", response_model=DocumentDeleteResponse)
def admin_delete_document(document_id: str) -> DocumentDeleteResponse:
    delete_document(document_id)
    return DocumentDeleteResponse(deleted=True)
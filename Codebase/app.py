import os
import re
import sqlite3
import uuid
from functools import lru_cache
from typing import Any, Sequence, cast

from fastapi import FastAPI, HTTPException, Request, Path
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

load_dotenv()

CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "banking_qa")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
CHAT_PROVIDER = os.getenv("CHAT_PROVIDER", "openrouter").lower()
OPENROUTER_MODEL = os.getenv("CHAT_MODEL", "qwen/qwen-2.5-7b-instruct")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:3b-instruct")
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))
MAX_MESSAGE_TOKENS = int(os.getenv("MAX_MESSAGE_TOKENS", "1200"))
THREAD_ID = os.getenv("LANGGRAPH_THREAD_ID", "global_session")
TEMPLATES = Jinja2Templates(directory="templates")

ASSISTANT_IDENTITY = (
    "I am the NUST banking assistant. I answer questions using the NUST banking "
    "Q&A data available in this system."
)

SYSTEM_PROMPT_TEMPLATE = """You are the NUST banking assistant.
If the user asks who you are, what your name is, or what system they are using, identify yourself as the NUST banking assistant.
Do not describe yourself as Qwen, OpenAI, Alibaba Cloud, OpenRouter, or a generic large language model.
Answer only from the retrieved context below.
If the context is missing, weak, or does not contain the answer, say: I do not have enough information in the provided banking data to answer that.
Do not invent policies, fees, requirements, or product details.
Keep the answer concise and directly relevant to the user's question.

Retrieved context:
{context}
"""

FALLBACK_ANSWER = "I do not have enough information in the provided banking data to answer that."


class State(MessagesState):
    context: list[str]
    trimmed_messages: list[BaseMessage]


def message_to_text(message: BaseMessage) -> str:
    if isinstance(message.content, str):
        return message.content.strip()

    if isinstance(message.content, list):
        parts: list[str] = []
        for item in message.content:
            if isinstance(item, str):
                parts.append(item)
                continue

            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))

        return "\n".join(part.strip() for part in parts if part.strip())

    return str(message.content).strip()


def get_latest_user_text(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message_to_text(message)
    return ""


def is_identity_query(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    identity_patterns = (
        r"\bwho are you\b",
        r"\bwhat are you\b",
        r"\bwhat is your name\b",
        r"\byour name\b",
        r"\bintroduce yourself\b",
        r"\bwhat system is this\b",
        r"\bwho made you\b",
    )
    return any(re.search(pattern, normalized) for pattern in identity_patterns)


@lru_cache(maxsize=1)
def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
        validate_model_on_init=True,
    )


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIRECTORY,
    )


@lru_cache(maxsize=2)
def get_chat_model(provider: str) -> Any:
    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set OPENROUTER_API_KEY or OPENAI_API_KEY before invoking the chat graph."
            )

        headers: dict[str, str] = {}
        http_referer = os.getenv("OPENROUTER_HTTP_REFERER")
        app_title = os.getenv("OPENROUTER_APP_TITLE")
        if http_referer:
            headers["HTTP-Referer"] = http_referer
        if app_title:
            headers["X-Title"] = app_title

        chat_openai: Any = ChatOpenAI
        return chat_openai(
            model=OPENROUTER_MODEL,
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            temperature=0.0,
            default_headers=headers or None,
        )

    if provider == "ollama":
        chat_ollama: Any = ChatOllama
        return chat_ollama(
            model=OLLAMA_CHAT_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.0,
        )

    raise ValueError(f"Unsupported provider: {provider}")


def format_context(documents: list[Any]) -> list[str]:
    context_entries: list[str] = []
    for document in documents:
        sheet = document.metadata.get("sheet", "Unknown")
        source_file = document.metadata.get("source_file", "Unknown")
        context_entries.append(
            f"Sheet: {sheet}\nSource: {source_file}\n{document.page_content}"
        )
    return context_entries


def retrieve_node(state: State, *, vector_store: Chroma | None = None) -> dict[str, list[str]]:
    store = vector_store or get_vector_store()
    query = get_latest_user_text(state["messages"])
    if not query:
        return {"context": []}

    documents = store.similarity_search(query, k=RETRIEVAL_TOP_K)
    return {"context": format_context(documents)}


def trim_node(state: State) -> dict[str, list[BaseMessage]]:
    trimmed = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=MAX_MESSAGE_TOKENS,
        start_on="human",
        end_on=("human", "tool"),
    )
    return {"trimmed_messages": list(trimmed)}


def generate_node(state: State, *, chat_model: Any = None) -> dict[str, list[Any]]:
    trimmed_messages = state.get("trimmed_messages") or state["messages"]
    latest_user_text = get_latest_user_text(trimmed_messages)
    if is_identity_query(latest_user_text):
        return {"messages": [AIMessage(content=ASSISTANT_IDENTITY)]}

    context = state.get("context", [])
    if not context:
        return {"messages": [AIMessage(content=FALLBACK_ANSWER)]}

    if not chat_model:
        raise ValueError("chat_model must be provided")
    
    model = chat_model
    context_block = "\n\n".join(context) if context else "No relevant context retrieved."

    response = model.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT_TEMPLATE.format(context=context_block)),
            *trimmed_messages,
        ]
    )
    return {"messages": [response]}


def build_graph(*, chat_model: Any = None, vector_store: Chroma | None = None):
    builder = StateGraph(State)

    builder.add_node(
        "retrieve",
        lambda state: retrieve_node(cast(State, state), vector_store=vector_store),
    )
    builder.add_node("trim", trim_node)
    builder.add_node(
        "generate",
        lambda state: generate_node(cast(State, state), chat_model=chat_model),
    )
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "trim")
    builder.add_edge("trim", "generate")
    builder.add_edge("generate", END)

    return builder.compile(checkpointer=MemorySaver())


@lru_cache(maxsize=2)
def get_graph(provider: str):
    return build_graph(chat_model=get_chat_model(provider))


def invoke_graph(message: str, thread_id: str = THREAD_ID, provider: str = CHAT_PROVIDER, graph: Any = None) -> dict[str, Any]:
    compiled_graph = graph or get_graph(provider)
    return compiled_graph.invoke(
        cast(State, {"messages": [HumanMessage(content=message)]}),
        config={"configurable": {"thread_id": thread_id}},
    )


DB_PATH = "chat_history.db"

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    chat_id: str | None = None
    provider: str = Field(default=CHAT_PROVIDER)

class ChatResponse(BaseModel):
    reply: str
    context_count: int
    provider: str
    chat_id: str

app = FastAPI(title="LLM Semester Project")

@app.on_event("startup")
def startup():
    conn = get_db()
    conn.execute('CREATE TABLE IF NOT EXISTS chats (id TEXT PRIMARY KEY, title TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')
    conn.execute('CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id TEXT, role TEXT, content TEXT, context_count INTEGER, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')
    conn.commit()
    conn.close()

@app.get("/chats")
def list_chats():
    conn = get_db()
    chats = conn.execute("SELECT id, title FROM chats ORDER BY created_at DESC").fetchall()
    conn.close()
    return [{"id": c["id"], "title": c["title"]} for c in chats]

@app.get("/chats/{chat_id}/messages")
def list_chat_messages(chat_id: str = Path(...)):
    conn = get_db()
    messages = conn.execute("SELECT role, content, context_count FROM messages WHERE chat_id = ? ORDER BY created_at ASC", (chat_id,)).fetchall()
    conn.close()
    return [{"role": m["role"], "content": m["content"], "context_count": m["context_count"]} for m in messages]

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


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    chat_id = payload.chat_id or str(uuid.uuid4())
    conn = get_db()
    if not payload.chat_id:
        title = payload.message[:30] + "..." if len(payload.message) > 30 else payload.message
        conn.execute("INSERT INTO chats (id, title) VALUES (?, ?)", (chat_id, title))
        conn.commit()
    
    conn.execute("INSERT INTO messages (chat_id, role, content, context_count) VALUES (?, ?, ?, ?)", (chat_id, "user", payload.message, 0))
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
    context_count = len(state.get("context", []))
    
    conn.execute("INSERT INTO messages (chat_id, role, content, context_count) VALUES (?, ?, ?, ?)", (chat_id, "assistant", reply, context_count))
    conn.commit()
    conn.close()

    return ChatResponse(
        reply=reply,
        context_count=context_count,
        provider=CHAT_PROVIDER,
        chat_id=chat_id
    )
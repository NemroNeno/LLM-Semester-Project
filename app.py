import os
import re
import sqlite3
import uuid
import json
from functools import lru_cache
from pathlib import Path as FilePath
from typing import Any, Sequence, cast

from fastapi import FastAPI, HTTPException, Request, Path
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
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
QA_DIRECTORY = FilePath(os.getenv("QA_DIRECTORY", "sheets_qa"))
CHAT_PROVIDER = os.getenv("CHAT_PROVIDER", "openrouter").lower()
OPENROUTER_MODEL = os.getenv("CHAT_MODEL", "qwen/qwen-2.5-7b-instruct")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:3b-instruct")
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))
RETRIEVAL_FETCH_K = int(os.getenv("RETRIEVAL_FETCH_K", "24"))
RETRIEVAL_MMR_FETCH_K = int(os.getenv("RETRIEVAL_MMR_FETCH_K", "40"))
RETRIEVAL_LEXICAL_FETCH_K = int(os.getenv("RETRIEVAL_LEXICAL_FETCH_K", "24"))
RETRIEVAL_BM25_FETCH_K = int(os.getenv("RETRIEVAL_BM25_FETCH_K", "24"))
MAX_MESSAGE_TOKENS = int(os.getenv("MAX_MESSAGE_TOKENS", "1200"))
THREAD_ID = os.getenv("LANGGRAPH_THREAD_ID", "global_session")
TEMPLATES = Jinja2Templates(directory="templates")

RERANK_DENSE_WEIGHT = float(os.getenv("RERANK_DENSE_WEIGHT", "0.45"))
RERANK_LEXICAL_WEIGHT = float(os.getenv("RERANK_LEXICAL_WEIGHT", "0.30"))
RERANK_RRF_WEIGHT = float(os.getenv("RERANK_RRF_WEIGHT", "0.15"))
RERANK_METADATA_WEIGHT = float(os.getenv("RERANK_METADATA_WEIGHT", "0.10"))
RERANK_BM25_WEIGHT = float(os.getenv("RERANK_BM25_WEIGHT", "0.10"))
RERANK_RRF_SCALE = float(os.getenv("RERANK_RRF_SCALE", "30.0"))
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "0.22"))

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
Format every answer in polished Markdown.
Use this response style:
- Keep the answer concise and directly relevant to the question.
- Start with a short heading: "### Answer".
- Use bullet points for multiple facts or steps.
- Use a table when comparing values, options, or requirements.
- Emphasize key terms with bold text.
- Keep paragraphs short and readable.
- End with a "### Quick Summary" section with 1-2 lines.
Do not include markdown code fences unless the user explicitly asks for code.
Keep the answer concise, directly relevant, and pleasant to read.

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


def build_page_content(question: str, answer: str) -> str:
    answer_text = answer if answer else "No answer provided in source data."
    return f"Question: {question}\nAnswer: {answer_text}"


@lru_cache(maxsize=1)
def get_qa_corpus_documents() -> list[Document]:
    if not QA_DIRECTORY.exists():
        return []

    documents: list[Document] = []
    for json_file in sorted(QA_DIRECTORY.glob("qa_*.json")):
        rows = json.loads(json_file.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            continue

        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue

            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()
            sheet = str(row.get("sheet", "")).strip()
            if not question or not sheet:
                continue

            documents.append(
                Document(
                    page_content=build_page_content(question, answer),
                    metadata={
                        "sheet": sheet,
                        "source_file": json_file.name,
                        "record_index": index,
                        "question": question,
                        "has_answer": bool(answer),
                    },
                )
            )

    return documents


@lru_cache(maxsize=1)
def get_bm25_retriever() -> Any | None:
    corpus = get_qa_corpus_documents()
    if not corpus:
        return None

    try:
        from langchain_community.retrievers import BM25Retriever
    except ImportError:
        return None

    retriever = BM25Retriever.from_documents(corpus)
    retriever.k = max(RETRIEVAL_BM25_FETCH_K, RETRIEVAL_TOP_K * 3)
    return retriever


def tokenize_for_rerank(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9%./-]+", text.lower()) if token}


def lexical_overlap_score(query: str, content: str) -> float:
    query_tokens = tokenize_for_rerank(query)
    if not query_tokens:
        return 0.0

    content_tokens = tokenize_for_rerank(content)
    if not content_tokens:
        return 0.0

    overlap = query_tokens & content_tokens
    coverage = len(overlap) / len(query_tokens)

    # Banking queries often hinge on exact numerics (limits, tenures, rates).
    numeric_tokens = {token for token in query_tokens if re.search(r"\d", token)}
    numeric_overlap = len(numeric_tokens & content_tokens) / max(1, len(numeric_tokens))

    return min(1.0, (coverage * 0.8) + (numeric_overlap * 0.2))


def metadata_match_score(query: str, document: Any) -> float:
    metadata = getattr(document, "metadata", {}) or {}
    question = str(metadata.get("question", ""))
    sheet = str(metadata.get("sheet", ""))
    source_file = str(metadata.get("source_file", ""))
    metadata_text = f"{question} {sheet} {source_file}".lower()

    query_tokens = tokenize_for_rerank(query)
    if not query_tokens or not metadata_text:
        return 0.0

    hits = sum(1 for token in query_tokens if token in metadata_text)
    return min(1.0, hits / max(1, len(query_tokens)))


def corpus_question_match_score(query: str, document: Any) -> float:
    metadata = getattr(document, "metadata", {}) or {}
    question_text = str(metadata.get("question", "")).lower()
    if not question_text:
        return 0.0

    normalized_query = tokenize_for_rerank(query)
    normalized_question = tokenize_for_rerank(question_text)
    if not normalized_query or not normalized_question:
        return 0.0

    overlap = normalized_query & normalized_question
    if not overlap:
        return 0.0

    coverage = len(overlap) / len(normalized_query)
    exact_bonus = 1.0 if query.strip().lower() == question_text.strip() else 0.0
    return min(1.0, (coverage * 0.7) + (exact_bonus * 0.3))


def normalize_dense_score(score: float) -> float:
    if 0.0 <= score <= 1.0:
        return score
    if score > 1.0:
        return 1.0 / (1.0 + score)
    return max(0.0, min(1.0, 1.0 + score))


def safe_dense_candidates(store: Chroma, query: str, fetch_k: int) -> list[tuple[Any, float]]:
    try:
        docs_with_scores = store.similarity_search_with_relevance_scores(query, k=fetch_k)
        return [(doc, normalize_dense_score(float(score))) for doc, score in docs_with_scores]
    except Exception:
        docs = store.similarity_search(query, k=fetch_k)
        return [(doc, max(0.0, 1.0 - (index * 0.05))) for index, doc in enumerate(docs)]


def safe_mmr_candidates(store: Chroma, query: str, top_k: int, fetch_k: int) -> list[Any]:
    try:
        return store.max_marginal_relevance_search(query, k=top_k, fetch_k=fetch_k)
    except Exception:
        return []


def safe_lexical_candidates(query: str, top_k: int) -> list[Any]:
    corpus = get_qa_corpus_documents()
    if not corpus:
        return []

    scored: list[tuple[float, Any]] = []
    for document in corpus:
        content_score = lexical_overlap_score(query, str(getattr(document, "page_content", "")))
        question_score = corpus_question_match_score(query, document)
        meta_score = metadata_match_score(query, document)
        final_score = (content_score * 0.45) + (question_score * 0.45) + (meta_score * 0.10)
        scored.append((final_score, document))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [document for score, document in scored[:top_k] if score > 0.0]


def safe_bm25_candidates(query: str, top_k: int) -> list[Any]:
    retriever = get_bm25_retriever()
    if not retriever:
        return []

    retriever.k = top_k
    try:
        return list(retriever.invoke(query))
    except Exception:
        return []


def document_key(document: Any) -> str:
    metadata = getattr(document, "metadata", {}) or {}
    source_file = str(metadata.get("source_file", ""))
    record_index = str(metadata.get("record_index", ""))
    sheet = str(metadata.get("sheet", ""))
    key = f"{source_file}:{record_index}:{sheet}"
    if key.strip(":"):
        return key
    return str(getattr(document, "page_content", ""))


def rerank_documents(query: str, dense_candidates: list[tuple[Any, float]], mmr_candidates: list[Any]) -> list[Any]:
    candidates: dict[str, dict[str, Any]] = {}

    for dense_rank, (document, dense_score) in enumerate(dense_candidates, start=1):
        key = document_key(document)
        bucket = candidates.setdefault(
            key,
            {
                "document": document,
                "dense_score": 0.0,
                "dense_rank": dense_rank,
                "mmr_rank": None,
                "lexical_rank": None,
                "bm25_rank": None,
            },
        )
        if dense_score > bucket["dense_score"]:
            bucket["dense_score"] = dense_score
        if dense_rank < bucket["dense_rank"]:
            bucket["dense_rank"] = dense_rank

    for mmr_rank, document in enumerate(mmr_candidates, start=1):
        key = document_key(document)
        bucket = candidates.setdefault(
            key,
            {
                "document": document,
                "dense_score": 0.0,
                "dense_rank": 10_000,
                "mmr_rank": mmr_rank,
                "lexical_rank": None,
                "bm25_rank": None,
            },
        )
        if bucket["mmr_rank"] is None or mmr_rank < bucket["mmr_rank"]:
            bucket["mmr_rank"] = mmr_rank

    lexical_candidates = safe_lexical_candidates(query, top_k=max(RETRIEVAL_LEXICAL_FETCH_K, RETRIEVAL_TOP_K * 3))
    for lexical_rank, document in enumerate(lexical_candidates, start=1):
        key = document_key(document)
        bucket = candidates.setdefault(
            key,
            {
                "document": document,
                "dense_score": 0.0,
                "dense_rank": 10_000,
                "mmr_rank": None,
                "lexical_rank": lexical_rank,
                "bm25_rank": None,
            },
        )
        if bucket["lexical_rank"] is None or lexical_rank < bucket["lexical_rank"]:
            bucket["lexical_rank"] = lexical_rank

    bm25_candidates = safe_bm25_candidates(query, top_k=max(RETRIEVAL_BM25_FETCH_K, RETRIEVAL_TOP_K * 3))
    for bm25_rank, document in enumerate(bm25_candidates, start=1):
        key = document_key(document)
        bucket = candidates.setdefault(
            key,
            {
                "document": document,
                "dense_score": 0.0,
                "dense_rank": 10_000,
                "mmr_rank": None,
                "lexical_rank": None,
                "bm25_rank": bm25_rank,
            },
        )
        if bucket["bm25_rank"] is None or bm25_rank < bucket["bm25_rank"]:
            bucket["bm25_rank"] = bm25_rank

    scored: list[tuple[float, Any]] = []
    rrf_k = 60.0
    for bucket in candidates.values():
        document = bucket["document"]
        dense_score = float(bucket["dense_score"])
        lexical_score = lexical_overlap_score(query, str(getattr(document, "page_content", "")))
        meta_score = metadata_match_score(query, document)

        rrf = 1.0 / (rrf_k + float(bucket["dense_rank"]))
        if bucket["mmr_rank"] is not None:
            rrf += 1.0 / (rrf_k + float(bucket["mmr_rank"]))
        if bucket["lexical_rank"] is not None:
            rrf += 1.0 / (rrf_k + float(bucket["lexical_rank"]))
        if bucket["bm25_rank"] is not None:
            rrf += 1.0 / (rrf_k + float(bucket["bm25_rank"]))
        rrf_score = min(1.0, rrf * RERANK_RRF_SCALE)

        bm25_rank = bucket.get("bm25_rank")
        bm25_score = 0.0
        if bm25_rank is not None:
            bm25_score = 1.0 / (1.0 + float(bm25_rank))

        final_score = (
            (dense_score * RERANK_DENSE_WEIGHT)
            + (lexical_score * RERANK_LEXICAL_WEIGHT)
            + (rrf_score * RERANK_RRF_WEIGHT)
            + (meta_score * RERANK_METADATA_WEIGHT)
            + (bm25_score * RERANK_BM25_WEIGHT)
        )
        scored.append((final_score, document))

    scored.sort(key=lambda item: item[0], reverse=True)
    filtered = [document for score, document in scored if score >= RERANK_MIN_SCORE]
    if filtered:
        return filtered[:RETRIEVAL_TOP_K]

    # Fall back to top scored docs if strict threshold removes everything.
    return [document for _, document in scored[:RETRIEVAL_TOP_K]]


def retrieve_node(state: State, *, vector_store: Chroma | None = None) -> dict[str, list[str]]:
    store = vector_store or get_vector_store()
    query = get_latest_user_text(state["messages"])
    if not query:
        return {"context": []}

    dense_candidates = safe_dense_candidates(store, query, fetch_k=RETRIEVAL_FETCH_K)
    mmr_candidates = safe_mmr_candidates(
        store,
        query,
        top_k=max(RETRIEVAL_TOP_K * 2, RETRIEVAL_TOP_K),
        fetch_k=RETRIEVAL_MMR_FETCH_K,
    )
    documents = rerank_documents(query, dense_candidates, mmr_candidates)
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
    rag_references: list[str] = Field(default_factory=list)
    provider: str
    chat_id: str

app = FastAPI(title="LLM Semester Project")

@app.on_event("startup")
def startup():
    conn = get_db()
    conn.execute('CREATE TABLE IF NOT EXISTS chats (id TEXT PRIMARY KEY, title TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')
    conn.execute('CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id TEXT, role TEXT, content TEXT, context_count INTEGER, rag_references TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(messages)").fetchall()}
    if "rag_references" not in columns:
        conn.execute("ALTER TABLE messages ADD COLUMN rag_references TEXT")
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
    messages = conn.execute(
        "SELECT role, content, context_count, rag_references FROM messages WHERE chat_id = ? ORDER BY created_at ASC",
        (chat_id,),
    ).fetchall()
    conn.close()

    def parse_rag_references(raw: Any) -> list[str]:
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            return []
        return []

    return [
        {
            "role": m["role"],
            "content": m["content"],
            "context_count": m["context_count"],
            "rag_references": parse_rag_references(m["rag_references"]),
        }
        for m in messages
    ]

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
        provider=CHAT_PROVIDER,
        chat_id=chat_id
    )
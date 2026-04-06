from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path as FilePath
from typing import Any, Sequence, cast

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from settings import (
    CHAT_PROVIDER,
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIRECTORY,
    MAX_MESSAGE_TOKENS,
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    OLLAMA_FINETUNED_CHAT_MODEL,
    OLLAMA_EMBEDDING_MODEL,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    QA_DIRECTORY,
    RERANK_BM25_WEIGHT,
    RERANK_DENSE_WEIGHT,
    RERANK_LEXICAL_WEIGHT,
    RERANK_METADATA_WEIGHT,
    RERANK_MIN_SCORE,
    RERANK_RRF_SCALE,
    RERANK_RRF_WEIGHT,
    RETRIEVAL_BM25_FETCH_K,
    RETRIEVAL_FETCH_K,
    RETRIEVAL_LEXICAL_FETCH_K,
    RETRIEVAL_MMR_FETCH_K,
    RETRIEVAL_TOP_K,
    SUMMARY_RECENT_MESSAGE_COUNT,
    SUMMARY_TRIGGER_TOKENS,
    THREAD_ID,
)

ASSISTANT_IDENTITY = (
    "I am the NUST banking assistant. I answer questions using the NUST banking "
    "Q&A data available in this system."
)

SYSTEM_PROMPT_TEMPLATE = """You are the NUST banking assistant.
If the user asks who you are, what your name is, or what system they are using, identify yourself as the NUST banking assistant.
Do not describe yourself as Qwen, OpenAI, Alibaba Cloud, OpenRouter, or a generic large language model.
This assistant is strictly for NUST banking products and policies only.
Never recommend, compare, or suggest products/accounts from other banks.
If the user asks for alternatives, migration options, or comparisons, only suggest NUST account/product options found in retrieved context.
If no relevant NUST options are present in retrieved context, return the fallback sentence exactly.
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
Do not include markdown code fences unless the user explicitly asks for code.
Keep the answer concise, directly relevant, and pleasant to read.

Retrieved context:
{context}
"""

FALLBACK_ANSWER = "I do not have enough information in the provided banking data to answer that."

DOMAIN_GUARDRAIL_ANSWER = (
    "I can only assist with NUST banking products and data. "
    "Please ask about NUST accounts, services, or policies."
)

EXTERNAL_BANK_PATTERNS = (
    r"\bhbl\b",
    r"\bhabib bank\b",
    r"\bubl\b",
    r"\bunited bank\b",
    r"\bmeezan\b",
    r"\bbank islami\b",
    r"\bstandard chartered\b",
    r"\bmcb\b",
    r"\ballied bank\b",
    r"\baskari\b",
    r"\bbank alfalah\b",
    r"\bfaysal bank\b",
    r"\bnbp\b",
    r"\bnational bank of pakistan\b",
    r"\bnational savings\b",
    r"\bnibs\b",
)

PROVIDER_ALIASES: dict[str, str] = {
    "openrouter": "openrouter",
    "ollama": "ollama",
    "ollama-finetuned": "ollama-finetuned",
    "ollama_finetuned": "ollama-finetuned",
    "ollama-fine-tuned": "ollama-finetuned",
    "finetuned": "ollama-finetuned",
}


def normalize_provider(provider: str) -> str:
    normalized = (provider or "").strip().lower()
    return PROVIDER_ALIASES.get(normalized, normalized)


class State(MessagesState):
    context: list[str]
    trimmed_messages: list[BaseMessage]
    conversation_summary: str


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


def references_external_bank(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return any(re.search(pattern, normalized) for pattern in EXTERNAL_BANK_PATTERNS)


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
    provider = normalize_provider(provider)
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

    if provider in {"ollama", "ollama-finetuned"}:
        chat_ollama: Any = ChatOllama
        selected_model = OLLAMA_FINETUNED_CHAT_MODEL if provider == "ollama-finetuned" else OLLAMA_CHAT_MODEL
        return chat_ollama(
            model=selected_model,
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
    qa_directory = FilePath(QA_DIRECTORY)
    if not qa_directory.exists():
        return []

    documents: list[Document] = []
    for json_file in sorted(qa_directory.glob("qa_*.json")):
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
    summary_message: list[BaseMessage] = []
    summary = (state.get("conversation_summary") or "").strip()
    if summary:
        summary_message = [
            SystemMessage(
                content=(
                    "Conversation summary from earlier turns. "
                    "Use it for continuity, but prioritize the latest user request and retrieved banking context.\n\n"
                    f"{summary}"
                )
            )
        ]

    reserved_tokens = 0
    if summary_message:
        reserved_tokens = count_tokens_approximately(summary_message)

    available_tokens = max(300, MAX_MESSAGE_TOKENS - reserved_tokens)
    trimmed = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=available_tokens,
        start_on="human",
        end_on=("human", "tool"),
    )
    return {"trimmed_messages": [*summary_message, *list(trimmed)]}


def summarize_node(state: State, *, chat_model: Any = None) -> dict[str, str]:
    messages = list(state["messages"])
    if count_tokens_approximately(messages) <= SUMMARY_TRIGGER_TOKENS:
        return {"conversation_summary": state.get("conversation_summary", "")}

    if len(messages) <= SUMMARY_RECENT_MESSAGE_COUNT + 2:
        return {"conversation_summary": state.get("conversation_summary", "")}

    if not chat_model:
        raise ValueError("chat_model must be provided for summarization")

    existing_summary = (state.get("conversation_summary") or "").strip()
    messages_to_summarize = messages[:-SUMMARY_RECENT_MESSAGE_COUNT]

    transcript_lines: list[str] = []
    for message in messages_to_summarize:
        if isinstance(message, HumanMessage):
            role = "User"
        elif isinstance(message, AIMessage):
            role = "Assistant"
        elif isinstance(message, SystemMessage):
            role = "System"
        else:
            role = "Other"

        text = message_to_text(message)
        if text:
            transcript_lines.append(f"{role}: {text}")

    if not transcript_lines:
        return {"conversation_summary": existing_summary}

    summarization_prompt = (
        "Summarize the conversation history for short-term memory.\n"
        "Keep facts, user constraints, preferences, and unresolved asks.\n"
        "Drop chit-chat and repetition.\n"
        "Use concise bullet points.\n"
        "Preserve exact financial figures, percentages, limits, tenures, and product names when present."
    )

    transcript_block = "\n".join(transcript_lines)
    summary_request = (
        f"Existing summary (may be empty):\n{existing_summary or 'None'}\n\n"
        "New conversation chunk to fold in:\n"
        f"{transcript_block}\n\n"
        "Return an updated compact summary."
    )

    summary_response = chat_model.invoke(
        [
            SystemMessage(content=summarization_prompt),
            HumanMessage(content=summary_request),
        ]
    )

    return {"conversation_summary": message_to_text(summary_response)}


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

    context_block = "\n\n".join(context) if context else "No relevant context retrieved."
    response = chat_model.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT_TEMPLATE.format(context=context_block)),
            *trimmed_messages,
        ]
    )

    response_text = message_to_text(response)
    if references_external_bank(response_text):
        return {"messages": [AIMessage(content=DOMAIN_GUARDRAIL_ANSWER)]}

    return {"messages": [response]}


def build_graph(*, chat_model: Any = None, vector_store: Chroma | None = None):
    builder = StateGraph(State)

    builder.add_node(
        "retrieve",
        lambda state: retrieve_node(cast(State, state), vector_store=vector_store),
    )
    builder.add_node(
        "summarize",
        lambda state: summarize_node(cast(State, state), chat_model=chat_model),
    )
    builder.add_node("trim", trim_node)
    builder.add_node(
        "generate",
        lambda state: generate_node(cast(State, state), chat_model=chat_model),
    )
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "summarize")
    builder.add_edge("summarize", "trim")
    builder.add_edge("trim", "generate")
    builder.add_edge("generate", END)

    return builder.compile(checkpointer=MemorySaver())


@lru_cache(maxsize=2)
def get_graph(provider: str):
    return build_graph(chat_model=get_chat_model(provider))


def invoke_graph(message: str, thread_id: str = THREAD_ID, provider: str = CHAT_PROVIDER, graph: Any = None) -> dict[str, Any]:
    normalized_provider = normalize_provider(provider)
    compiled_graph = graph or get_graph(normalized_provider)
    return compiled_graph.invoke(
        cast(State, {"messages": [HumanMessage(content=message)]}),
        config={"configurable": {"thread_id": thread_id}},
    )
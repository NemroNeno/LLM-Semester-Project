# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NUST banking RAG chatbot. Users ask banking questions; the system retrieves relevant Q&A pairs from a local Chroma vector store and generates answers via an LLM. Domain-restricted to NUST banking products only.

## Commands

### Setup

```bash
uv sync
ollama pull nomic-embed-text
export OPENROUTER_API_KEY="your-key"   # required for default chat provider
```

### Ingest corpus into Chroma

```bash
uv run python ingest.py
```

Reads all `sheets_qa/qa_*.json`, resets the `banking_qa` Chroma collection, and re-embeds everything. Must be run before the app can answer questions.

### Run the server

```bash
uv run uvicorn app:app --reload
# or
uv run python main.py   # no reload by default; controlled by UVICORN_RELOAD env var
```

App available at `http://127.0.0.1:8000/`.

### Run Ragas evaluation

```bash
uv run python evaluate_rag_retrieval.py --limit 20 --output ragas_report.json
uv run python evaluate_rag_retrieval.py --metrics faithfulness,factual_correctness --provider ollama
```

Supported metrics: `context_precision`, `context_recall`, `faithfulness`, `factual_correctness`.

## Architecture

### Data flow

```
sheets_qa/qa_*.json → ingest.py → chroma_db/   (one-time setup)

User → POST /chat → LangGraph pipeline → SQLite (chat_history.db) → response
```

### LangGraph pipeline (`app.py`)

Four sequential nodes: `retrieve → summarize → trim → generate`

- **retrieve**: Hybrid retrieval combining dense (Chroma similarity), MMR, custom lexical overlap, and BM25. Results are merged and reranked with configurable weights (`RERANK_*` env vars) using Reciprocal Rank Fusion. Falls back to top-scored docs if the minimum score threshold removes everything.
- **summarize**: When conversation token count exceeds `SUMMARY_TRIGGER_TOKENS`, compresses older messages into a rolling bullet-point summary via the LLM. Recent `SUMMARY_RECENT_MESSAGE_COUNT` messages are kept verbatim.
- **trim**: Applies `trim_messages` to enforce `MAX_MESSAGE_TOKENS`, prepending the rolling summary as a `SystemMessage` if one exists.
- **generate**: Checks for identity queries (short-circuits with a fixed string), checks for empty context (returns fallback), then calls the LLM. Post-generation output is scanned for external bank mentions; if found, returns a domain guardrail response instead.

### Key modules

| File | Purpose |
|------|---------|
| `ingest.py` | One-shot corpus loader; resets then repopulates Chroma collection |
| `app.py` | FastAPI app + entire LangGraph pipeline + retrieval logic |
| `evaluate_rag_retrieval.py` | Ragas-based evaluation; imports `invoke_graph` from `app.py` |
| `main.py` | Thin `uvicorn.run` wrapper |
| `templates/index.html` | Jinja2 chat UI |
| `sheets_qa/qa_*.json` | Runtime corpus: `{question, answer, sheet}` records |
| `qa_extractor.py`, `qa_openai.py` | Legacy preprocessing utilities; do not modify |

### Persistence

- **Chroma** (`chroma_db/`): Vector embeddings via local Ollama `nomic-embed-text`
- **SQLite** (`chat_history.db`): `chats` and `messages` tables for multi-session history; created at startup
- **LangGraph MemorySaver**: In-process short-term memory per `thread_id` (aligned with `chat_id`)

### Chat providers

Switched via `CHAT_PROVIDER` env var:
- `openrouter` (default): `ChatOpenAI` pointed at `https://openrouter.ai/api/v1`, requires `OPENROUTER_API_KEY`
- `ollama`: `ChatOllama` using local model (default `qwen2.5:3b-instruct`)

Both providers share the same graph; `get_chat_model()` and `get_graph()` are `lru_cache`-memoized.

### Domain guardrails

Two layers in `app.py`:
1. **Identity interception** (`is_identity_query`): Returns a fixed assistant identity string without calling the LLM.
2. **Output filter** (`references_external_bank`): If the LLM response mentions a competitor bank, it is replaced with `DOMAIN_GUARDRAIL_ANSWER`.

## Key environment variables

| Variable | Default | Notes |
|----------|---------|-------|
| `CHAT_PROVIDER` | `openrouter` | `openrouter` or `ollama` |
| `OPENROUTER_API_KEY` | — | Required for OpenRouter |
| `CHAT_MODEL` | `qwen/qwen-2.5-7b-instruct` | OpenRouter model |
| `OLLAMA_CHAT_MODEL` | `qwen2.5:3b-instruct` | Ollama chat model |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Must be pulled via `ollama pull` |
| `CHROMA_PERSIST_DIRECTORY` | `chroma_db` | |
| `RETRIEVAL_TOP_K` | `4` | Final docs returned to LLM |
| `MAX_MESSAGE_TOKENS` | `1200` | Token budget for trimmed history |
| `QA_DIRECTORY` | `sheets_qa` | Source JSON corpus directory |

# LLM Semester Project

Local RAG chatbot for banking Q&A data using LangGraph, FastAPI, OpenRouter or Ollama for generation, and Chroma.

## Current Scope

- Runtime corpus: `sheets_qa/*.json`
- Legacy preprocessing scripts kept as-is: `qa_extractor.py`, `qa_openai.py`
- Environment management: `uv`
- Python version: 3.11+

## Stack

- FastAPI + Uvicorn for the web app and API
- Jinja2 for the minimal chat UI
- LangGraph for orchestration and short-term memory
- LangChain Core for messages and trimming utilities
- `langchain-openai` for `ChatOpenAI` against OpenRouter's OpenAI-compatible endpoint
- `langchain-ollama` for local `OllamaEmbeddings` and later local `ChatOllama` usage
- `langchain-chroma` + `chromadb` for persistent local vector storage

## Why These Packages

Current LangChain docs recommend standalone provider packages for integrations. That means this project uses `langchain-openai`, `langchain-ollama`, and `langchain-chroma` instead of the older `langchain-community` integration path.

## Local Prerequisites

1. Install Python 3.11 or newer.
2. Install `uv`.
3. Install Ollama and make sure the Ollama service is running locally.
4. Pull the local embedding model required by this project:

```powershell
ollama pull nomic-embed-text
```

5. For generation during development, set an OpenRouter API key:

```powershell
$env:OPENROUTER_API_KEY = "your-openrouter-key"
```

When your local chat model is ready, you can switch generation back to Ollama by setting `CHAT_PROVIDER=ollama` and ensuring the local chat model is available.

## Install Dependencies

This repository is already initialized as a `uv` project.

```powershell
uv sync
```

If you need to add or refresh packages later, `pyproject.toml` is the source of truth.

## Planned Runtime Configuration

The application code added in later tasks should keep configuration minimal and local-first. The intended defaults are:

- Chat provider default: `openrouter`
- OpenRouter chat model default: `qwen/qwen-2.5-7b-instruct`
- Ollama chat model default: `qwen2.5:3b`
- Embedding model: `nomic-embed-text`
- Chroma persist directory: `./chroma_db`
- Chroma collection name: `banking_qa`
- LangGraph thread ID: `global_session`

These values can stay as constants initially or be moved to environment variables if needed.

## Expected Workflow

The implementation tasks after bootstrap will add two main entry points:

1. Ingestion utility to embed the JSON corpus into Chroma.
2. FastAPI app to serve the chatbot and HTML interface.

Intended commands:

```powershell
uv run python ingest.py
uv run uvicorn app:app --reload
```

After starting Uvicorn, open `http://127.0.0.1:8000/` in the browser.

## Architecture Summary

- `ingest.py` loads `sheets_qa/*.json`, validates each Q&A record, and refreshes the local `banking_qa` Chroma collection.
- `app.py` builds a three-node LangGraph flow: `retrieve -> trim -> generate`.
- Retrieval stays local through Chroma plus Ollama embeddings.
- Generation currently defaults to OpenRouter through `ChatOpenAI` until the local chat model is ready.
- Conversation history is preserved in memory with a fixed thread ID through LangGraph's in-memory checkpointer.

## Basic Local Run

1. Install dependencies with `uv sync`.
2. Ensure Ollama is running and `nomic-embed-text` is available.
3. Set `OPENROUTER_API_KEY` for the current shell if using the default chat provider.
4. Run `uv run python ingest.py`.
5. Run `uv run uvicorn app:app --reload`.
6. Open `http://127.0.0.1:8000/` and send a banking question.

## Admin Document Ingestion

The main UI now includes a lightweight admin panel for dynamic ingestion.

- Upload supported files: `.txt`, `.pdf`, `.csv`, `.xlsx`, `.xls`, `.xlsm`
- Uploaded files are saved locally in a configurable folder and tracked in SQLite
- The app extracts text, generates Q/A pairs with OpenRouter Gemini 2.5 Flash, stores the generated pairs, and indexes them into Chroma
- Ingestion runs in the background and the UI polls progress until completion
- Deleting a document removes the source file, derived Q/A rows, and vector-store entries

Relevant environment variables:

- `DOCUMENT_UPLOAD_DIRECTORY` default: `uploaded_documents`
- `OPENROUTER_INGEST_MODEL` default: `google/gemini-2.5-flash`
- `INGESTION_DEFAULT_QA_COUNT` default: `5`
- `INGESTION_MAX_CHUNK_CHARS` default: `6000`
- `INGESTION_JOB_POLL_SECONDS` default: `1.0`

## Troubleshooting

- If `/chat` returns a `503`, set `OPENROUTER_API_KEY` or switch to `CHAT_PROVIDER=ollama` after the local chat model is available.
- If the app fails to retrieve context, rerun `uv run python ingest.py` and confirm `chroma_db` exists.
- If embeddings fail, verify that Ollama is running and `ollama list` shows `nomic-embed-text`.
- If you want to move back to local generation later, set `CHAT_PROVIDER=ollama` and ensure `OLLAMA_CHAT_MODEL` points to an installed local model.
- If retrieval finds no useful documents, the app returns a safe fallback instead of fabricating an answer.

## Data Source

The chatbot should use the JSON files under `sheets_qa/` as the retrieval corpus. Each record already contains:

- `question`
- `answer`
- `sheet`

The ingestion step should combine question and answer into embeddable text and preserve `sheet` as metadata.

## Ingestion Behavior

- `ingest.py` reads all `qa_*.json` files under `sheets_qa/`
- each record becomes one embedded document
- the script preserves `sheet`, `source_file`, `record_index`, `question`, and `has_answer` as metadata
- rerunning the script refreshes the target Chroma collection before adding documents, so it does not silently duplicate vectors

Optional environment variables for ingestion and later runtime code:

- `CHAT_PROVIDER` default: `openrouter`
- `CHAT_MODEL` default: `qwen/qwen-2.5-7b-instruct`
- `OPENROUTER_API_KEY` required for OpenRouter generation
- `OPENROUTER_BASE_URL` default: `https://openrouter.ai/api/v1`
- `OLLAMA_CHAT_MODEL` default: `qwen2.5:3b`
- `OLLAMA_EMBEDDING_MODEL` default: `nomic-embed-text`
- `OLLAMA_BASE_URL` default: Ollama local default
- `CHROMA_PERSIST_DIRECTORY` default: `chroma_db`
- `CHROMA_COLLECTION_NAME` default: `banking_qa`
- `QA_DIRECTORY` default: `sheets_qa`
- `INGEST_BATCH_SIZE` default: `64`
- `RETRIEVAL_TOP_K` default: `4`
- `MAX_MESSAGE_TOKENS` default: `1200`
- `LANGGRAPH_THREAD_ID` default: `global_session`

## Notes

- Keep the project minimal and easy to trace.
- Preserve `qa_extractor.py` and `qa_openai.py` as legacy preprocessing utilities.
- Use official LangChain and LangGraph docs before implementation changes involving LangGraph state, memory, Ollama integrations, or vector store behavior.

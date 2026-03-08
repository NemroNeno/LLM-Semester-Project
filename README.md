# LLM Semester Project

Local RAG chatbot for banking Q&A data using LangGraph, FastAPI, Ollama, and Chroma.

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
- `langchain-ollama` for `ChatOllama` and `OllamaEmbeddings`
- `langchain-chroma` + `chromadb` for persistent local vector storage

## Why These Packages

Current LangChain docs recommend standalone provider packages for integrations. That means this project uses `langchain-ollama` and `langchain-chroma` instead of the older `langchain-community` integration path for Ollama and Chroma.

## Local Prerequisites

1. Install Python 3.11 or newer.
2. Install `uv`.
3. Install Ollama and make sure the Ollama service is running locally.
4. Pull the models required by this project:

```powershell
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

## Install Dependencies

This repository is already initialized as a `uv` project.

```powershell
uv sync
```

If you need to add or refresh packages later, `pyproject.toml` is the source of truth.

## Planned Runtime Configuration

The application code added in later tasks should keep configuration minimal and local-first. The intended defaults are:

- Chat model: `qwen2.5:3b`
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

- `OLLAMA_EMBEDDING_MODEL` default: `nomic-embed-text`
- `OLLAMA_BASE_URL` default: Ollama local default
- `CHROMA_PERSIST_DIRECTORY` default: `chroma_db`
- `CHROMA_COLLECTION_NAME` default: `banking_qa`
- `QA_DIRECTORY` default: `sheets_qa`
- `INGEST_BATCH_SIZE` default: `64`

## Notes

- Keep the project minimal and easy to trace.
- Preserve `qa_extractor.py` and `qa_openai.py` as legacy preprocessing utilities.
- Use official LangChain and LangGraph docs before implementation changes involving LangGraph state, memory, Ollama integrations, or vector store behavior.

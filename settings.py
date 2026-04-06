from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "banking_qa")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
QA_DIRECTORY = os.getenv("QA_DIRECTORY", "sheets_qa")
CHAT_PROVIDER = os.getenv("CHAT_PROVIDER", "openrouter").strip().lower()
OPENROUTER_MODEL = os.getenv("CHAT_MODEL", "qwen/qwen-2.5-7b-instruct")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:3b-instruct")
OLLAMA_FINETUNED_CHAT_MODEL = os.getenv(
	"OLLAMA_FINETUNED_CHAT_MODEL",
	"qwen2.5:3b-instruct-bitext-cs-dataet-tuned",
)
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))
RETRIEVAL_FETCH_K = int(os.getenv("RETRIEVAL_FETCH_K", "24"))
RETRIEVAL_MMR_FETCH_K = int(os.getenv("RETRIEVAL_MMR_FETCH_K", "40"))
RETRIEVAL_LEXICAL_FETCH_K = int(os.getenv("RETRIEVAL_LEXICAL_FETCH_K", "24"))
RETRIEVAL_BM25_FETCH_K = int(os.getenv("RETRIEVAL_BM25_FETCH_K", "24"))
MAX_MESSAGE_TOKENS = int(os.getenv("MAX_MESSAGE_TOKENS", "1200"))
SUMMARY_TRIGGER_TOKENS = int(os.getenv("SUMMARY_TRIGGER_TOKENS", str(MAX_MESSAGE_TOKENS)))
SUMMARY_RECENT_MESSAGE_COUNT = int(os.getenv("SUMMARY_RECENT_MESSAGE_COUNT", "8"))
THREAD_ID = os.getenv("LANGGRAPH_THREAD_ID", "global_session")

RERANK_DENSE_WEIGHT = float(os.getenv("RERANK_DENSE_WEIGHT", "0.45"))
RERANK_LEXICAL_WEIGHT = float(os.getenv("RERANK_LEXICAL_WEIGHT", "0.30"))
RERANK_RRF_WEIGHT = float(os.getenv("RERANK_RRF_WEIGHT", "0.15"))
RERANK_METADATA_WEIGHT = float(os.getenv("RERANK_METADATA_WEIGHT", "0.10"))
RERANK_BM25_WEIGHT = float(os.getenv("RERANK_BM25_WEIGHT", "0.10"))
RERANK_RRF_SCALE = float(os.getenv("RERANK_RRF_SCALE", "30.0"))
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "0.22"))

DB_PATH = os.getenv("CHAT_HISTORY_DB", "chat_history.db")
TEMPLATES_DIRECTORY = "templates"
UPLOAD_DIRECTORY = os.getenv("DOCUMENT_UPLOAD_DIRECTORY", "uploaded_documents")
OPENROUTER_INGEST_MODEL = os.getenv("OPENROUTER_INGEST_MODEL", "google/gemini-2.5-flash")
INGESTION_DEFAULT_QA_COUNT = int(os.getenv("INGESTION_DEFAULT_QA_COUNT", "5"))
INGESTION_MAX_CHUNK_CHARS = int(os.getenv("INGESTION_MAX_CHUNK_CHARS", "6000"))
INGESTION_JOB_POLL_SECONDS = float(os.getenv("INGESTION_JOB_POLL_SECONDS", "1.0"))
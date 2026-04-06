# Software Requirements Specification (SRS)

## 1. Introduction

### 1.1 Purpose

This document specifies the functional and non-functional requirements of the LLM Semester Project: a domain-restricted NUST banking assistant that uses Retrieval-Augmented Generation (RAG) with local vector search, conversational memory, and an admin ingestion workflow.

### 1.2 Product Scope

The system provides:

- A web chat interface for end users to ask NUST banking questions
- Retrieval from local Chroma vector store backed by banking Q&A corpus
- Response generation through selectable providers (OpenRouter, local Ollama base model, local Ollama fine-tuned model)
- Conversation history persistence and memory-aware summarization
- An admin panel for uploading new documents, auto-generating Q&A pairs, indexing vectors, monitoring progress, and deleting ingested content
- A RAG evaluation script using Ragas metrics

### 1.3 Definitions and Abbreviations

- RAG: Retrieval-Augmented Generation
- LLM: Large Language Model
- QA: Question/Answer pair
- MMR: Max Marginal Relevance
- RRF: Reciprocal Rank Fusion
- SRS: Software Requirements Specification

### 1.4 References

- LangGraph persistence and checkpointer usage: https://docs.langchain.com/oss/python/langgraph/persistence
- LangGraph memory and trimming guidance: https://docs.langchain.com/oss/python/langgraph/add-memory
- LangChain vector store interface: https://docs.langchain.com/oss/python/integrations/vectorstores/index

## 2. Overall Description

### 2.1 User Classes

- End User
Uses chat UI, selects provider, starts chats, and asks banking questions.

- Admin User
Uploads files for ingestion, views ingestion progress/status, previews extracted text, and deletes document-derived data.

### 2.2 Operating Environment

- OS: Windows (validated), Python-compatible cross-platform execution
- Runtime: Python 3.11+
- App server: FastAPI + Uvicorn
- Local services: Ollama for embeddings and optional generation
- Optional remote service: OpenRouter API for generation and ingestion QA synthesis
- Storage: SQLite (chat and ingestion metadata), Chroma persistent store, local filesystem uploads

### 2.3 Constraints

- Domain restriction to NUST banking context
- Retrieval quality depends on available indexed data
- OpenRouter-dependent features require API key and network access
- Local model inference requires installed Ollama models and sufficient host resources

### 2.4 Assumptions and Dependencies

- Required packages are installed via pyproject/uv
- Ollama daemon is running when using local embeddings/chat models
- Ingestion source files are within supported types
- Admin ingestion currently runs as in-process background thread jobs

## 3. Functional Requirements

### 3.1 Chat and Session Management

FR-1 The system shall create a new chat session when no chat_id is provided.

FR-2 The system shall persist user and assistant messages in SQLite.

FR-3 The system shall list existing chats sorted by recency.

FR-4 The system shall retrieve complete ordered message history for a selected chat.

FR-5 The system shall provide a pre-check endpoint indicating whether the next user turn triggers summarization.

### 3.2 Provider and Model Selection

FR-6 The system shall support provider selection per request via UI/API.

FR-7 Supported providers shall include:
- openrouter
- ollama
- ollama-finetuned

FR-8 The system shall normalize provider aliases and whitespace variations before invocation.

FR-9 If provider credentials are missing or unavailable, the system shall return actionable HTTP errors.

### 3.3 Retrieval and Ranking

FR-10 The system shall retrieve candidate documents from Chroma dense search.

FR-11 The system shall retrieve diversity candidates using MMR when available.

FR-12 The system shall compute lexical and metadata-aware scores for re-ranking.

FR-13 The system shall optionally include BM25 candidates when available.

FR-14 The system shall fuse evidence using weighted scoring and RRF-derived rank signals.

FR-15 The system shall return top_k ranked contexts to the generation node.

FR-16 If strict minimum score filtering removes all candidates, the system shall fall back to highest scored candidates.

### 3.4 Conversational Memory and Trimming

FR-17 The system shall maintain short-term conversation state per thread_id using LangGraph checkpointer.

FR-18 When token usage exceeds configured threshold, the system shall summarize older turns.

FR-19 The system shall preserve a configurable number of recent messages verbatim while summarizing older context.

FR-20 The system shall trim messages to configured token budget before final generation.

### 3.5 Guardrails and Response Behavior

FR-21 The system shall detect identity queries and return fixed assistant identity text without model generation.

FR-22 If no retrieval context exists, the system shall return a fixed fallback response.

FR-23 The system shall block final outputs that reference external bank entities by replacing with domain guardrail answer.

FR-24 The system shall format generated responses in markdown per system prompt style rules.

### 3.6 Admin Document Ingestion

FR-25 The system shall allow file upload from admin panel and admin API.

FR-26 Supported upload extensions shall include:
- .txt
- .pdf
- .csv
- .xlsx
- .xls
- .xlsm

FR-27 The system shall store uploaded files under configured upload directory with generated internal IDs.

FR-28 The system shall create a document record and ingestion job record for each upload.

FR-29 Ingestion shall run asynchronously via background worker thread.

FR-30 The system shall extract text based on file type parser.

FR-31 The system shall split extracted text into chunks and generate QA pairs via OpenRouter ingest model.

FR-32 The system shall parse and sanitize LLM JSON output into QA pair records.

FR-33 If model output is invalid, the system shall apply deterministic fallback QA generation at chunk level.

FR-34 The system shall embed and store generated QA documents in Chroma with deterministic vector IDs.

FR-35 The system shall persist generated QA pairs and vector IDs in SQLite.

FR-36 The system shall expose ingestion status (stage, progress, errors) through admin API.

FR-37 The UI shall poll ingestion status and render progress percentage bars.

FR-38 The system shall support document deletion that removes vectors, QA rows, source file, and soft-deletes document record.

### 3.7 UI Requirements

FR-39 The main UI shall provide a user/admin panel toggle button fixed on top-right viewport.

FR-40 User panel shall include:
- Chat history list
- New chat button
- Provider/model select
- Streaming-like typing indicator
- Message composer
- RAG source dropdown for assistant answers

FR-41 Admin panel shall include:
- File selection and upload controls
- Ingestion status text
- Upload progress bar
- List of ingested documents with status/qa_count/progress
- Document preview section showing metadata and extracted text
- Delete action per document

### 3.8 Data Ingestion from Existing QA Corpus

FR-42 The standalone ingest script shall load all qa_*.json files from source directory.

FR-43 Each corpus row shall validate required fields (question, answer, sheet).

FR-44 The script shall reset and repopulate target Chroma collection.

### 3.9 Evaluation

FR-45 The evaluation script shall run chatbot generation over sampled QA examples.

FR-46 The evaluation script shall support provider selection for evaluation LLM.

FR-47 Supported metric set shall include:
- context_precision
- context_recall
- faithfulness
- factual_correctness

FR-48 The evaluation script shall output summary metrics and optional JSON report.

## 4. External Interface Requirements

### 4.1 HTTP API

- GET /
Returns chat/admin web UI.

- GET /chats
Returns chat list.

- GET /chats/{chat_id}/messages
Returns ordered messages for chat.

- POST /chat/will-summarize
Input: message, optional chat_id. Output: will_summarize bool.

- POST /chat
Input: message, optional chat_id, provider. Output: reply, context_count, rag_references, provider, chat_id.

- GET /admin/documents
Returns active (non-deleted) documents.

- GET /admin/documents/{document_id}
Returns document + latest ingestion job status.

- POST /admin/documents
Multipart upload, returns document_id and job_id.

- DELETE /admin/documents/{document_id}
Deletes vectors + derived artifacts and marks document deleted.

### 4.2 Storage Interfaces

- SQLite tables: chats, messages, documents, ingestion_jobs, document_qa_pairs
- Chroma collection: banking_qa (configurable)
- Local files: sheets_qa source corpus, uploaded_documents runtime uploads

## 5. Non-Functional Requirements

### 5.1 Reliability

NFR-1 API shall return deterministic error responses on provider/config failures.

NFR-2 Ingestion failures shall be captured in document/job error fields.

NFR-3 DB initialization shall include migration-safe column additions for legacy schemas.

### 5.2 Performance

NFR-4 Chat response latency should remain acceptable for interactive usage under typical local workloads.

NFR-5 Ingestion job progress should update frequently enough for user feedback (poll interval configurable).

NFR-6 Retrieval shall bound candidate sizes through configurable fetch/top_k values.

### 5.3 Security and Privacy

NFR-7 API keys must be sourced from environment variables, never hard-coded.

NFR-8 Uploaded file handling should sanitize filenames and constrain supported types.

NFR-9 System currently has no auth layer; deployment in shared environments requires additional access control.

### 5.4 Maintainability

NFR-10 Codebase shall preserve modular separation across API, RAG, DB, ingestion, and settings modules.

NFR-11 Config values shall be centralized and environment-driven.

NFR-12 Documentation shall track both inference path and finetuning artifacts.

## 6. Configuration Requirements

Core runtime environment variables include:

- CHAT_PROVIDER
- CHAT_MODEL
- OLLAMA_CHAT_MODEL
- OLLAMA_FINETUNED_CHAT_MODEL
- OLLAMA_EMBEDDING_MODEL
- OPENROUTER_API_KEY
- OPENROUTER_BASE_URL
- OPENROUTER_INGEST_MODEL
- CHROMA_PERSIST_DIRECTORY
- CHROMA_COLLECTION_NAME
- QA_DIRECTORY
- CHAT_HISTORY_DB
- DOCUMENT_UPLOAD_DIRECTORY
- RETRIEVAL_TOP_K and related retrieval/rerank knobs
- SUMMARY_TRIGGER_TOKENS and MAX_MESSAGE_TOKENS

## 7. Fine-Tuning Requirements and Outputs

### 7.1 Fine-Tuning Inputs

- Base model: Qwen/Qwen2.5-3B-Instruct
- Dataset: Bitext customer support dataset
- Config files:
- finetuning/finetune_config.yaml (light adaptation profile)
- finetuning/qlora_config.yaml (expanded QLoRA profile)

### 7.2 Fine-Tuning Output Artifacts

Artifacts available in finetuning/finetuned_model include:

- adapter_model.safetensors
- adapter.gguf
- adapter_config.json
- tokenizer files and chat template
- Modelfile for Ollama model creation

### 7.3 Runtime Integration

The runtime shall support selecting Ollama fine-tuned model using provider value ollama-finetuned mapped to configured OLLAMA_FINETUNED_CHAT_MODEL.

### 7.4 Observed Fine-Tuning Metrics (Artifact Snapshot)

The generation comparison artifact reports:

- pre_perplexity: 24.675692425483533
- post_perplexity: 2.407075336683488

These values indicate substantial fit to training-style distribution but do not by themselves guarantee better factuality for NUST banking RAG outputs.

## 8. Risks and Limitations

- Fine-tuning dataset domain differs from NUST banking corpus; style gains may not equal domain factual gains.
- Admin ingestion currently trusts generated QA without human review workflow.
- In-process threading is suitable for lightweight workloads, but not durable queue semantics.
- No authentication/authorization on admin routes in current implementation.

## 9. Future Enhancements

- Add role-based auth for admin endpoints
- Add QA pair review/approval before vector indexing
- Add queue-backed workers and retry policies
- Add structured observability (metrics/traces)
- Add regression tests for retrieval/reranking and guardrails

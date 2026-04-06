# Architecture and Tech Stack

## 1. System Overview

The project is a modular FastAPI application implementing a hybrid RAG + knowledge-graph chatbot for NUST banking data, with optional remote generation and admin-driven dynamic ingestion.

Primary flow:

1. User submits message from web UI.
2. API runs Guardrails input validation; blocked input returns a safe refusal.
3. API stores accepted user message in SQLite.
4. LangGraph pipeline executes retrieve -> summarize -> trim -> generate.
5. Generate node runs Guardrails output validation before final persistence.
6. Assistant response and RAG references are persisted.
7. Assistant response, RAG references, and KG references are persisted.
8. UI renders markdown answer with both RAG and KG source snippets.

Admin ingestion flow:

1. Admin uploads document in UI/API.
2. Backend stores file and creates document/job records.
3. Background thread extracts text and generates QA pairs.
4. QA pairs are embedded and indexed in Chroma.
5. Document text chunks are also inserted into KG as episodic entries.
6. Job/document progress is polled from UI until completion.

## 2. Module Architecture

### 2.1 settings.py

Responsibility:

- Centralized environment-driven runtime configuration
- Defaults for providers, models, retrieval, summarization, ingestion behavior, and KG integration

Key design:

- Single source of truth for constants consumed by all modules

### 2.2 db.py

Responsibility:

- SQLite connection and schema initialization
- Chat persistence and retrieval
- Summarization pre-check token accounting
- Document/job/QA persistence for ingestion subsystem
- Persistence of KG references per chat message and KG episode IDs per document

Schema components:

- chats
- messages
- documents
- ingestion_jobs
- document_qa_pairs

Notable KG-related fields:

- messages.kg_references
- documents.kg_episode_ids

Migration behavior:

- Startup init includes table creation and additive column migration checks for backward compatibility.

### 2.3 rag.py

Responsibility:

- Provider/model selection and normalization
- Embedding and vector store access
- Knowledge graph fact retrieval integration
- Hybrid candidate retrieval and weighted reranking
- LangGraph node definitions and graph compilation
- Graph invocation entrypoint

Node chain:

- retrieve: gathers and reranks vector context and pulls KG facts
- summarize: rolls older history into compact memory when threshold exceeded
- trim: token-budget trims retained messages
- generate: answers with system prompt and guardrails

Key behaviors:

- Identity query short-circuit
- Empty-context fallback response
- Guardrails output validation for toxicity, private data, and grounding checks
- External bank mention output guardrail as fallback layer

### 2.4 document_ingestion.py

Responsibility:

- Upload persistence
- Text extraction by file type
- Chunking and OpenRouter QA synthesis
- QA/vector document creation
- Chroma indexing and SQLite metadata updates
- KG episode ingestion for uploaded document chunks
- Document deletion lifecycle

Supported extractors:

- TXT parser
- PDF parser (pypdf)
- CSV parser (pandas)
- Excel parser for xlsx/xls/xlsm (pandas + openpyxl/xlrd)

Execution model:

- Per-upload daemon thread for ingestion job processing

### 2.5 api.py

Responsibility:

- FastAPI routing and request/response models
- Chat lifecycle endpoints
- Admin ingestion endpoints
- Startup DB initialization hook
- Chat payload support for both rag_references and kg_references
- Guardrails input validation before graph invocation

### 2.13 guardrails_service.py

Responsibility:

- Centralized Guardrails AI validation for user input and model output
- ToxicLanguage, PrivateData, and AntiHallucination validator flow

Key design:

- Input validation blocks unsafe content before graph execution
- Output validation blocks unsafe or weakly grounded responses after generation
- Guardrails runtime errors degrade gracefully to preserve chat availability

### 2.6 knowledge_graph.py

Responsibility:

- Graphiti client construction and lifecycle
- Neo4j-backed KG ingestion and search
- Sync-safe async execution bridge for FastAPI request path
- Fallback keyword search against Neo4j when Graphiti semantic search fails

Key design:

- Dedicated background event loop thread for async Graphiti calls from sync code paths
- Timeout-bounded calls for search/ingestion/deletion
- Graceful degradation to fallback retrieval instead of failing chat

### 2.7 ingest_markdown_kg.py

Responsibility:

- One-time static KG bootstrap from `sheets_markdown/*.md`
- Chunking of markdown files before KG insertion
- Progress tracking using `tqdm`

### 2.8 clear_neo4j.py

Responsibility:

- Guarded local utility to wipe all Neo4j graph data for reset/re-ingestion workflows

### 2.9 app.py

Responsibility:

- Compatibility facade and export surface to keep entrypoints/imports stable

### 2.10 ingest.py

Responsibility:

- One-shot ingestion of canonical sheets_qa corpus
- Collection reset and repopulation workflow for Chroma

### 2.11 evaluate_rag_retrieval.py

Responsibility:

- Batch evaluation harness for chatbot with Ragas metrics
- Provider-aware evaluator LLM setup
- JSON report generation

### 2.12 templates/index.html

Responsibility:

- Unified user + admin interface
- Chat rendering, provider selection, history controls
- Admin upload/list/poll/delete/preview controls

## 3. Runtime Component Diagram (Logical)

- Browser UI
- FastAPI API layer
- LangGraph execution engine
- Graphiti knowledge graph service layer
- LLM providers:
- OpenRouter via ChatOpenAI
- Ollama via ChatOllama
- Embedding provider:
- OllamaEmbeddings
- Chroma persistent vector store
- Neo4j graph database (knowledge graph persistence)
- SQLite persistence layer
- Local file storage for uploads and source corpus

## 4. Data Architecture

### 4.1 Retrieval Corpus Data

Source corpus:

- sheets_qa/qa_*.json

Embedded format:

- page_content: Question + Answer text
- metadata: sheet, source_file, record_index, question, has_answer

### 4.2 Chat Data

Stored in SQLite:

- chat headers (id/title/timestamp)
- messages (role/content/context_count/rag_references/timestamp)

### 4.3 Ingestion Data

Document record tracks:

- file identity and location
- status and progress
- extracted text and length
- QA count and vector IDs
- latest job and errors
- soft delete timestamp

Job record tracks:

- lifecycle stage
- percentage progress
- status/message/error timestamps

QA pair rows track:

- question/answer payload
- vector ID linkage

### 4.4 Knowledge Graph Data

Graph persistence (Neo4j):

- Episodic nodes from markdown and uploaded-document ingestion
- RELATES_TO fact edges used for semantic/fallback retrieval

Runtime metadata persistence (SQLite):

- `messages.kg_references` stores per-turn KG sources shown in UI
- `documents.kg_episode_ids` stores KG episode IDs for cleanup on delete

## 5. Retrieval and Reranking Strategy

Candidate sources:

- Dense similarity from Chroma
- MMR from Chroma
- Lexical overlap against corpus
- Optional BM25 retriever
- Neo4j/Graphiti KG fact search (top-K)

Fusion approach:

- Weighted score uses dense + lexical + metadata + BM25 + RRF features
- Configurable thresholds and weights via environment variables
- Fallback to top scores when threshold filtering returns empty

Rationale:

- Dense retrieval captures semantic similarity
- Lexical/BM25 recover term-specific and numeric queries
- RRF reduces dependence on a single retriever signal
- KG facts provide relation-oriented evidence complementary to vector chunks

Failure handling:

- If Graphiti semantic search fails (for example provider quota limit), system falls back to Neo4j keyword fact matching.

## 6. Memory and Context Management

Short-term memory:

- LangGraph MemorySaver checkpointer keyed by thread_id

Summarization trigger:

- Triggered once token count crosses SUMMARY_TRIGGER_TOKENS
- Preserves last N recent messages verbatim

Context window protection:

- trim_messages enforces MAX_MESSAGE_TOKENS budget
- Summary inserted as system guidance before recent turns

## 7. Guardrail Architecture

Input-side guardrail:

- Identity intent detection and deterministic identity answer

Output-side guardrail:

- External-bank regex checks replace response with domain constraint message

Fallback behavior:

- Missing retrieval context yields deterministic fallback answer

## 8. Ingestion Pipeline Internals

Stages (high-level):

1. queued/saving
2. extracting
3. generating
4. saving_qa
5. completed or failed

QA generation protocol:

- Prompt asks for strict JSON schema with pairs array
- Parser attempts direct JSON parse, then object extraction fallback
- If parsing fails, creates deterministic fallback QA pair from chunk text

Vector write path:

- Builds synthetic QA documents with source metadata
- Uses deterministic IDs: doc:{document_id}:qa:{index}
- Writes to Chroma then persists same IDs in SQLite

Deletion path:

- Reads vector IDs from document record or QA table
- Deletes vectors best-effort
- Deletes QA rows and source file
- Marks document soft-deleted

## 9. API Contract Summary

Chat APIs:

- GET /chats
- GET /chats/{chat_id}/messages
- POST /chat/will-summarize
- POST /chat
Includes both `rag_references` and `kg_references` in response payload.

Admin APIs:

- GET /admin/documents
- GET /admin/documents/{document_id}
- POST /admin/documents
- DELETE /admin/documents/{document_id}

## 10. Frontend Architecture

Client behaviors:

- Maintains active panel mode (user/admin)
- Maintains active chat_id and selected document
- Fetch-based interaction with API
- Poll loop for ingestion status updates
- Markdown rendering for assistant messages via marked

Primary UX features:

- Top-right panel switch
- Sidebar chat history and provider selector
- RAG sources dropdown per assistant turn
- KG sources dropdown per assistant turn
- Admin progress bars and extracted-text preview

## 11. Technology Stack

Application and API:

- FastAPI
- Uvicorn
- Pydantic
- Jinja2

RAG and orchestration:

- LangGraph
- LangChain Core
- langchain-openai
- langchain-ollama
- langchain-chroma
- chromadb
- graphiti-core
- rank-bm25

Data and document handling:

- sqlite3
- pandas
- pypdf
- openpyxl
- xlrd
- python-multipart
- neo4j (driver used by utility/fallback path)

Evaluation:

- ragas
- openai SDK
- tqdm

Environment and packaging:

- uv
- python-dotenv

## 12. Fine-Tuning Subsystem (Inspected Artifacts)

### 12.1 Training Profile Summary

Observed configuration patterns across finetuning config files:

- Base model: Qwen/Qwen2.5-3B-Instruct
- QLoRA-style low-rank adaptation
- 4-bit NF4 quantization with bfloat16 compute
- ChatML-style prompt template with im_start/im_end tokens

Two config intents exist:

- A light adaptation profile (rank 8, targeted modules q_proj/v_proj, 6k sample subset, 1 epoch)
- An expanded profile (rank 64, broader module coverage, 3 epochs)

### 12.2 Produced Model Artifacts

Under finetuning/finetuned_model:

- Adapter weights (safetensors and gguf)
- PEFT adapter config (r=8, alpha=16, dropout=0.05, q_proj/v_proj)
- Tokenizer and chat template files
- Ollama Modelfile pointing FROM qwen2.5:3b-instruct and ADAPTER ./adapter.gguf

### 12.3 Deployment Integration

Current app integration path:

- Ollama model was built with custom tag (default configured as qwen2.5:3b-instruct-bitext-cs-dataet-tuned)
- Provider alias normalization maps finetuned variants to ollama-finetuned
- UI model picker exposes the fine-tuned option

### 12.4 Fine-Tuning Evaluation Artifact Snapshot

From finetuning_results/generation_comparison.json:

- pre_perplexity: 24.6757
- post_perplexity: 2.4071

Interpretation:

- The model appears more adapted to training response distribution/style
- RAG grounding quality for NUST banking still depends primarily on retrieval context and guardrails

## 13. Operational Considerations

### 13.1 Startup Sequence

1. Ensure dependencies installed with uv sync.
2. Ensure Ollama service and embedding model availability.
3. Run ingestion for baseline corpus (ingest.py) if needed.
4. Optionally bootstrap static KG with ingest_markdown_kg.py.
5. Start API server via uvicorn or main.py.

### 13.2 Failure Modes

- Missing OpenRouter key for openrouter provider
- Missing local Ollama model for selected provider
- Empty or unparsable uploaded files
- LLM QA generation returning invalid JSON
- Chroma deletion/indexing transient errors
- Graphiti semantic search failures due to provider quota or external API limits
- Neo4j connectivity/auth failures (KG falls back or returns empty KG context)

### 13.3 Scalability Constraints

- In-process thread jobs are simple but not horizontally durable
- SQLite and local filesystem are single-host oriented
- No distributed queue or object storage in current design

## 14. Security and Compliance Notes

- Admin endpoints currently have no authentication/authorization
- Uploaded files are locally stored and should be protected by host-level access controls
- API key material is env-based; avoid committing secrets in repository files

## 15. Recommended Next Architectural Steps

1. Add authn/authz for admin ingestion operations.
2. Move ingestion workers to durable queue (RQ/Celery/Arq or equivalent).
3. Add structured telemetry for retrieval scores, stages, and failure analytics.
4. Add integration tests for provider selection, ingestion lifecycle, and guardrails.
5. Optionally add human QA review gate before vector indexing.

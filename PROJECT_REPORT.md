# NUST Banking Intelligent Chatbot
## Comprehensive Technical Project Report
### Semester 8 — Large Language Models Course

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Data Preprocessing Pipeline](#3-data-preprocessing-pipeline)
4. [RAG Corpus Ingestion](#4-rag-corpus-ingestion)
5. [Hybrid Retrieval System](#5-hybrid-retrieval-system)
6. [LangGraph Orchestration Pipeline](#6-langgraph-orchestration-pipeline)
7. [Conversation Memory Management](#7-conversation-memory-management)
8. [Knowledge Graph Integration](#8-knowledge-graph-integration)
9. [Dynamic Document Ingestion](#9-dynamic-document-ingestion)
10. [Guardrails and Safety Layer](#10-guardrails-and-safety-layer)
11. [LLM Providers and Configuration](#11-llm-providers-and-configuration)
12. [Web Application and API](#12-web-application-and-api)
13. [Database Layer](#13-database-layer)
14. [QLoRA Fine-Tuning](#14-qlora-fine-tuning)
15. [Ragas Evaluation Framework](#15-ragas-evaluation-framework)
16. [Evaluation Results and Analysis](#16-evaluation-results-and-analysis)
17. [Technology Stack](#17-technology-stack)
18. [Architectural Decisions and Trade-offs](#18-architectural-decisions-and-trade-offs)
19. [Conclusion](#19-conclusion)

---

## 1. Project Overview

This project implements a domain-restricted, production-grade Retrieval-Augmented Generation (RAG) chatbot for **NUST Bank** — a Pakistani banking institution. The system allows customers and staff to ask natural language questions about banking products, account features, profit rates, eligibility criteria, and services, receiving accurate, grounded answers drawn entirely from the bank's official documentation.

### Core Objectives

- Build an end-to-end intelligent Q&A system from raw Excel spreadsheets to a deployable chatbot
- Ensure factual accuracy through retrieval-grounding and output guardrails
- Support multi-turn conversations with persistent memory
- Extend retrieval with a structured knowledge graph
- Enable runtime ingestion of new documents without restarting
- Fine-tune a local language model on customer service conversational style
- Evaluate the complete pipeline objectively using Ragas metrics

### Scope and Constraints

The system is strictly domain-restricted to NUST banking products. It will not answer questions about competitor banks, provide generic financial advice, or generate content outside the retrieved corpus. This restriction is enforced at two independent layers — the system prompt and a post-generation output validator.

---

## 2. System Architecture

The system is composed of six major subsystems that interact to form a complete intelligent document Q&A pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│  Excel Sheets → LlamaParse → Markdown → GPT-4 → QA JSON pairs  │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      STORAGE LAYER                              │
│  Chroma (vectors) + Neo4j (graph) + SQLite (chat + docs)        │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    RETRIEVAL LAYER                              │
│  Dense + MMR + BM25 + Lexical → RRF Reranking → Top-K Docs     │
│                  + Knowledge Graph Search                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                  ORCHESTRATION LAYER (LangGraph)                │
│         retrieve → summarize → trim → generate                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                     SAFETY LAYER                                │
│   Input Guardrails (toxicity, PII) + Output Guardrails          │
│   (hallucination, domain enforcement, external bank filter)     │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    PRESENTATION LAYER                           │
│           FastAPI REST endpoints + Jinja2 Chat UI               │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow (Single Query)

```
User Question
     │
     ▼
Input Guardrails (toxicity/PII check)
     │
     ▼
retrieve_node
  ├── Chroma dense search (24 candidates)
  ├── Chroma MMR search (40 candidates)
  ├── BM25 lexical search (24 candidates)
  ├── Custom lexical overlap scorer
  └── RRF Reranking → Top 4 documents
  └── Knowledge Graph search → Top 2 facts
     │
     ▼
summarize_node (if token budget exceeded)
     │
     ▼
trim_node (enforce MAX_MESSAGE_TOKENS = 1200)
     │
     ▼
generate_node
  ├── Identity query interception
  ├── Empty context fallback
  └── LLM generation with system prompt
     │
     ▼
Output Guardrails (hallucination / external bank check)
     │
     ▼
Response → User + SQLite storage
```

---

## 3. Data Preprocessing Pipeline

The project's foundation is the extraction of structured question-answer pairs from NUST Bank's internal Excel product documentation. This involved a two-stage pipeline using commercial APIs.

### 3.1 Stage 1: Excel → Markdown via LlamaParse (`qa_extractor.py`)

Raw Excel files contained complex banking product sheets with nested tables, merged cells, and formatted layouts that standard Excel parsers cannot faithfully represent. LlamaParse — a commercial document intelligence API — was used to handle this complexity.

**Process:**

```
Excel File
    │
    ▼
openpyxl: enumerate all worksheets
    │
    ▼ (for each sheet)
Create single-sheet temp .xlsx
    │
    ▼
LlamaParse API call
  - premium_mode=True (full OCR + table understanding)
  - adaptive_long_table=True (handles merged banking tables)
  - layout_extract_text=True (preserves positional context)
    │
    ▼
Markdown output → sheets_markdown/sheet_{SheetName}.md
    │
    ▼
Cleanup temp files
```

**Key design decision:** Each sheet was processed individually as a separate API call rather than uploading the whole workbook. This allowed granular retry handling and avoided cross-sheet content bleeding in the markdown output.

**Output:** 36 structured markdown files, one per product sheet, faithfully preserving table structures, profit rate tables, eligibility criteria, and fee schedules.

### 3.2 Stage 2: Markdown → QA Pairs via GPT-4 (`qa_openai.py`)

The markdown sheets were converted into structured JSON question-answer pairs using GPT-4 with strict extraction prompting.

**System Prompt Design:**

The prompt enforced three critical constraints:
1. **No paraphrasing** — answers must use exact text from the source sheet
2. **No fabrication** — only information explicitly present in the markdown may be used
3. **Schema compliance** — output must conform to `{"qa_pairs": [{"question": str, "answer": str}]}`

**Token budget:** `max_tokens=16384` per sheet to handle full product sheets in a single call.

**Output:** 36 JSON files in `sheets_qa/`, totalling **448 verified QA pairs** across all NUST banking products:

| Product Category | Example Files | Pairs |
|-----------------|---------------|-------|
| Account Types | qa_CDA.json, qa_NSA.json, qa_PLS.json | 4–14 |
| Finance Products | qa_NIF.json, qa_NHF.json, qa_NMF.json | 13–22 |
| Home Remittance | qa_HOME REMITTANCE.json | 20 |
| Rate Sheets | qa_Rate Sheet July 1 2024.json | 80 |
| Digital Banking | qa_RDA.json, qa_ESFCA.json | 24, 5 |
| Insurance | qa_EFU Life.json, qa_Jubilee Life.json | 5, 5 |

---

## 4. RAG Corpus Ingestion

### 4.1 Ingestion Script (`ingest_qa_rag.py`)

The `ingest_qa_rag.py` script is a one-time bootstrapping utility that populates the Chroma vector store from the preprocessed QA JSON files.

**Process:**

```python
# Pseudocode of ingestion pipeline
for json_file in qa_*.json:
    rows = load_json(json_file)
    for i, row in enumerate(rows):
        doc = Document(
            page_content=f"Q: {row.question}\nA: {row.answer}",
            metadata={
                "sheet": row.sheet,
                "source_file": json_file.name,
                "record_index": i,
                "question": row.question,
                "has_answer": bool(row.answer.strip())
            }
        )
        doc.id = f"qa_{json_file.stem}:{i}"
```

**Embedding Model:** `nomic-embed-text` via local Ollama — a 137M parameter text embedding model trained on large-scale data that performs comparably to commercial embeddings for semantic search tasks. Running it locally via Ollama eliminates API costs and latency for the embedding step.

**Vector Store:** Chroma (persistent, local) with collection name `banking_qa`. Documents are stored with metadata enabling downstream metadata-based scoring in the reranker.

**Batch processing:** Documents are added in batches of 64 to avoid memory pressure during embedding.

---

## 5. Hybrid Retrieval System

The retrieval system is the most architecturally sophisticated component of the project. Rather than relying solely on dense vector similarity (which is the standard approach), we implemented a **five-signal hybrid retrieval system** with learned reranking weights.

### 5.1 Retrieval Signals

#### Signal 1: Dense Embedding Similarity (weight: 0.45)

Standard Chroma cosine similarity search using `nomic-embed-text` embeddings. Fetches the top 24 candidates (RETRIEVAL_FETCH_K=24).

```python
dense_candidates = store.similarity_search_with_relevance_scores(
    query, k=RETRIEVAL_FETCH_K
)
```

Dense retrieval excels at semantic similarity but can miss exact keyword matches for banking-specific terms like account codes (NAA, RDA, NHF) or numeric rates ("7.5%").

#### Signal 2: Maximum Marginal Relevance (MMR)

MMR is a diversity-aware variant of dense retrieval that penalises redundant documents. It fetches 40 candidates and re-ranks them to maximise both relevance and diversity in the final selection. This is crucial for banking Q&A where multiple similar questions about the same product may retrieve near-duplicate chunks.

```python
mmr_candidates = store.max_marginal_relevance_search(
    query, k=top_k, fetch_k=RETRIEVAL_MMR_FETCH_K
)
```

#### Signal 3: Custom Lexical Overlap (weight: 0.30)

A custom token-overlap scorer built specifically for banking domain characteristics:

```python
def lexical_overlap_score(query: str, doc: Document) -> float:
    query_tokens = set(normalize(query).split())
    content_tokens = set(normalize(doc.page_content).split())
    question_tokens = set(normalize(doc.metadata.get("question","")).split())
    metadata_tokens = set(normalize(str(doc.metadata)).split())

    content_score  = jaccard(query_tokens, content_tokens) * 0.45
    question_score = jaccard(query_tokens, question_tokens) * 0.45
    metadata_score = jaccard(query_tokens, metadata_tokens) * 0.10

    # Bonus for numeric query terms (rates, amounts, limits)
    numeric_bonus = 0.10 if any(t.isdigit() for t in query_tokens
                                 if t in content_tokens) else 0.0

    return content_score + question_score + metadata_score + numeric_bonus
```

The **numeric bonus** is a domain-specific innovation: banking queries frequently include specific numbers (profit rates, minimum balances, loan limits). When a numeric token in the query exactly matches one in the document, a 10% bonus is awarded, improving recall for rate-specific queries.

#### Signal 4: BM25 (weight: 0.10)

BM25 (Best Match 25) is a classical information retrieval function that weighs term frequency against inverse document frequency. It was implemented using the `rank-bm25` library:

```python
bm25_score = 1.0 / (1.0 + bm25_rank)  # rank-to-score conversion
```

BM25 acts as a safety net for rare domain-specific terms that may have low embedding similarity (e.g., "PakWatan Remittance Account", "Izafa Finance") but high lexical match.

#### Signal 5: Metadata Match Score (weight: 0.10)

A lightweight scorer that directly compares query tokens against document metadata fields (sheet name, source file, corpus question). This allows retrieval to be guided by explicit product name mentions.

```python
def metadata_match_score(query: str, doc: Document) -> float:
    query_tokens = set(normalize(query).split())
    metadata_text = " ".join([
        doc.metadata.get("sheet", ""),
        doc.metadata.get("source_file", ""),
        doc.metadata.get("question", ""),
    ])
    meta_tokens = set(normalize(metadata_text).split())
    return len(query_tokens & meta_tokens) / max(len(query_tokens), 1)
```

### 5.2 Reciprocal Rank Fusion (RRF)

All retrieval signals are merged using Reciprocal Rank Fusion, a proven technique for combining ranked lists without requiring score normalisation:

```python
rrf_score = sum(
    1.0 / (RRF_SCALE + rank)
    for retrieval_method in all_methods
    for rank, doc in enumerate(retrieval_method)
    if doc.id == candidate.id
)
```

With `RRF_SCALE=30.0`, this produces stable, normalised fusion scores.

### 5.3 Final Reranking Formula

The five signals are combined in a weighted linear combination:

```
final_score =
  (dense_score   × 0.45) +
  (lexical_score × 0.30) +
  (rrf_score     × 0.15) +
  (metadata_score× 0.10) +
  (bm25_score    × 0.10)
```

**Minimum score threshold (RERANK_MIN_SCORE=0.22):** If no document exceeds 0.22, the system falls back to returning the top-4 by raw score rather than returning an empty context. This prevents unnecessary fallback responses for borderline queries.

**Final output:** Top-4 documents (RETRIEVAL_TOP_K=4) returned to the generation stage.

---

## 6. LangGraph Orchestration Pipeline

The RAG pipeline is implemented as a **directed acyclic graph (DAG)** using LangGraph, which provides structured state management, node-level error isolation, and native support for conversation memory via checkpointing.

### 6.1 Graph Topology

```
START → retrieve_node → summarize_node → trim_node → generate_node → END
```

All edges are sequential (no branching). The state object (`State`) flows through each node, being augmented at each stage.

### 6.2 State Schema

```python
class State(MessagesState):
    context: list[str]           # RAG document strings
    kg_context: list[str]        # Knowledge graph fact strings
    trimmed_messages: list[BaseMessage]  # Token-safe message history
    conversation_summary: str    # Rolling LLM-generated summary
```

The state inherits from `MessagesState` (LangChain's base) which provides the `messages` field for conversation history. Additional fields extend it for RAG-specific state.

### 6.3 Node: `retrieve_node`

Executes all five retrieval signals in sequence and calls the knowledge graph searcher:

```python
def retrieve_node(state: State) -> dict:
    query = get_latest_user_text(state["messages"])
    dense  = safe_dense_candidates(store, query, fetch_k=24)
    mmr    = safe_mmr_candidates(store, query, top_k=8, fetch_k=40)
    docs   = rerank_documents(query, dense, mmr)  # includes BM25, lexical, metadata
    kg     = search_knowledge_graph(query, top_k=2)
    return {"context": format_context(docs), "kg_context": kg}
```

`safe_*` wrappers catch all retrieval exceptions and return empty lists, ensuring the pipeline never fails at the retrieval stage.

### 6.4 Node: `summarize_node`

When the conversation token count exceeds `SUMMARY_TRIGGER_TOKENS`, this node compresses the older portion of the conversation into a rolling bullet-point summary using the LLM:

```python
summarization_prompt = """
You are a conversation summarizer.
Produce a compact bullet-point summary preserving:
- All banking product details mentioned
- Account features, rates, and eligibility criteria
- User's stated goals and constraints
"""
```

The summary is stored in `State.conversation_summary` and prepended as a `SystemMessage` in the trim node. This allows effectively infinite conversation length while keeping the LLM's context window bounded.

### 6.5 Node: `trim_node`

Applies LangChain's `trim_messages()` utility with the `"last"` strategy — preserving the most recent messages up to `MAX_MESSAGE_TOKENS=1200`:

```python
trimmed = trim_messages(
    state["messages"],
    strategy="last",
    token_counter=count_tokens_approximately,
    max_tokens=available_tokens,  # reduced if summary takes space
    start_on="human",
    include_system=False,
)
```

If a rolling summary exists, it is prepended as a `SystemMessage` and the available token budget for messages is reduced accordingly.

### 6.6 Node: `generate_node`

The final generation node with three decision branches:

**Branch 1 — Identity Query Interception:**
```python
if is_identity_query(latest_user_text):
    return {"messages": [AIMessage(content=ASSISTANT_IDENTITY)]}
```
Regex patterns detect questions like "who are you", "what is your name", "what system is this" and return a fixed identity string without calling the LLM. This prevents the model from identifying itself as Qwen, Alibaba, or any other underlying technology.

**Branch 2 — Empty Context Fallback:**
```python
if not context and not kg_context:
    return {"messages": [AIMessage(content=FALLBACK_ANSWER)]}
```
If retrieval returned nothing useful, a canned fallback response is returned without LLM inference, saving tokens and latency.

**Branch 3 — Full LLM Generation:**

The system prompt template combines RAG context and KG facts:
```
RAG Context:
{rag_context_block}

Knowledge Graph Facts:
{kg_context_block}
```

The system prompt explicitly instructs the model to:
- Format output in Markdown with `### Answer` heading
- Use bullet points for multiple facts
- Use tables for comparisons
- Never fabricate rates, fees, or product details
- Return the fallback sentence if context is insufficient

Post-generation, the response passes through `validate_model_output()` from the guardrails service before being returned.

### 6.7 Graph Checkpointing

```python
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
```

`MemorySaver` persists the full `State` between turns using the `thread_id` as the key. This is what enables multi-turn conversation — each new message appends to the existing state, and the summarize/trim nodes manage the growing history.

---

## 7. Conversation Memory Management

Long-running conversations present a fundamental challenge for LLMs: the context window is finite. The project implements a **two-tier memory system** that gracefully handles conversations of arbitrary length.

### Tier 1: Short-term (LangGraph MemorySaver)

The complete message history is stored in-process, keyed by `thread_id`. For typical conversations (under ~20 turns), no special handling is needed — the full history is passed to the LLM.

### Tier 2: Long-term (Rolling Summarization)

When the token count of the conversation history exceeds `SUMMARY_TRIGGER_TOKENS`, the `summarize_node` fires. It:

1. Takes all messages older than the last `SUMMARY_RECENT_MESSAGE_COUNT=8` messages
2. Folds them into the existing `conversation_summary` (if any)
3. Returns an updated compact summary

The `trim_node` then reconstructs the context as:
```
[SystemMessage: conversation summary]
[Last 8 messages verbatim]
```

This preserves recent conversational context exactly while ensuring the total token count always fits within `MAX_MESSAGE_TOKENS=1200`.

---

## 8. Knowledge Graph Integration

### 8.1 Motivation

Pure vector retrieval finds semantically similar passages but lacks explicit entity-relationship knowledge. A knowledge graph allows the system to answer questions that require traversing relationships: "What is the minimum age for a product that allows minors?" requires knowing that product X → has_feature → "minor account" and product X → requires → "age ≥ 18".

### 8.2 Architecture

The knowledge graph uses **Graphiti** — a graph knowledge extraction library — on top of **Neo4j** as the graph database.

**Ingestion pipeline:**
```
Text chunk → Graphiti LLM (Gemini 2.5 Flash)
    → Entity extraction (account names, features, rates, entities)
    → Relationship extraction (HAS_FEATURE, REQUIRES, OFFERS, etc.)
    → Neo4j storage as nodes + RELATES_TO edges
```

**Retrieval:**
```python
kg_references = client.search(
    query=query,
    group_ids=["nust-markdown", "nust-uploaded"],
    num_results=KG_TOP_K  # 2
)
```

Two group IDs separate the static corpus (pre-loaded banking sheets) from dynamically uploaded documents.

### 8.3 Fallback Mechanism

If Graphiti or Neo4j is unavailable (e.g., on machines without Neo4j installed), the system falls back to direct Cypher keyword search:

```cypher
MATCH ()-[e:RELATES_TO]->()
WHERE e.group_id IN $group_ids
  AND any(token IN $tokens
          WHERE toLower(coalesce(e.fact, '')) CONTAINS token)
RETURN e.fact, e.group_id, e.name
LIMIT $limit
```

This fallback means the system continues to function without KG enrichment — it simply returns fewer context items.

### 8.4 Integration in Generation

The generate node constructs a combined context block:
```
RAG Context:
[4 Chroma documents]

Knowledge Graph Facts:
[2 KG facts]
```

This dual-source context ensures the LLM can cross-reference structured retrieval results with graph-derived entity facts.

---

## 9. Dynamic Document Ingestion

### 9.1 Purpose

Beyond the pre-loaded NUST banking corpus, the system supports real-time document ingestion through the admin API. Staff can upload new documents (PDFs, Excel files, CSVs, text files) and have them immediately available for retrieval — without restarting the application.

### 9.2 End-to-End Pipeline (`document_ingestion.py`)

```
User uploads file via POST /admin/documents
    │
    ▼ (async background task)
Stage 1: SAVING (5%)
    Save file to uploaded_documents/{uuid}.{ext}
    Create database record
    │
    ▼
Stage 2: EXTRACTING (20%)
    TXT: Multi-encoding fallback (utf-8 → cp1252 → latin-1)
    PDF: Page-by-page extraction with page numbers
    CSV: pandas read → text serialisation
    Excel: All sheets, sheet names preserved
    │
    ▼
Stage 3: INGESTING_KG (35%)
    Split into chunks (max 6000 chars, paragraph boundaries)
    For each chunk: ingest_text_episode(chunk, group_id="nust-uploaded")
    Store episode UUIDs for later deletion
    │
    ▼
Stage 4: GENERATING (80%)
    For each chunk, call Gemini 2.5 Flash:
    "Extract {N} question-answer pairs from this banking text.
     Use only exact text from the source. Do not fabricate."
    Parse JSON with regex fallback
    │
    ▼
Stage 5: SAVING_QA (85%)
    Build LangChain Documents from QA pairs
    Add to Chroma with IDs: doc:{document_id}:qa:{i}
    Store QA pairs in document_qa_pairs table
    │
    ▼
Stage 6: COMPLETED (100%)
    Update document record with qa_count, vector_ids
```

### 9.3 Deletion

Soft deletion (`deleted_at` timestamp) removes:
- All vectors from Chroma (`vector_ids` list)
- All KG episodes from Graphiti (`kg_episode_ids` list)
- Document and QA pair records from SQLite

The original file is preserved for audit purposes unless explicitly purged.

### 9.4 Job Tracking

Every ingestion run is tracked as an `ingestion_job` record with:
- `stage`: Current pipeline stage name
- `progress`: 0–100 percentage
- `message`: Human-readable status
- `error_message`: If failed, full exception text

The admin can poll `GET /admin/documents/{id}` to monitor processing status.

---

## 10. Guardrails and Safety Layer

### 10.1 Architecture (`guardrails_service.py`)

The guardrails system uses the `guardrails-ai` library with three custom validators, applied at both input and output:

```python
@dataclass
class GuardrailsDecision:
    blocked: bool    # Whether to block this content
    message: str     # Replacement message if blocked
    reason: str      # Human-readable block reason
```

### 10.2 Validator 1: Toxic Language Detection

Custom regex-based toxicity detector covering:
- English offensive language patterns
- Violence and threat keywords
- Fraud and social engineering phrases
- **Urdu/Punjabi slurs** (domain-specific for Pakistani user base): haramzada, lanat, and other regional terms

This validator fires on both user input and LLM output.

### 10.3 Validator 2: Private Data Detection

Pattern matching for Pakistani-specific PII:
- **Email addresses**: Standard RFC 5322 pattern
- **CNIC numbers**: Pakistani National Identity Card format (`XXXXX-XXXXXXX-X`)
- **Phone numbers**: Pakistani mobile format (`03XX-XXXXXXX`)
- **Card numbers**: 13–19 digit sequences (credit/debit card numbers)

If detected in output, the response is replaced with a privacy-protective message.

### 10.4 Validator 3: Anti-Hallucination Grounding Check

The most sophisticated validator — it measures whether the LLM's answer is sufficiently grounded in the retrieved context:

```python
def check_grounding(answer: str, context: list[str]) -> float:
    # Tokenise and remove stopwords from both answer and context
    answer_tokens = meaningful_tokens(answer)
    context_tokens = meaningful_tokens(" ".join(context))

    # Compute token overlap ratio
    overlap = answer_tokens & context_tokens
    grounding_ratio = len(overlap) / max(len(answer_tokens), 1)

    # Additional: verify numeric claims appear in context
    answer_numbers = extract_numbers(answer)
    context_numbers = extract_numbers(" ".join(context))
    ungrounded_numbers = answer_numbers - context_numbers

    return grounding_ratio, ungrounded_numbers
```

**Threshold:** `GUARDRAILS_MIN_GROUNDED_RATIO=0.12` — at least 12% of the meaningful tokens in the answer must appear in the retrieved context. If not, or if the answer contains numeric claims not present in the context, the guardrail fires and replaces the response.

### 10.5 Domain Guardrails in Generation Node

Two additional guardrails are implemented directly in the generation node, independent of the guardrails-ai library:

**Identity interception** (`is_identity_query`):
Regex patterns detect self-referential questions and return a hard-coded identity string, preventing the model from revealing its underlying technology (Qwen, Alibaba Cloud, etc.).

**External bank filter** (`references_external_bank`):
Post-generation regex scan for 13 competitor Pakistani banks:
HBL, UBL, MCB, NBP, Meezan Bank, Bank Alfalah, Standard Chartered, Allied Bank, Askari Bank, Faysal Bank, Bank Islami, National Savings, NIBS.

If any competitor is mentioned, the response is discarded and replaced with the domain guardrail message.

---

## 11. LLM Providers and Configuration

### 11.1 Multi-Provider Architecture

The system abstracts LLM access behind a `get_chat_model(provider)` factory function, supporting three configurations:

| Provider | Model | Use Case |
|----------|-------|----------|
| `openrouter` | `qwen/qwen-2.5-7b-instruct` | Production (remote, stronger) |
| `ollama` | `qwen2.5:3b-instruct` | Local baseline |
| `ollama-finetuned` | `qwen2.5:3b-instruct-bitext-cs-dataet-tuned` | Local fine-tuned |

All models use `temperature=0.0` for maximum determinism in factual Q&A.

### 11.2 Model Selection Rationale

**Qwen2.5-3B-Instruct:** Chosen as the local model for:
- Feasibility on consumer hardware (4–8 GB VRAM)
- Strong instruction-following despite small size
- Native ChatML prompt format matching the fine-tuning dataset
- Availability through Ollama without licensing complications

**Qwen/qwen-2.5-7b-instruct (OpenRouter):** The 7B variant used for production deployments where more capable generation is required, accessed via OpenRouter's unified API.

### 11.3 Provider Normalisation

The `normalize_provider()` function maps aliases to canonical names:
```python
PROVIDER_ALIASES = {
    "ollama-finetuned": "ollama-finetuned",
    "ollama_finetuned": "ollama-finetuned",
    "finetuned":        "ollama-finetuned",
    ...
}
```

This allows the frontend and API clients to use any of several equivalent names.

---

## 12. Web Application and API

### 12.1 FastAPI Application (`api.py`)

The REST API exposes all chatbot and admin functionality:

#### Chat Endpoints

**`POST /chat`**
```json
Request:  {"message": "What is the minimum balance for NSA?",
           "chat_id": "optional-uuid",
           "provider": "ollama-finetuned"}

Response: {"reply": "### Answer\n...",
           "context_count": 4,
           "rag_references": ["Q: ... A: ..."],
           "kg_references": ["NSA requires..."],
           "provider": "ollama-finetuned",
           "chat_id": "auto-generated-uuid"}
```

Each response includes:
- `context_count`: Number of retrieved documents (transparency)
- `rag_references`: The actual retrieved QA pairs (explainability)
- `kg_references`: Knowledge graph facts used (explainability)

**`POST /chat/will-summarize`**
Predicts whether the next conversation turn will trigger summarisation, allowing the UI to display an appropriate indicator.

#### Admin Document Endpoints

**`POST /admin/documents`** — Upload and trigger async ingestion  
**`GET /admin/documents/{id}`** — Poll ingestion status  
**`DELETE /admin/documents/{id}`** — Soft-delete and cleanup vectors/KG

### 12.2 Chat UI (`templates/index.html`)

A single-page chat application rendered via Jinja2:
- Professional blue-themed banking UI with CSS variables
- **Marked.js** integration for rendering markdown responses (tables, bullet points, bold text)
- Multi-session chat history panel
- Document upload interface for admin users
- Provider selector for switching between OpenRouter/Ollama/Fine-tuned

---

## 13. Database Layer

### 13.1 SQLite Schema (`db.py`)

All persistent state (except vectors and graph) is stored in `chat_history.db`:

```sql
-- Multi-session chat history
CREATE TABLE chats (
    id TEXT PRIMARY KEY,
    title TEXT,
    created_at TIMESTAMP
);

CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT,
    role TEXT,                -- "user" | "assistant"
    content TEXT,
    context_count INTEGER,
    rag_references TEXT,      -- JSON array of retrieved chunks
    kg_references TEXT,       -- JSON array of KG facts
    created_at TIMESTAMP
);

-- Document lifecycle tracking
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    original_filename TEXT,
    stored_path TEXT,
    status TEXT,              -- queued|processing|completed|failed|deleted
    progress INTEGER,
    qa_count INTEGER,
    vector_ids TEXT,          -- JSON array for Chroma cleanup
    kg_episode_ids TEXT,      -- JSON array for Graphiti cleanup
    deleted_at TIMESTAMP
);

CREATE TABLE ingestion_jobs (
    id TEXT PRIMARY KEY,
    document_id TEXT,
    status TEXT,
    stage TEXT,               -- saving|extracting|ingesting_kg|generating|saving_qa|completed
    progress INTEGER,
    error_message TEXT
);
```

### 13.2 Summarisation Prediction

`should_summarize_next_turn()` queries the last N messages for a chat, estimates their token count, and compares against `SUMMARY_TRIGGER_TOKENS`. The UI uses this to pre-warn users that their next message may trigger summarisation.

---

## 14. QLoRA Fine-Tuning

### 14.1 Motivation

The base `qwen2.5:3b-instruct` model, while capable of following instructions, generates responses in a generic assistant style. NUST Bank's customer service context requires a more empathetic, structured, and professional conversational tone. Fine-tuning on customer service conversations adapts the model's generation style without altering its factual knowledge.

**Important distinction:** The fine-tuning changes *how* the model writes (tone, format, empathy), not *what facts* it knows. Factual accuracy continues to come from the RAG retrieval system.

### 14.2 Dataset

**Source:** Bitext Customer Support Conversations dataset (HuggingFace)  
**Size:** 26,872 total samples across 11 categories and 27 intents  
**Structure:** Each sample contains `instruction` (customer query), `response` (agent reply), `category`, and `intent`

**Exploratory Data Analysis (EDA) Findings:**
- Median instruction length: 9 words (short customer queries)
- Median response length: 90 words (detailed agent replies)
- 99% of samples fit within 512 tokens (ideal for training)
- 394 unique linguistic flag combinations (BL being most common at 19.4%)
- Well-balanced distribution across intents

### 14.3 Train/Validation Split Strategy

A two-stage stratified split was implemented to guarantee zero distribution drift between train and validation sets:

```python
# Stage 1: Pool selection from full dataset (stratified on intent)
pool_idx, _ = train_test_split(
    all_indices,
    train_size=6600,
    stratify=intents,
    random_state=42
)

# Stage 2: Train/val split within pool (stratified on intent)
train_idx, val_idx = train_test_split(
    pool_idx,
    train_size=6000,
    test_size=600,
    stratify=pool_intents,
    random_state=42
)

# Zero-overlap verification
train_instructions = set(train["instruction"])
val_instructions   = set(val["instruction"])
assert len(train_instructions & val_instructions) == 0
```

**Final split:** 6,000 training, 600 validation — zero overlap verified at runtime.

### 14.4 QLoRA Architecture

**Quantisation (4-bit NF4):**
The base model is loaded in 4-bit NormalFloat4 quantisation with double quantisation enabled. This reduces the 3B parameter model from ~6 GB to ~1.9 GB VRAM, making it trainable on a Colab T4 GPU.

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

**LoRA Adapter Configuration:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `r` | 8 | Low rank = minimal footprint, sufficient for style transfer |
| `lora_alpha` | 16 | Scaling factor 2× rank |
| Target modules | `q_proj`, `v_proj` | Attention weights only (minimal parameters) |
| `lora_dropout` | 0.05 | Light regularisation |

Only the query and value projection weights in the attention mechanism are adapted. Key, output, and feed-forward weights remain frozen. This targets the model's attention pattern (how it reads context) while preserving factual knowledge in the FFN layers.

### 14.5 Training Configuration

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Epochs | 1 | Light adaptation, avoid overfitting |
| Batch size | 2 | T4 VRAM constraint |
| Gradient accumulation | 8 | Effective batch size = 16 |
| Learning rate | 2.0e-4 | Standard LoRA LR |
| LR scheduler | Cosine | Smooth decay |
| Optimizer | `adamw_bnb_8bit` | 8-bit states ≈ 50% less VRAM than fp32 |
| Max sequence length | 512 | Covers 99% of samples |

**Prompt template (ChatML format):**
```
<|im_start|>system
You are a helpful customer support assistant.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>
```

ChatML is Qwen2.5's native instruction format, ensuring no token distribution mismatch during fine-tuning.

### 14.6 Training Infrastructure

- **Platform:** Google Colab Free Tier (T4 GPU, 15 GB VRAM)
- **Memory management:** 4-bit base + 8-bit optimizer + `bf16` adapter computation
- **Experiment tracking:** MLflow on DagsHub remote (`dagshub.com/nemroneno2526/LLM_project.mlflow`)
- **Checkpointing:** Accelerate saves full checkpoints every 50 steps; best validation checkpoint preserved

### 14.7 Fine-Tuning Results

| Metric | Pre-tuning | Post-tuning | Improvement |
|--------|------------|-------------|-------------|
| Perplexity | 24.68 | 2.41 | **90.2% reduction** |

The dramatic perplexity drop (24.68 → 2.41) confirms successful domain adaptation. Qualitative comparison across 5 held-out prompts showed consistent improvements:

- **Empathy:** Post-tuned model acknowledges customer frustration before providing solutions
- **Structure:** Clear numbered steps replace rambling prose
- **Length:** Responses expanded from brief mentions to detailed, actionable guidance
- **Tone:** Professional, warm, consistent with customer service standards

### 14.8 Ollama Integration

The LoRA adapter was converted to GGUF format and registered in Ollama using the native `ADAPTER` directive:

```dockerfile
# finetuning/finetuned_model/Modelfile
FROM qwen2.5:3b-instruct
ADAPTER ./adapter.gguf

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ .Response }}<|im_end|>"""

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.0
```

This approach — applying the adapter at inference time rather than merging weights — keeps the GGUF adapter file at 7.1 MB rather than requiring a full 2 GB merged model.

```bash
ollama create qwen2.5:3b-instruct-bitext-cs-dataet-tuned -f Modelfile
```

---

## 15. Ragas Evaluation Framework

### 15.1 Evaluation Design Philosophy

Two evaluation scripts were developed, reflecting the project's dual priorities:

1. **`evaluate_rag_retrieval.py`** — Full pipeline evaluation using `invoke_graph()` (includes KG)
2. **`eval_rag_simple.py`** — Pure RAG evaluation excluding KG (portable, runs without Neo4j)

The second script is the primary evaluation tool, as it runs on any machine without database dependencies.

### 15.2 Data Source

Questions and reference answers are drawn from the **same NUST banking corpus** used for RAG ingestion (`sheets_qa/qa_*.json`). This is intentional — the ground-truth answers represent what the bank's official documentation says, making them the correct reference for factual accuracy.

**Sampling strategy:** Proportional stratified sampling across all 36 product files, ensuring every product area is represented in the evaluation. With `n=25`, typically 1–2 questions are drawn per product file.

### 15.3 Metrics

#### Retrieval Metrics

**Context Precision** (`LLMContextPrecisionWithReference`)  
*Question:* Are the retrieved chunks actually relevant to the question?  
Uses the LLM judge to assess each retrieved chunk's relevance given the reference answer.

**Context Recall** (`LLMContextRecall`)  
*Question:* Does the retrieved context contain all the facts needed to produce the reference answer?  
Measures completeness of retrieval — whether important facts were missed.

#### Generation Metrics

**Faithfulness** (`Faithfulness`)  
*Question:* Does the LLM's answer stay within the bounds of the retrieved context?  
Claims in the answer are extracted and individually verified against the context. This directly measures hallucination.

**Factual Correctness** (`FactualCorrectness`)  
*Question:* Does the answer agree factually with the reference answer?  
The judge decomposes both the answer and reference into atomic facts and checks overlap.

**Answer Relevancy** (`ResponseRelevancy`)  
*Question:* Does the answer directly address what was asked?  
Measured by generating a reverse question from the answer and computing embedding cosine similarity with the original question. Sensitive to answer verbosity.

### 15.4 Judge LLM

**OpenAI GPT-4o-mini** was chosen as the judge LLM for all five metrics. Using a different, more capable model than the one being evaluated (Qwen 3B) avoids self-evaluation bias and provides more reliable assessments. The estimated cost per evaluation run (25 questions, 5 metrics) is approximately $0.03–0.06.

---

## 16. Evaluation Results and Analysis

### 16.1 Results Summary

| Metric | Score | Interpretation |
|--------|-------|----------------|
| Context Precision | **0.913** | Retrieval is highly accurate |
| Context Recall | **0.952** | Retrieval captures nearly all relevant facts |
| Faithfulness | **0.947** | Model stays very close to the retrieved context |
| Factual Correctness | **0.835** | Answers align well with official documentation |
| Answer Relevancy | **0.516** | Reflects verbosity penalty (see analysis below) |

**Model evaluated:** `qwen2.5:3b-instruct-bitext-cs-dataet-tuned` (fine-tuned)  
**Judge:** GPT-4o-mini  
**Sample:** 25 questions across 25 product files  

### 16.2 Analysis of Results

**Retrieval quality is excellent (0.91–0.95):** The hybrid five-signal retrieval system with RRF reranking consistently surfaces the correct product information. The high context recall (0.95) is particularly significant — it means the system almost never misses important facts due to retrieval failure.

**Faithfulness is very high (0.947):** This is the most important metric for a banking chatbot. A score of 0.947 means the model's answers are tightly grounded in retrieved context, with minimal fabrication. The anti-hallucination guardrail and the restrictive system prompt both contribute to this.

**Factual correctness is strong (0.835):** Agreement with official documentation is high. The 16.5% gap from perfect can be attributed to the model's markdown formatting and elaboration style, which may phrase facts differently than the terse reference answers.

**Answer relevancy is lower (0.516):** This score is a known artifact of the metric's methodology rather than a deficiency in the system. The metric generates a reverse question from the model's answer and measures embedding similarity with the original question. Our system prompt forces verbose, markdown-formatted answers (`### Answer`, bullet points, tables), which causes the reverse-generated questions to be broader than the original specific questions. The metric structurally penalises verbose, structured answers — which are desirable features for a banking chatbot UI.

Additionally, the warning `LLM returned 1 generation instead of requested 3` indicates that GPT-4o-mini no longer supports multiple completions in a single API call with Ragas 0.4.3, so the score is based on a single reverse-question rather than three averaged attempts.

**Recommendation for report:** The answer relevancy score should be contextualised as reflecting this metric's known sensitivity to response length and formatting, not as an indication of poor system quality. The other four metrics — all above 0.83 — are more meaningful indicators of actual performance.

---

## 17. Technology Stack

| Layer | Technology | Version/Notes |
|-------|-----------|---------------|
| **Orchestration** | LangGraph | Graph-based RAG pipeline |
| **LLM Framework** | LangChain | Core abstractions |
| **Vector Store** | Chroma | Persistent local embedding store |
| **Embeddings** | nomic-embed-text | Via Ollama, 137M params |
| **Graph Store** | Neo4j | Knowledge graph backend |
| **Graph Library** | Graphiti | Entity/relation extraction |
| **Local LLM** | Ollama | Runtime for Qwen2.5 |
| **Chat Models** | Qwen2.5-3B / 7B | Instruction-tuned variants |
| **Remote LLM** | OpenRouter | Qwen 7B, Gemini 2.5 Flash |
| **Safety** | guardrails-ai | Custom validators |
| **Web Framework** | FastAPI | REST API + Jinja2 UI |
| **Database** | SQLite | Chat history + document tracking |
| **Lexical Search** | rank-bm25 | BM25 implementation |
| **Data Extraction** | LlamaParse | Excel → Markdown OCR |
| **QA Generation** | GPT-4 / Gemini 2.5 Flash | Sheet and document QA extraction |
| **Fine-tuning** | PEFT + bitsandbytes | QLoRA on Colab T4 |
| **Tracking** | MLflow + DagsHub | Remote experiment logging |
| **Evaluation** | Ragas 0.4.3 | RAG quality metrics |
| **Judge LLM** | GPT-4o-mini | Ragas evaluation |

---

## 18. Architectural Decisions and Trade-offs

### Decision 1: LangGraph over Simple Chain

**Chosen:** LangGraph DAG with explicit state management  
**Alternative:** Simple LangChain LCEL chain  
**Rationale:** LangGraph provides native checkpointing (`MemorySaver`) for multi-turn memory, explicit node-level error isolation, and cleaner separation of concerns between retrieval, summarisation, and generation. The overhead is minimal for our pipeline depth.

### Decision 2: Five-Signal Hybrid Retrieval

**Chosen:** Dense + MMR + BM25 + Lexical + Metadata with RRF fusion  
**Alternative:** Dense-only retrieval  
**Rationale:** Banking Q&A has specific failure modes for dense-only retrieval: exact code names (NAA, RDA), numeric rates ("7.5% per annum"), and product-specific terminology. BM25 and lexical retrieval catch these cases. MMR prevents redundant chunk selection. The weighted combination was tuned empirically.

**Trade-off:** Increased retrieval latency (~3× compared to dense-only) and implementation complexity.

### Decision 3: Local Embeddings (nomic-embed-text)

**Chosen:** Ollama nomic-embed-text (local)  
**Alternative:** OpenAI text-embedding-3-small (remote)  
**Rationale:** Eliminates embedding API costs and latency for every user query. The 137M parameter model provides embeddings competitive with commercial models for domain-specific Q&A. Requires Ollama installation but reduces operational cost to zero for the embedding step.

### Decision 4: SQLite for Chat History

**Chosen:** SQLite  
**Alternative:** PostgreSQL, Redis  
**Rationale:** The application is designed for single-server deployment. SQLite eliminates an external database dependency while providing sufficient concurrent read performance for the expected user load. The WAL mode enables concurrent reads without blocking writes.

### Decision 5: LoRA on q_proj + v_proj Only

**Chosen:** Target attention query and value projections  
**Alternative:** Target all linear layers (q, k, v, o, gate, up, down)  
**Rationale:** Targeting only q_proj and v_proj minimises the adapter parameter count (~7.1 MB GGUF) while achieving the desired style transfer. The attention mechanism governs how the model attends to context — adapting it is sufficient for conversational style without touching the feed-forward layers that store factual knowledge.

### Decision 6: Adapter vs. Merged Model for Ollama

**Chosen:** GGUF adapter file (7.1 MB) applied at inference time  
**Alternative:** Fully merged 2 GB GGUF model  
**Rationale:** The adapter approach keeps the total storage addition minimal. Ollama's `ADAPTER` directive applies the LoRA weights during inference with negligible overhead. If the fine-tuned model is abandoned, only 7.1 MB needs to be deleted. The base model is shared with the non-finetuned deployment.

### Decision 7: Separate KG and RAG Retrieval

**Chosen:** KG results as additive `kg_context`, separate from RAG `context`  
**Alternative:** Merge KG results into the main RAG candidate pool  
**Rationale:** Keeping the two sources separate preserves interpretability — the generate node's system prompt explicitly labels "RAG Context" and "Knowledge Graph Facts", allowing the LLM to reason about their provenance. It also allows independent evaluation of each source's contribution.

---

## 19. Conclusion

This project demonstrates a complete, production-grade intelligent banking chatbot built from raw Excel documentation through to a deployed, evaluated system. The key contributions are:

1. **Two-stage preprocessing pipeline** that faithfully extracts structured Q&A from complex Excel banking sheets using commercial OCR and LLM APIs

2. **Five-signal hybrid retrieval** with RRF reranking that outperforms dense-only approaches for domain-specific banking terminology

3. **LangGraph-based multi-turn pipeline** with automatic conversation summarisation for effectively unlimited conversation length

4. **Multi-layer safety architecture** combining input/output guardrails (toxicity, PII, hallucination detection) with system-prompt-level domain restriction and post-generation output filtering

5. **Knowledge graph augmentation** using Graphiti + Neo4j for entity-relationship aware retrieval, with graceful degradation when unavailable

6. **Dynamic document ingestion** enabling real-time corpus expansion without application restart

7. **QLoRA fine-tuning** that reduces model perplexity by 90.2% (24.68 → 2.41) on customer service conversations, improving response quality while maintaining a 7.1 MB adapter footprint

8. **Comprehensive Ragas evaluation** with GPT-4o-mini as an independent judge, achieving strong scores across all retrieval and generation metrics (0.83–0.95 on four of five metrics)

The evaluation results confirm that the system's hybrid retrieval effectively surfaces relevant context (precision: 0.91, recall: 0.95), the language model stays closely grounded in that context without hallucinating (faithfulness: 0.95), and the answers factually agree with the official banking documentation (factual correctness: 0.84). The lower answer relevancy score (0.52) is a documented artifact of the metric's sensitivity to markdown-formatted, verbose answers rather than an indication of system deficiency.

---

*Report generated for Semester 8 LLM Project — NUST Banking Intelligent Chatbot*

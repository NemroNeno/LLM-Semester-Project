"""
eval_rag_simple.py
------------------
Comprehensive Ragas evaluation of the RAG pipeline (Chroma retrieval only).
GraphRAG / Knowledge Graph / Neo4j is intentionally excluded — safe to run locally.

WHERE THE QUESTIONS COME FROM
------------------------------
Questions and reference answers are loaded from the project's own QA corpus:
    sheets_qa/qa_*.json   (36 JSON files, 448 QA pairs total)

Each file corresponds to one NUST banking product sheet, for example:
  • qa_RDA.json          — Roshan Digital Account        (24 pairs)
  • qa_Rate Sheet ...    — Interest/profit rate sheet     (80 pairs)
  • qa_NMF.json         — NUST Mahana Finance            (21 pairs)
  • qa_NIF.json         — NUST Izafa Finance             (22 pairs)
  • qa_HOME REMITTANCE  — Remittance services            (20 pairs)
  • ... and 31 more product sheets

A random stratified sample is drawn across all files so every product area
is represented proportionally.  The reference answer from the JSON is used as
ground truth for Ragas reference-based metrics.

METRICS COMPUTED
-----------------
Retrieval:
  context_precision   — Are the retrieved chunks actually relevant to the question?
  context_recall      — Does the retrieved context contain all facts needed to answer?

Generation:
  faithfulness        — Does the LLM answer stay grounded in the retrieved context?
  answer_relevancy    — Is the LLM answer relevant to the question asked?
  factual_correctness — Does the LLM answer agree factually with the reference answer?

JUDGE LLM
----------
OpenAI gpt-4o-mini (requires OPENAI_API_KEY in .env)
Estimated cost: ~$0.03–0.06 for 25 questions with all 5 metrics.

CHATBOT
--------
Local Ollama model (OLLAMA_CHAT_MODEL from settings, default: qwen2.5:3b-instruct)
Pure Chroma retrieval path — no KG/Neo4j dependency.

SETUP
------
Add to .env:
    OPENAI_API_KEY=sk-...

Ollama must be running with nomic-embed-text pulled.

RUN
----
    uv run python eval_rag_simple.py               # 25 questions (default)
    uv run python eval_rag_simple.py --n 10        # smaller / cheaper
    uv run python eval_rag_simple.py --n 50 --output report.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from openai import OpenAI

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from ragas import EvaluationDataset, evaluate
from ragas.llms import llm_factory
from ragas.metrics import (
    Faithfulness,
    FactualCorrectness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

# ResponseRelevancy needs an embeddings model — handle both old and new Ragas API names
try:
    from ragas.metrics import ResponseRelevancy as _ResponseRelevancy
    _RELEVANCY_CLS = _ResponseRelevancy
except ImportError:
    try:
        from ragas.metrics import AnswerRelevancy as _AnswerRelevancy
        _RELEVANCY_CLS = _AnswerRelevancy
    except ImportError:
        _RELEVANCY_CLS = None

# Import only pure RAG functions — no invoke_graph, no KG
from rag import (
    SYSTEM_PROMPT_TEMPLATE,
    format_context,
    get_chat_model,
    get_vector_store,
    message_to_text,
    rerank_documents,
    safe_dense_candidates,
    safe_mmr_candidates,
)
from settings import (
    OLLAMA_FINETUNED_CHAT_MODEL,
    QA_DIRECTORY,
    RETRIEVAL_FETCH_K,
    RETRIEVAL_MMR_FETCH_K,
    RETRIEVAL_TOP_K,
)

load_dotenv()

JUDGE_MODEL      = "gpt-4o-mini"
EMBED_MODEL      = "text-embedding-3-small"   # for answer_relevancy metric
CHAT_PROVIDER    = "ollama-finetuned"         # uses OLLAMA_FINETUNED_CHAT_MODEL from settings
DEFAULT_N        = 25
SEED             = 42


# ── Data loading ──────────────────────────────────────────────────────────────

def load_qa_samples(qa_dir_str: str, n: int, seed: int) -> list[dict]:
    """
    Load all valid QA pairs from sheets_qa/ then draw a proportional random
    sample of n items so every product file is represented.

    Returns list of dicts with keys: question, reference, source_file, sheet.
    """
    qa_dir = Path(qa_dir_str)
    if not qa_dir.exists():
        raise FileNotFoundError(f"QA directory not found: {qa_dir.resolve()}")

    # Collect all valid pairs, keeping source metadata
    by_file: dict[str, list[dict]] = {}
    for json_file in sorted(qa_dir.glob("qa_*.json")):
        rows = json.loads(json_file.read_text(encoding="utf-8"))
        valid = [
            {
                "question":    str(r.get("question", "")).strip(),
                "reference":   str(r.get("answer", "")).strip(),
                "sheet":       str(r.get("sheet", json_file.stem)).strip(),
                "source_file": json_file.name,
            }
            for r in rows
            if isinstance(r, dict)
            and len(str(r.get("question", "")).strip()) > 5
            and len(str(r.get("answer", "")).strip()) > 10
        ]
        if valid:
            by_file[json_file.name] = valid

    all_items = [item for items in by_file.values() for item in items]
    total = len(all_items)
    if not all_items:
        raise ValueError(f"No valid QA pairs found in {qa_dir}")

    rng = random.Random(seed)

    # Proportional sampling: pick ceil(n * file_count / total) from each file
    selected: list[dict] = []
    remaining = n
    files = list(by_file.items())
    rng.shuffle(files)
    for i, (fname, items) in enumerate(files):
        files_left = len(files) - i
        take = max(1, round(remaining / files_left)) if remaining > 0 else 0
        take = min(take, len(items), remaining)
        shuffled = list(items)
        rng.shuffle(shuffled)
        selected.extend(shuffled[:take])
        remaining -= take
        if remaining <= 0:
            break

    rng.shuffle(selected)
    selected = selected[:n]   # guarantee exactly n

    # Print data provenance
    print("\n" + "="*55)
    print("  DATA SOURCE")
    print("="*55)
    print(f"  QA corpus:   {qa_dir.resolve()}")
    print(f"  Total pairs: {total} across {len(by_file)} product files")
    print(f"  Sample size: {len(selected)} (seed={seed})")
    files_in_sample = sorted({s['source_file'] for s in selected})
    print(f"  Files used:  {len(files_in_sample)}")
    for fname in files_in_sample:
        count = sum(1 for s in selected if s['source_file'] == fname)
        print(f"    • {fname:<40} {count} questions")
    print("="*55 + "\n")

    return selected


# ── RAG inference (no KG) ─────────────────────────────────────────────────────

def rag_answer(question: str, chat_model) -> tuple[str, list[str]]:
    """
    Pure Chroma retrieval + LLM. Returns (response_text, retrieved_contexts).
    No KG / Neo4j involved.
    """
    store = get_vector_store()

    dense = safe_dense_candidates(store, question, fetch_k=RETRIEVAL_FETCH_K)
    mmr   = safe_mmr_candidates(
        store, question,
        top_k=max(RETRIEVAL_TOP_K * 2, RETRIEVAL_TOP_K),
        fetch_k=RETRIEVAL_MMR_FETCH_K,
    )
    docs            = rerank_documents(question, dense, mmr)
    context_strings = format_context(docs)

    if not context_strings:
        fallback = "I do not have enough information in the provided banking data to answer that."
        return fallback, []

    system_content = SYSTEM_PROMPT_TEMPLATE.format(context="\n\n".join(context_strings))
    response = chat_model.invoke([
        SystemMessage(content=system_content),
        HumanMessage(content=question),
    ])
    return message_to_text(response), context_strings


# ── Ragas judge setup ─────────────────────────────────────────────────────────

def build_judge(openai_key: str):
    """OpenAI GPT-4o-mini as the Ragas evaluator LLM."""
    client = OpenAI(api_key=openai_key)
    return llm_factory(JUDGE_MODEL, provider="openai", client=client)


def build_embeddings(openai_key: str):
    """OpenAI text-embedding-3-small for answer_relevancy metric."""
    try:
        from ragas.embeddings import OpenAIEmbeddings
        client = OpenAI(api_key=openai_key)
        return OpenAIEmbeddings(client=client, model=EMBED_MODEL)
    except Exception as e:
        print(f"  [warn] Could not build embeddings: {e}")
        return None


def build_metrics(judge, embeddings) -> tuple[list, dict[str, str]]:
    """
    Build all 5 Ragas metrics.
    Returns (metric_objects, label→ragas_column_name map).
    """
    metrics = [
        LLMContextPrecisionWithReference(llm=judge),
        LLMContextRecall(llm=judge),
        Faithfulness(llm=judge),
        FactualCorrectness(llm=judge),
    ]
    col_map = {
        "context_precision":   "llm_context_precision_with_reference",
        "context_recall":      "llm_context_recall",
        "faithfulness":        "faithfulness",
        "factual_correctness": "factual_correctness",
    }

    # answer_relevancy needs embeddings; add if available
    if _RELEVANCY_CLS is not None and embeddings is not None:
        try:
            metrics.append(_RELEVANCY_CLS(llm=judge, embeddings=embeddings))
            col_map["answer_relevancy"] = "answer_relevancy"
        except TypeError:
            try:
                metrics.append(_RELEVANCY_CLS(llm=judge))
                col_map["answer_relevancy"] = "answer_relevancy"
            except Exception:
                print("  [warn] answer_relevancy metric could not be initialised — skipping")
    elif _RELEVANCY_CLS is None:
        print("  [warn] ResponseRelevancy/AnswerRelevancy not found in this Ragas version — skipping")

    return metrics, col_map


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive Ragas RAG evaluation (no GraphRAG/Neo4j)."
    )
    parser.add_argument("--n",      type=int,  default=DEFAULT_N, help=f"Questions to evaluate (default: {DEFAULT_N})")
    parser.add_argument("--seed",   type=int,  default=SEED,      help="Random seed (default: 42)")
    parser.add_argument("--output", type=Path, default=None,      help="JSON report output path")
    args = parser.parse_args()

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise SystemExit(
            "\nOPENAI_API_KEY is not set.\n"
            "Add it to your .env file:  OPENAI_API_KEY=sk-...\n"
        )

    # ── Load QA samples ────────────────────────────────────────────────────────
    samples = load_qa_samples(QA_DIRECTORY, n=args.n, seed=args.seed)

    # ── Load chatbot ───────────────────────────────────────────────────────────
    print(f"Chatbot  : {OLLAMA_FINETUNED_CHAT_MODEL}  (provider: {CHAT_PROVIDER})")
    print(f"Judge    : {JUDGE_MODEL}  (OpenAI)")
    print(f"Embedder : {EMBED_MODEL}  (OpenAI, for answer_relevancy)\n")
    chat_model = get_chat_model(CHAT_PROVIDER)

    # ── Run RAG over each question ─────────────────────────────────────────────
    print(f"Running RAG pipeline over {len(samples)} questions ...")
    ragas_rows: list[dict] = []
    for i, sample in enumerate(samples, start=1):
        q   = sample["question"]
        ref = sample["reference"]
        print(f"  [{i:>2}/{len(samples)}] [{sample['source_file']:<38}]  {q[:55]}{'...' if len(q)>55 else ''}")
        response, contexts = rag_answer(q, chat_model)
        ragas_rows.append({
            "user_input":         q,
            "response":           response,
            "reference":          ref,
            "retrieved_contexts": contexts,
        })

    dataset = EvaluationDataset.from_list(ragas_rows)

    # ── Build judge + metrics ──────────────────────────────────────────────────
    print("\nInitialising Ragas judge and metrics ...")
    judge      = build_judge(openai_key)
    embeddings = build_embeddings(openai_key)
    metrics, col_map = build_metrics(judge, embeddings)
    print(f"  Metrics: {', '.join(col_map.keys())}")

    # ── Evaluate ───────────────────────────────────────────────────────────────
    print("\nRunning Ragas evaluate() — this calls OpenAI API ...")
    result = evaluate(dataset=dataset, metrics=metrics, raise_exceptions=False)

    # ── Summarise ──────────────────────────────────────────────────────────────
    df      = result.to_pandas()
    summary: dict[str, float | None] = {}

    print("\n" + "="*55)
    print(f"  RAGAS RESULTS   judge={JUDGE_MODEL}   n={len(samples)}")
    print("="*55)
    print(f"  {'Metric':<28} {'Score':>8}")
    print(f"  {'-'*28} {'-'*8}")

    SECTION = {
        "context_precision":   "-- Retrieval --",
        "context_recall":      None,
        "faithfulness":        "-- Generation --",
        "answer_relevancy":    None,
        "factual_correctness": None,
    }

    printed_section: set[str] = set()
    for label, col in col_map.items():
        section = SECTION.get(label)
        if section and section not in printed_section:
            print(f"\n  {section}")
            printed_section.add(section)

        # Try exact column name then fuzzy match
        actual_col = col if col in df.columns else next(
            (c for c in df.columns if label.replace("_", "") in c.replace("_", "")), None
        )
        if actual_col and df[actual_col].dropna().size:
            val = float(df[actual_col].dropna().mean())
        else:
            val = None
        summary[label] = val
        display = f"{val:.4f}" if val is not None else "  n/a"
        print(f"  {label:<28} {display:>8}")

    print("\n" + "="*55)

    # ── Save report ────────────────────────────────────────────────────────────
    output_path = args.output or Path("eval_rag_report.json")

    files_in_sample = {}
    for s in samples:
        files_in_sample.setdefault(s["source_file"], []).append(s["question"])

    report = {
        "evaluation_summary": {
            "n_questions":  len(samples),
            "chat_model":   OLLAMA_FINETUNED_CHAT_MODEL,
            "judge_model":  JUDGE_MODEL,
            "embed_model":  EMBED_MODEL,
            "provider":     CHAT_PROVIDER,
            "seed":         args.seed,
        },
        "data_source": {
            "directory":    str(Path(QA_DIRECTORY).resolve()),
            "description":  "NUST banking product Q&A corpus (sheets_qa/qa_*.json)",
            "files_sampled": {
                fname: {"n_questions": len(qs), "questions": qs}
                for fname, qs in sorted(files_in_sample.items())
            },
        },
        "metrics": summary,
    }

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFull report saved → {output_path.resolve()}")


if __name__ == "__main__":
    main()

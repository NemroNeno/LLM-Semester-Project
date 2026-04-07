"""
eval_answer_relevancy.py
-------------------------
Computes answer_relevancy for 5 questions using the finetuned RAG pipeline,
then merges the score into eval_rag_report.json.

Only OpenAI API calls made are for the answer_relevancy metric itself
(LLM reverse-question generation + embeddings similarity).
Chatbot runs locally via Ollama — no extra OpenAI cost there.

Run:
    uv run python eval_answer_relevancy.py
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from openai import OpenAI
from ragas import EvaluationDataset, evaluate
from ragas.llms import llm_factory
from ragas.metrics import ResponseRelevancy
from langchain_openai import OpenAIEmbeddings as LangChainOpenAIEmbeddings

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
    RETRIEVAL_FETCH_K,
    RETRIEVAL_MMR_FETCH_K,
    RETRIEVAL_TOP_K,
)

load_dotenv()

REPORT_PATH   = Path("eval_rag_report.json")
JUDGE_MODEL   = "gpt-4o"
EMBED_MODEL   = "text-embedding-3-small"
CHAT_PROVIDER = "ollama-finetuned"
N_QUESTIONS   = 5


def rag_answer(question: str, chat_model) -> tuple[str, list[str]]:
    store    = get_vector_store()
    dense    = safe_dense_candidates(store, question, fetch_k=RETRIEVAL_FETCH_K)
    mmr      = safe_mmr_candidates(store, question,
                   top_k=max(RETRIEVAL_TOP_K * 2, RETRIEVAL_TOP_K),
                   fetch_k=RETRIEVAL_MMR_FETCH_K)
    docs     = rerank_documents(question, dense, mmr)
    contexts = format_context(docs)
    if not contexts:
        return "I do not have enough information in the provided banking data to answer that.", []
    response = chat_model.invoke([
        SystemMessage(content=SYSTEM_PROMPT_TEMPLATE.format(context="\n\n".join(contexts))),
        HumanMessage(content=question),
    ])
    return message_to_text(response), contexts


def main() -> None:
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise SystemExit("OPENAI_API_KEY not set in .env")

    # ── Load existing report to reuse the same questions ──────────────────────
    if not REPORT_PATH.exists():
        raise SystemExit(f"{REPORT_PATH} not found — run eval_rag_simple.py first.")

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    all_questions = [
        q
        for file_data in report["data_source"]["files_sampled"].values()
        for q in file_data["questions"]
    ]
    questions = all_questions[:N_QUESTIONS]
    print(f"Using {len(questions)} questions from existing report:")
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q[:70]}{'...' if len(q) > 70 else ''}")

    # ── Run RAG chatbot (Ollama — no OpenAI cost here) ─────────────────────────
    print(f"\nRunning RAG pipeline (finetuned Ollama model) ...")
    chat_model = get_chat_model(CHAT_PROVIDER)
    rows: list[dict] = []
    for i, q in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] generating answer ...")
        response, contexts = rag_answer(q, chat_model)
        rows.append({
            "user_input":         q,
            "response":           response,
            "retrieved_contexts": contexts,
        })

    dataset = EvaluationDataset.from_list(rows)

    # ── Build judge + embeddings (OpenAI — only calls made to OpenAI) ─────────
    print(f"\nInitialising OpenAI judge ({JUDGE_MODEL}) and embeddings ({EMBED_MODEL}) ...")
    client     = OpenAI(api_key=openai_key)
    judge      = llm_factory(JUDGE_MODEL, provider="openai", client=client)
    embeddings = LangChainOpenAIEmbeddings(model=EMBED_MODEL, api_key=openai_key)
    metric     = ResponseRelevancy(llm=judge, embeddings=embeddings, strictness=1)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    print("Running answer_relevancy evaluation (OpenAI API calls happening now) ...")
    result = evaluate(dataset=dataset, metrics=[metric], raise_exceptions=False)

    df  = result.to_pandas()
    col = next((c for c in df.columns if "relevancy" in c or "relevance" in c), None)
    if col and df[col].dropna().size:
        score = float(df[col].dropna().mean())
    else:
        score = None

    print(f"\n  answer_relevancy: {f'{score:.4f}' if score is not None else 'n/a'}")

    # ── Merge into existing report ─────────────────────────────────────────────
    report["metrics"]["answer_relevancy"] = score
    report["evaluation_summary"]["answer_relevancy_n"] = len(questions)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nUpdated {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    main()

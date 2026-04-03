from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is unavailable
    tqdm = None

load_dotenv()

QA_DIRECTORY = Path(os.getenv("QA_DIRECTORY", "sheets_qa"))
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "banking_qa")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

DEFAULT_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))
DEFAULT_FETCH_K = int(os.getenv("RETRIEVAL_FETCH_K", "24"))
DEFAULT_MMR_FETCH_K = int(os.getenv("RETRIEVAL_MMR_FETCH_K", "40"))
DEFAULT_BM25_FETCH_K = int(os.getenv("RETRIEVAL_BM25_FETCH_K", "24"))
OLLAMA_REQUEST_TIMEOUT = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "5"))


@dataclass(frozen=True)
class Example:
    query: str
    relevant_key: str
    sheet: str
    source_file: str
    record_index: int


@dataclass(frozen=True)
class RerankConfig:
    dense_weight: float
    lexical_weight: float
    rrf_weight: float
    metadata_weight: float
    bm25_weight: float = 0.10
    rrf_scale: float = 30.0
    min_score: float = 0.22
    fetch_k: int = DEFAULT_FETCH_K
    mmr_fetch_k: int = DEFAULT_MMR_FETCH_K
    bm25_fetch_k: int = DEFAULT_BM25_FETCH_K


@dataclass(frozen=True)
class Metrics:
    recall_at_k: float
    mrr: float
    ndcg_at_k: float


def load_examples(qa_directory: Path, limit: int | None = None) -> list[Example]:
    if not qa_directory.exists():
        raise FileNotFoundError(f"QA directory does not exist: {qa_directory}")

    examples: list[Example] = []
    for json_file in sorted(qa_directory.glob("qa_*.json")):
        rows = json.loads(json_file.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"Expected a JSON array in {json_file}")

        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue

            question = str(row.get("question", "")).strip()
            sheet = str(row.get("sheet", "")).strip()
            if not question or not sheet:
                continue

            examples.append(
                Example(
                    query=question,
                    relevant_key=f"{json_file.name}:{index}:{sheet}",
                    sheet=sheet,
                    source_file=json_file.name,
                    record_index=index,
                )
            )

    if limit is not None:
        return examples[:limit]
    return examples


def build_page_content(question: str, answer: str) -> str:
    answer_text = answer if answer else "No answer provided in source data."
    return f"Question: {question}\nAnswer: {answer_text}"


def load_corpus_documents(qa_directory: Path) -> list[Document]:
    documents: list[Document] = []
    if not qa_directory.exists():
        return documents

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


def iter_progress(items: list[Any], *, desc: str, leave: bool = False) -> Any:
    if tqdm is None:
        print(f"{desc}: {len(items)} item(s)")
        return items
    return tqdm(items, desc=desc, leave=leave)


def build_vector_store() -> Chroma:
    embeddings = OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
        validate_model_on_init=True,
    )
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIRECTORY,
    )


def check_ollama_available() -> None:
    base_url = (OLLAMA_BASE_URL or "http://localhost:11434").rstrip("/")
    tags_url = f"{base_url}/api/tags"
    request = urllib.request.Request(tags_url, method="GET")

    try:
        with urllib.request.urlopen(request, timeout=OLLAMA_REQUEST_TIMEOUT):
            return
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            "Ollama is not reachable for embedding queries. "
            f"Checked {tags_url}. Start Ollama or set OLLAMA_BASE_URL, then rerun."
        ) from exc


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

    query_tokens = tokenize_for_rerank(query)
    question_tokens = tokenize_for_rerank(question_text)
    if not query_tokens or not question_tokens:
        return 0.0

    overlap = query_tokens & question_tokens
    if not overlap:
        return 0.0

    coverage = len(overlap) / len(query_tokens)
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


def safe_lexical_candidates(query: str, corpus: list[Document], top_k: int) -> list[Any]:
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


def safe_bm25_candidates(query: str, corpus: list[Document], top_k: int) -> list[Any]:
    if not corpus:
        return []

    try:
        from langchain_community.retrievers import BM25Retriever
    except ImportError:
        return []

    retriever = BM25Retriever.from_documents(corpus)
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


def rerank_documents(
    query: str,
    dense_candidates: list[tuple[Any, float]],
    mmr_candidates: list[Any],
    config: RerankConfig,
    corpus: list[Document],
) -> list[tuple[float, Any]]:
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

    lexical_candidates = safe_lexical_candidates(query, corpus, top_k=max(24, config.fetch_k * 2))
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

    bm25_candidates = safe_bm25_candidates(query, corpus, top_k=max(config.bm25_fetch_k, config.fetch_k * 2))
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
    total_weight = max(
        1e-9,
        config.dense_weight
        + config.lexical_weight
        + config.rrf_weight
        + config.metadata_weight
        + config.bm25_weight,
    )

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
        rrf_score = min(1.0, rrf * config.rrf_scale)

        bm25_rank = bucket.get("bm25_rank")
        bm25_score = 0.0
        if bm25_rank is not None:
            bm25_score = 1.0 / (1.0 + float(bm25_rank))

        final_score = (
            (dense_score * config.dense_weight)
            + (lexical_score * config.lexical_weight)
            + (rrf_score * config.rrf_weight)
            + (meta_score * config.metadata_weight)
            + (bm25_score * config.bm25_weight)
        ) / total_weight
        scored.append((final_score, document))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def rank_example(
    store: Chroma,
    example: Example,
    config: RerankConfig,
    top_k: int,
    corpus: list[Document],
    use_mmr: bool = True,
) -> list[tuple[float, Any]]:
    dense_candidates = safe_dense_candidates(store, example.query, fetch_k=config.fetch_k)
    mmr_candidates = []
    if use_mmr and config.mmr_fetch_k > 0:
        mmr_candidates = safe_mmr_candidates(
            store,
            example.query,
            top_k=max(top_k * 2, top_k),
            fetch_k=config.mmr_fetch_k,
        )
    return rerank_documents(example.query, dense_candidates, mmr_candidates, config, corpus)


def metric_from_rank(rank: int | None, k: int) -> tuple[float, float, float]:
    if rank is None:
        return 0.0, 0.0, 0.0
    recall = 1.0 if rank <= k else 0.0
    reciprocal_rank = 1.0 / rank
    ndcg = 1.0 / math.log2(rank + 1) if rank <= k else 0.0
    return recall, reciprocal_rank, ndcg


def evaluate_config(
    store: Chroma,
    examples: list[Example],
    config: RerankConfig,
    top_k: int,
    corpus: list[Document],
    use_mmr: bool = True,
) -> Metrics:
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []

    for example in iter_progress(examples, desc="Evaluating queries"):
        ranked = rank_example(store, example, config, top_k=top_k, corpus=corpus, use_mmr=use_mmr)
        relevant_rank = None
        for rank, (_, document) in enumerate(ranked, start=1):
            if document_key(document) == example.relevant_key:
                relevant_rank = rank
                break

        recall, reciprocal_rank, ndcg = metric_from_rank(relevant_rank, top_k)
        recalls.append(recall)
        reciprocal_ranks.append(reciprocal_rank)
        ndcgs.append(ndcg)

    total = max(1, len(examples))
    return Metrics(
        recall_at_k=sum(recalls) / total,
        mrr=sum(reciprocal_ranks) / total,
        ndcg_at_k=sum(ndcgs) / total,
    )


def iter_weight_candidates(step: float = 0.1) -> list[RerankConfig]:
    candidates: list[RerankConfig] = []
    values = [round(i * step, 2) for i in range(1, int(1 / step))]

    for dense_weight in values:
        for lexical_weight in values:
            for rrf_weight in values:
                metadata_weight = round(1.0 - dense_weight - lexical_weight - rrf_weight, 2)
                if metadata_weight <= 0:
                    continue
                if metadata_weight not in values:
                    continue
                candidates.append(
                    RerankConfig(
                        dense_weight=dense_weight,
                        lexical_weight=lexical_weight,
                        rrf_weight=rrf_weight,
                        metadata_weight=metadata_weight,
                    )
                )

    return candidates


def evaluate_baselines(store: Chroma, examples: list[Example], top_k: int, corpus: list[Document]) -> dict[str, Metrics]:
    dense_metrics = evaluate_config(
        store,
        examples,
        RerankConfig(
            dense_weight=1.0,
            lexical_weight=0.0,
            rrf_weight=0.0,
            metadata_weight=0.0,
            bm25_weight=0.0,
            fetch_k=top_k,
            mmr_fetch_k=top_k,
            bm25_fetch_k=top_k,
        ),
        top_k=top_k,
        corpus=corpus,
    )

    current_metrics = evaluate_config(
        store,
        examples,
        RerankConfig(
            dense_weight=0.45,
            lexical_weight=0.30,
            rrf_weight=0.15,
            metadata_weight=0.10,
            bm25_weight=0.10,
        ),
        top_k=top_k,
        corpus=corpus,
    )

    lexical_metrics = evaluate_config(
        store,
        examples,
        RerankConfig(
            dense_weight=0.0,
            lexical_weight=1.0,
            rrf_weight=0.0,
            metadata_weight=0.0,
            bm25_weight=0.0,
            fetch_k=top_k,
            mmr_fetch_k=top_k,
            bm25_fetch_k=top_k,
        ),
        top_k=top_k,
        corpus=corpus,
    )

    return {
        "dense": dense_metrics,
        "current_hybrid": current_metrics,
        "lexical": lexical_metrics,
    }


def grid_search_best_config(
    store: Chroma,
    examples: list[Example],
    top_k: int,
    corpus: list[Document],
    max_trials: int | None = None,
) -> tuple[RerankConfig, Metrics]:
    best_config: RerankConfig | None = None
    best_metrics: Metrics | None = None

    for trial_index, config in enumerate(iter_progress(iter_weight_candidates(), desc="Grid search weights", leave=True), start=1):
        if max_trials is not None and trial_index > max_trials:
            break

        metrics = evaluate_config(store, examples, config, top_k=top_k, corpus=corpus)
        if best_metrics is None:
            best_config = config
            best_metrics = metrics
            continue

        if (metrics.mrr, metrics.ndcg_at_k, metrics.recall_at_k) > (
            best_metrics.mrr,
            best_metrics.ndcg_at_k,
            best_metrics.recall_at_k,
        ):
            best_config = config
            best_metrics = metrics

    if best_config is None or best_metrics is None:
        raise RuntimeError("No weight candidates were evaluated.")

    return best_config, best_metrics


def format_metrics(label: str, metrics: Metrics) -> str:
    return (
        f"{label:<18}"
        f" Recall@k={metrics.recall_at_k:.3f}"
        f" MRR={metrics.mrr:.3f}"
        f" nDCG@k={metrics.ndcg_at_k:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and tune RAG retrieval reranking.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of QA examples evaluated.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="How many final documents to consider relevant for metrics.")
    parser.add_argument("--tune", action="store_true", help="Run a grid search over rerank weights.")
    parser.add_argument("--max-trials", type=int, default=None, help="Limit grid search trials for a quick pass.")
    parser.add_argument("--output", type=Path, default=None, help="Write the results to a JSON file.")
    args = parser.parse_args()

    examples = load_examples(QA_DIRECTORY, limit=args.limit)
    if not examples:
        raise SystemExit(f"No evaluation examples found in {QA_DIRECTORY}")

    print(f"Loaded {len(examples)} QA example(s)")
    check_ollama_available()
    print("Ollama embedding service reachable")

    print("Opening Chroma vector store...")
    store = build_vector_store()
    print("Loading local QA corpus for lexical reranking...")
    corpus = load_corpus_documents(QA_DIRECTORY)
    print(f"Loaded {len(corpus)} corpus document(s)")

    baselines = evaluate_baselines(store, examples, top_k=args.top_k, corpus=corpus)
    print(f"Examples: {len(examples)}")
    for label, metrics in baselines.items():
        print(format_metrics(label, metrics))

    report: dict[str, Any] = {
        "examples": len(examples),
        "top_k": args.top_k,
        "baselines": {name: asdict(metrics) for name, metrics in baselines.items()},
    }

    if args.tune:
        best_config, best_metrics = grid_search_best_config(
            store,
            examples,
            top_k=args.top_k,
            corpus=corpus,
            max_trials=args.max_trials,
        )
        print(format_metrics("best_grid", best_metrics))
        print(
            "Best weights: "
            f"dense={best_config.dense_weight:.2f}, "
            f"lexical={best_config.lexical_weight:.2f}, "
            f"rrf={best_config.rrf_weight:.2f}, "
            f"metadata={best_config.metadata_weight:.2f}"
        )
        report["best_grid"] = {
            "config": asdict(best_config),
            "metrics": asdict(best_metrics),
        }

    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
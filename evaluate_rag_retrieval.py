from __future__ import annotations

import argparse
import json
import os
import random
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from ragas import EvaluationDataset, evaluate
from ragas.llms import llm_factory
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"Importing .* from 'ragas\.metrics' is deprecated.*",
)
from ragas.metrics import (
    FactualCorrectness,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

from app import (
    CHAT_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    invoke_graph,
    message_to_text,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is unavailable
    tqdm = None

load_dotenv()

QA_DIRECTORY = Path(os.getenv("QA_DIRECTORY", "sheets_qa"))

SUPPORTED_METRICS = (
    "context_precision",
    "context_recall",
    "faithfulness",
    "factual_correctness",
)


@dataclass(frozen=True)
class Example:
    query: str
    reference: str
    sheet: str
    source_file: str
    record_index: int


def iter_progress(items: list[Any], *, desc: str) -> Any:
    if tqdm is None:
        print(f"{desc}: {len(items)} item(s)")
        return items
    return tqdm(items, desc=desc)


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
            answer = str(row.get("answer", "")).strip()
            sheet = str(row.get("sheet", "")).strip()
            if not question or not answer or not sheet:
                continue

            examples.append(
                Example(
                    query=question,
                    reference=answer,
                    sheet=sheet,
                    source_file=json_file.name,
                    record_index=index,
                )
            )

    if limit is not None:
        return examples[:limit]
    return examples


def build_ragas_llm(provider: str, model: str | None = None) -> Any:
    normalized_provider = provider.lower().strip()

    if normalized_provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENROUTER_API_KEY or OPENAI_API_KEY for Ragas evaluation.")

        client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
        llm_name = model or OPENROUTER_MODEL
        return llm_factory(llm_name, provider="openai", client=client)

    if normalized_provider == "ollama":
        base_url = (OLLAMA_BASE_URL or "http://localhost:11434").rstrip("/")
        client = OpenAI(api_key=os.getenv("OLLAMA_API_KEY", "ollama"), base_url=f"{base_url}/v1")
        llm_name = model or OLLAMA_CHAT_MODEL
        return llm_factory(llm_name, provider="openai", client=client)

    raise ValueError(f"Unsupported eval provider: {provider}")


def parse_metrics(raw_metrics: str) -> list[str]:
    parsed = [item.strip().lower() for item in raw_metrics.split(",") if item.strip()]
    if not parsed:
        raise ValueError("At least one metric must be specified.")

    invalid = [metric for metric in parsed if metric not in SUPPORTED_METRICS]
    if invalid:
        raise ValueError(
            "Unsupported metric(s): "
            f"{', '.join(invalid)}. Supported: {', '.join(SUPPORTED_METRICS)}"
        )
    return parsed


def build_metrics(metric_names: list[str], evaluator_llm: Any) -> tuple[list[Any], dict[str, str]]:
    metric_factories: dict[str, Any] = {
        "context_precision": lambda llm: LLMContextPrecisionWithReference(llm=llm),
        "context_recall": lambda llm: LLMContextRecall(llm=llm),
        "faithfulness": lambda llm: Faithfulness(llm=llm),
        "factual_correctness": lambda llm: FactualCorrectness(llm=llm),
    }

    metrics: list[Any] = []
    column_map: dict[str, str] = {}
    for name in metric_names:
        metric = metric_factories[name](evaluator_llm)
        metrics.append(metric)
        column_map[name] = metric.name

    return metrics, column_map


def run_chatbot_examples(examples: list[Example], provider: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []

    for index, example in enumerate(iter_progress(examples, desc="Running chatbot over eval set"), start=1):
        thread_id = f"ragas-eval-{index}-{uuid.uuid4().hex[:8]}"
        state = invoke_graph(example.query, thread_id=thread_id, provider=provider)

        messages = state.get("messages", [])
        response_text = message_to_text(messages[-1]) if messages else ""
        retrieved_contexts = [str(entry) for entry in state.get("context", []) if str(entry).strip()]

        if not response_text:
            response_text = "I do not have enough information in the provided banking data to answer that."

        samples.append(
            {
                "user_input": example.query,
                "response": response_text,
                "reference": example.reference,
                "retrieved_contexts": retrieved_contexts,
            }
        )

    return samples


def summarize_result(result: Any, metric_names: list[str], column_map: dict[str, str]) -> dict[str, float | None]:
    df = result.to_pandas()
    summary: dict[str, float | None] = {}
    for metric in metric_names:
        metric_column = column_map.get(metric, metric)
        if metric_column not in df.columns:
            summary[metric] = None
            continue

        series = df[metric_column].dropna()
        summary[metric] = float(series.mean()) if len(series) else None
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate chatbot responses with Ragas metrics.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of QA examples evaluated.")
    parser.add_argument(
        "--provider",
        type=str,
        default=CHAT_PROVIDER,
        choices=["openrouter", "ollama"],
        help="LLM provider for chatbot generation and evaluation LLM.",
    )
    parser.add_argument(
        "--eval-model",
        type=str,
        default=None,
        help="Override eval-LLM model name used by Ragas (defaults to active chat model).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=",".join(SUPPORTED_METRICS),
        help=(
            "Comma-separated metric names. "
            f"Supported: {', '.join(SUPPORTED_METRICS)}"
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling/shuffling.")
    parser.add_argument("--output", type=Path, default=None, help="Write evaluation report to a JSON file.")
    args = parser.parse_args()

    metric_names = parse_metrics(args.metrics)

    examples = load_examples(QA_DIRECTORY)
    if not examples:
        raise SystemExit(f"No QA examples with answers found in {QA_DIRECTORY}")

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    if args.limit is not None:
        examples = examples[: args.limit]

    print(f"Loaded {len(examples)} example(s) for chatbot evaluation")
    samples = run_chatbot_examples(examples, provider=args.provider)
    dataset = EvaluationDataset.from_list(samples)

    evaluator_llm = build_ragas_llm(args.provider, model=args.eval_model)
    metrics, column_map = build_metrics(metric_names, evaluator_llm)

    print(f"Running Ragas evaluate() with metrics: {', '.join(metric_names)}")
    result = evaluate(dataset=dataset, metrics=metrics, raise_exceptions=False)
    summary = summarize_result(result, metric_names, column_map)

    print("Ragas metric summary:")
    for name in metric_names:
        value = summary.get(name)
        value_text = f"{value:.4f}" if isinstance(value, float) else "n/a"
        print(f"  {name}: {value_text}")

    report = {
        "examples": len(samples),
        "provider": args.provider,
        "eval_model": args.eval_model or (OPENROUTER_MODEL if args.provider == "openrouter" else OLLAMA_CHAT_MODEL),
        "metrics": summary,
    }

    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
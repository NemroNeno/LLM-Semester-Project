from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from guardrails import Guard
from guardrails.validator_base import FailResult, PassResult, Validator, register_validator

from settings import (
    GUARDRAILS_ENABLED,
    GUARDRAILS_INPUT_BLOCK_MESSAGE,
    GUARDRAILS_LOG_FAILURES,
    GUARDRAILS_MIN_GROUNDED_RATIO,
    GUARDRAILS_OUTPUT_BLOCK_MESSAGE,
)


@dataclass
class GuardrailsDecision:
    blocked: bool
    message: str
    reason: str = ""


_TOXIC_PATTERNS = (
    r"\b(hate|idiot|stupid|dumb|moron|abuse|violent|kill)\b",
    r"\b(fraud|scam someone|cheat someone)\b",
    r"\b(fuck|fucking|fucker|shit|bitch|bastard|asshole|haramzada|lanat)\b",
)

_PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\d[\s-]?){10,14}\b")
_EMAIL_PATTERN = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
_CNIC_PATTERN = re.compile(r"\b\d{5}-\d{7}-\d\b")
_CARD_PATTERN = re.compile(r"\b(?:\d[ -]?){13,19}\b")

_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "is",
    "are",
    "for",
    "of",
    "in",
    "on",
    "with",
    "from",
    "at",
    "this",
    "that",
    "be",
    "as",
    "it",
    "by",
    "if",
    "you",
    "your",
    "we",
    "our",
    "i",
    "me",
    "my",
}

warnings.filterwarnings(
    "ignore",
    message=r"Could not obtain an event loop\. Falling back to synchronous validation\.",
    category=UserWarning,
)


def _log_guardrails(message: str) -> None:
    if GUARDRAILS_LOG_FAILURES:
        print(f"[guardrails] {message}")


def _extract_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9%./-]+", text.lower())
        if len(token) > 2 and token not in _STOPWORDS
    }


@register_validator(name="toxic_language", data_type="string")
class ToxicLanguageValidator(Validator):
    def validate(self, value: Any, metadata: dict[str, Any]) -> Any:
        text = str(value or "").strip().lower()
        if any(re.search(pattern, text) for pattern in _TOXIC_PATTERNS):
            return FailResult(errorMessage="Detected unsafe or toxic language.")
        return PassResult()


@register_validator(name="private_data", data_type="string")
class PrivateDataValidator(Validator):
    def validate(self, value: Any, metadata: dict[str, Any]) -> Any:
        text = str(value or "")
        if _EMAIL_PATTERN.search(text):
            return FailResult(errorMessage="Detected an email address in content.")
        if _CNIC_PATTERN.search(text):
            return FailResult(errorMessage="Detected a CNIC-like identifier in content.")
        if _PHONE_PATTERN.search(text):
            return FailResult(errorMessage="Detected a phone number in content.")
        if _CARD_PATTERN.search(text):
            return FailResult(errorMessage="Detected card-like numeric content.")
        return PassResult()


@register_validator(name="anti_hallucination", data_type="string")
class AntiHallucinationValidator(Validator):
    def __init__(self, min_grounded_ratio: float = GUARDRAILS_MIN_GROUNDED_RATIO, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.min_grounded_ratio = max(0.0, min(1.0, float(min_grounded_ratio)))

    def validate(self, value: Any, metadata: dict[str, Any]) -> Any:
        answer_text = str(value or "")
        context_text = str(metadata.get("grounding_context", ""))
        if not context_text.strip():
            return PassResult()

        answer_tokens = _extract_tokens(answer_text)
        context_tokens = _extract_tokens(context_text)
        if not answer_tokens or not context_tokens:
            return PassResult()

        overlap = answer_tokens & context_tokens
        grounded_ratio = len(overlap) / max(1, len(answer_tokens))
        if grounded_ratio < self.min_grounded_ratio:
            return FailResult(
                errorMessage=(
                    "Generated response appears weakly grounded in retrieved banking context."
                )
            )

        answer_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", answer_text))
        context_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", context_text))
        if answer_numbers and not answer_numbers.issubset(context_numbers):
            return FailResult(
                errorMessage="Generated response contains numeric claims not present in retrieved context."
            )

        return PassResult()


@lru_cache(maxsize=1)
def _get_input_guard() -> Any | None:
    if not GUARDRAILS_ENABLED:
        return None

    return Guard.for_string(
        validators=[
            ToxicLanguageValidator(),
            PrivateDataValidator(),
        ],
        name="chat_input_guard",
        description="Validates incoming user content for toxicity and sensitive data.",
    )


@lru_cache(maxsize=1)
def _get_output_guard() -> Any | None:
    if not GUARDRAILS_ENABLED:
        return None

    return Guard.for_string(
        validators=[
            ToxicLanguageValidator(),
            PrivateDataValidator(),
            AntiHallucinationValidator(),
        ],
        name="chat_output_guard",
        description="Validates outgoing assistant content for toxicity, sensitive data, and grounding.",
    )


def validate_user_input(text: str) -> GuardrailsDecision:
    if not text.strip():
        return GuardrailsDecision(blocked=True, message=GUARDRAILS_INPUT_BLOCK_MESSAGE, reason="empty_input")

    if not GUARDRAILS_ENABLED:
        return GuardrailsDecision(blocked=False, message=text)

    guard = _get_input_guard()
    if guard is None:
        return GuardrailsDecision(blocked=False, message=text)

    try:
        outcome = guard.validate(text, metadata={})
    except Exception as exc:
        reason = str(exc) or "input_validation_failed"
        _log_guardrails(f"Input blocked: {reason}")
        return GuardrailsDecision(blocked=True, message=GUARDRAILS_INPUT_BLOCK_MESSAGE, reason=reason)

    if outcome.validation_passed:
        return GuardrailsDecision(blocked=False, message=text)

    reason = str(getattr(outcome, "error", "input_validation_failed") or "input_validation_failed")
    _log_guardrails(f"Input blocked: {reason}")
    return GuardrailsDecision(blocked=True, message=GUARDRAILS_INPUT_BLOCK_MESSAGE, reason=reason)


def validate_model_output(answer: str, *, user_question: str, rag_context: list[str], kg_context: list[str]) -> GuardrailsDecision:
    if not GUARDRAILS_ENABLED:
        return GuardrailsDecision(blocked=False, message=answer)

    guard = _get_output_guard()
    if guard is None:
        return GuardrailsDecision(blocked=False, message=answer)

    grounding_context = "\n\n".join([
        f"User Question:\n{user_question}",
        "RAG Context:\n" + ("\n\n".join(rag_context) if rag_context else "No RAG context."),
        "Knowledge Graph Facts:\n" + ("\n\n".join(kg_context) if kg_context else "No KG facts."),
    ])

    try:
        outcome = guard.validate(
            answer,
            metadata={"grounding_context": grounding_context},
        )
    except Exception as exc:
        reason = str(exc) or "output_validation_failed"
        _log_guardrails(f"Output blocked: {reason}")
        return GuardrailsDecision(blocked=True, message=GUARDRAILS_OUTPUT_BLOCK_MESSAGE, reason=reason)

    if outcome.validation_passed:
        return GuardrailsDecision(blocked=False, message=answer)

    reason = str(getattr(outcome, "error", "output_validation_failed") or "output_validation_failed")
    _log_guardrails(f"Output blocked: {reason}")
    return GuardrailsDecision(blocked=True, message=GUARDRAILS_OUTPUT_BLOCK_MESSAGE, reason=reason)

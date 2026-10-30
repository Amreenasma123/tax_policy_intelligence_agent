"""Typed state passed between LangGraph nodes."""
from typing import Any, TypedDict

class TaxGuardState(TypedDict, total=False):
    question: str
    country_code: str
    as_of_date: str
    topic: str | None
    top_k: int
    request_id: str
    redacted_question: str
    pii_counts: dict[str, int]
    safety_flags: list[str]
    request: Any
    plan: dict[str, Any]
    results: list[Any]
    response: dict[str, Any]
    error: str
    trace: list[str]

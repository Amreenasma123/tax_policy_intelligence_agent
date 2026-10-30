"""Best-effort PII redaction for the TaxGuard AI demo.

This is a lightweight regex-based first layer, not a certified DLP system.
It intentionally returns category/count metadata, never the matched values.
"""
from __future__ import annotations
import re
from dataclasses import dataclass

@dataclass(frozen=True)
class RedactionResult:
    text: str
    counts: dict[str, int]

# More specific identifiers run before generic phone detection to avoid masking
# their classification as phone numbers. Patterns favour precision over coverage.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)),
    ("us_ssn", re.compile(r"\b(?!000|666|9\d\d)\d{3}[- ]?(?!00)\d{2}[- ]?(?!0000)\d{4}\b")),
    ("india_pan", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.I)),
    ("india_aadhaar_candidate", re.compile(r"(?<!\d)(?:\d{4}[ -]?){2}\d{4}(?!\d)")),
    ("phone_candidate", re.compile(r"(?<!\w)(?:\+\d{1,3}[ .()-]?)?(?:\d[ .()-]?){9,12}\d(?!\w)")),
]

def redact_pii(text: str) -> RedactionResult:
    """Redact common PII patterns before retrieval; preserve a safe audit summary."""
    redacted = text
    counts: dict[str, int] = {}
    for category, pattern in _PATTERNS:
        redacted, n = pattern.subn(f"[REDACTED_{category.upper()}]", redacted)
        if n:
            counts[category] = counts.get(category, 0) + n
    return RedactionResult(text=redacted, counts=counts)

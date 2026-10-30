import re
from datetime import date
PATTERNS = [
    r"ignore (all |any )?(previous|prior|system|the )?(instructions|prompts)",
    r"disregard (all |any )?(previous|prior|system|the )?(instructions|prompts)",
    r"(reveal|show|print|display|expose) (me )?(the )?(system prompt|secrets|api key|confidential data|private data)",
    r"(show|expose|dump|leak) (me )?(all )?(confidential|sensitive|private) (data|information|records)",
    r"you are now the system",
    r"bypass (all )?(safety|security|guardrails|restrictions)",
    r"developer message|hidden instructions|system message",
]
def detect_injection(text):
    return ["prompt_injection_signal"] if any(re.search(p,text,re.I) for p in PATTERNS) else []
def validate_date(value):
    try: date.fromisoformat(value); return True
    except (ValueError,TypeError): return False
def assess_evidence(results,min_score=.055):
    if not results: return False,["no_eligible_evidence"]
    # RRF is a ranking score, not confidence. Use the separate relevance proxy for this demo gate.
    confidence = getattr(results[0], "relevance_score", results[0].score)
    flags=["low_retrieval_confidence"] if confidence < min_score else []
    return not flags,flags
def validate_citations(citations,results):
    allowed={r.chunk.chunk_id for r in results}
    return all(c.get("chunk_id") in allowed for c in citations)

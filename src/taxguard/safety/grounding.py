"""Conservative, explainable lexical evidence-grounding checks for the synthetic demo.

This is a heuristic screening layer, not a semantic entailment model. It deliberately
fails closed when a material answer sentence has weak overlap with its cited evidence.
"""
from __future__ import annotations
import re
from collections import Counter

_STOP = set('the a an and or but if then of to in on for from with by is are was were be been being this that these those it its as at may must should can could would will not no any each when where what which who how required requires policy document section'.split())

def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9]+", (text or "").lower()) if len(t) > 2 and t not in _STOP]

def split_claims(answer: str) -> list[str]:
    """Split an answer into sentence-like claims; ignore empty fragments."""
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", answer or "") if len(_tokens(x)) >= 3]

def validate_grounding(answer: str, citations: list, retrieved_chunks: list[dict], min_overlap: float = 0.30) -> dict:
    """Check each material sentence against the text of its cited, retrieved chunks.

    Overlap = distinct non-stopword claim tokens found in cited evidence / claim tokens.
    Returns per-claim scores and an overall pass/fail. Not a substitute for human review.
    """
    chunks = {c.get('chunk_id'): c for c in retrieved_chunks if isinstance(c, dict) and c.get('chunk_id')}
    claim_rows=[]; errors=[]
    citation_ids=[c.get('chunk_id') for c in citations if isinstance(c,dict)]
    for claim in split_claims(answer):
        terms=set(_tokens(claim))
        evidence_text=' '.join(str(chunks[cid].get('text','')) for cid in citation_ids if cid in chunks)
        evidence_terms=set(_tokens(evidence_text))
        overlap=len(terms & evidence_terms)/max(1,len(terms))
        supported=overlap >= min_overlap
        claim_rows.append({'claim':claim,'token_overlap':round(overlap,3),'supported_heuristically':supported})
        if not supported: errors.append('claim_low_evidence_overlap')
    if not claim_rows:
        errors.append('no_material_claims_detected')
    return {'grounded_heuristically':bool(claim_rows) and not errors,'claims_checked':len(claim_rows),
            'supported_claims':sum(r['supported_heuristically'] for r in claim_rows),
            'claim_results':claim_rows,'validation_errors':sorted(set(errors)),
            'method':'distinct non-stopword token overlap against cited retrieved chunks',
            'threshold':min_overlap,'limitation':'Heuristic only; lexical overlap does not prove entailment or detect every contradiction.'}

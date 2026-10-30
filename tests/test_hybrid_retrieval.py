import json
from pathlib import Path
from taxguard.ingestion.pipeline import load_documents
from taxguard.retrieval.search import PolicyRetriever

def test_hybrid_retriever_excludes_superseded_country_policies():
    chunks=load_documents("data/synthetic_policies")
    r=PolicyRetriever(chunks,use_dense=False)
    out=r.search("business travel itemized costs purpose", "CA", "2025-08-01", top_k=5)
    # The bundled CA corpus is explicitly marked superseded. Fail closed rather than
    # treating an old policy as current merely to make retrieval return results.
    assert out == []

def test_eval_set_has_50_cases_with_expected_fields():
    cases=[json.loads(x) for x in Path("data/eval/questions.jsonl").read_text().splitlines() if x.strip()]
    assert len(cases)==50
    assert all("expected_answer" in c and "expected_status" in c for c in cases)
    assert sum(c["case_type"]=="supported" for c in cases)==40
    assert sum(c["case_type"]=="no_eligible_evidence" for c in cases)==10

def test_search_stages_exposes_lexical_and_fused_lists():
    chunks=load_documents("data/synthetic_policies")
    r=PolicyRetriever(chunks,use_dense=False)
    stages=r.search_stages("business travel records and receipts", "US", "2025-08-01", top_k=3)
    assert stages["eligible_count"] > 0
    assert stages["tfidf"] and stages["bm25"] and stages["fused"]
    assert stages["dense_enabled"] is False
    assert stages["reranker_enabled"] is False

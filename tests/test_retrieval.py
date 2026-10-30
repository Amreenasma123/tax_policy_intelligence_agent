from taxguard.ingestion.pipeline import load_documents
from taxguard.retrieval.search import PolicyRetriever
def test_corpus_has_twenty_documents():
    assert len(set(c.document_id for c in load_documents("data/synthetic_policies")))==20
def test_country_isolation():
    results=PolicyRetriever(load_documents("data/synthetic_policies")).search("business travel records expenses","CA","2025-08-01")
    assert all(r.chunk.country_code=="CA" for r in results)
def test_future_policy_date_has_no_results():
    assert PolicyRetriever(load_documents("data/synthetic_policies")).search("business travel","CA","2030-01-01")==[]

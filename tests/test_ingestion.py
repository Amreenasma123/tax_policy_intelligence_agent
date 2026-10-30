from taxguard.ingestion.pipeline import load_documents, write_manifest
def test_manifest_detects_first_ingestion(tmp_path):
    chunks=load_documents("data/synthetic_policies")
    first=write_manifest(chunks,tmp_path/"manifest.json")
    assert len(first["changed"])==20
    second=write_manifest(chunks,tmp_path/"manifest.json")
    assert second["changed"]==[] and len(second["unchanged"])==20

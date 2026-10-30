# Validation Summary

Validated on 2026-10-21 in the available execution environment.

- Unit/integration tests: **14 passed** (`pytest -q`).
- Labeled evaluation cases: **50**.
- Expected decision statuses matched: **50/50 (100%)**.
- Document Recall@5: **100%**.
- Expected chunk Recall@5: **100%**.
- Expected-answer evidence coverage: **92.5%**.
- Deterministic planner route accuracy: **100%**.
- Abstention precision/recall/F1: **1.0 / 1.0 / 1.0**.
- Retrieval actually evaluated: **BM25 + TF-IDF + RRF**.
- Dense Chroma + MiniLM branch: implemented as optional, **not evaluated in this environment** because its optional dependencies/model weights were unavailable.
- LangGraph graph code: included and will be selected when `langgraph` is installed. This environment could not install LangGraph because package-network access was unavailable, so the graph-specific runtime branch was not executed here; the deterministic fallback was tested.
- LLM generation: no LLM called. Prompt template and optional provider adapter are included but are not wired into the default answer path.

Interpretation: these results apply only to the small, curated synthetic test set. They do not establish production quality, legal correctness, dense retrieval quality, or generated-answer accuracy.

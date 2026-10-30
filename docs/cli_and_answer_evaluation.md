# CLI and generated-answer evaluation

## Commands

- Ingest/chunk/manifest: `python scripts/taxguard.py ingest`
- Ingest plus optional Chroma/MiniLM embedding sync: `python scripts/taxguard.py ingest --dense`
- Inspect lexical/hybrid retrieval: `python scripts/taxguard.py retrieve --question "..." --country US --as-of 2025-08-01 --top-k 5`
- Retrieval/decision/abstention evaluation: `python scripts/taxguard.py evaluate`
- Dense retrieval evaluation (only when optional packages and model weights are available): `python scripts/taxguard.py evaluate --dense`
- LLM answer generation: `python scripts/taxguard.py generate --question "..." --country US --as-of 2025-08-01`
- LLM generated-answer checks: `python scripts/taxguard.py evaluate-generation`

## Current retrieval internals

The current lexical baseline uses scikit-learn TF-IDF (unigrams and bigrams) and an in-project BM25 implementation. RRF fuses their rankings. When enabled, Chroma + `all-MiniLM-L6-v2` contributes a dense semantic ranking to the same RRF. Country and effective-date filtering precedes final ranking. The current code exposes TF-IDF/BM25/overlap/RRF scores, but not raw dense cosine similarity, and has no cross-encoder reranker yet.

## What generated-answer evaluation currently means

The generation evaluation script calls the configured hosted endpoint for supported cases and checks that output parses as JSON, contains an answer, and cites chunk IDs present in the retrieved context. It also records expected-evidence coverage. These are useful automated smoke checks but are not a true accuracy or groundedness score. A human reviewer must verify each claim against its cited passage. For stronger evaluation, add labels for claim-level support, completeness, citation entailment, contradiction, correct abstention, privacy leakage and prompt injection. RAGAS/DeepEval or an LLM-as-judge can assist, but judge outputs should be calibrated against human annotations.

## LLM settings

No model is selected or called by default. The optional OpenAI-compatible provider reads `TAXGUARD_LLM_BASE_URL`, `TAXGUARD_LLM_API_KEY`, `TAXGUARD_LLM_MODEL`, `TAXGUARD_LLM_TEMPERATURE` (default 0.1), `TAXGUARD_LLM_MAX_TOKENS` (default 800) and `TAXGUARD_LLM_TIMEOUT_SECONDS` (default 25). Use a hosted `gpt-oss-20b` endpoint if the provider offers it and its data terms meet the assignment requirements. Do not say the project currently uses gpt-oss until the model is configured and the run report records its ID.

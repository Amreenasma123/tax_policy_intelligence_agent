# TaxGuard AI — Synthetic Tax Policy RAG Demo

See [`docs/END_TO_END_RUNBOOK.md`](docs/END_TO_END_RUNBOOK.md) for full Windows PowerShell setup, every major command, safety controls, evaluation interpretation, and troubleshooting.

Quick start:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/run_safety_evaluation.py
python scripts/taxguard.py ingest
python scripts/taxguard.py retrieve --question "What records are required for business travel expenses?" --country US --as-of 2025-08-01
```

To generate an LLM answer, configure `TAXGUARD_LLM_BASE_URL`, `TAXGUARD_LLM_MODEL`, and `TAXGUARD_LLM_API_KEY` in the current shell, then run the `generate` command documented in the runbook.

**Scope warning:** all bundled policy documents are synthetic. This is not legal/tax advice or a production-certified system. Grounding is a lexical-overlap heuristic and requires human review. See runbook for stale-policy status in the bundled non-US corpus.


## Streamlit UI and Redis history

See [`docs/UI_REDIS_AND_DEMO_HARDENING.md`](docs/UI_REDIS_AND_DEMO_HARDENING.md). The UI intentionally shows only a concise answer and minimal citations. Redis history is optional and scoped to the current Streamlit session.

Demo question sets: [`docs/DEMO_QUESTION_SET.md`](docs/DEMO_QUESTION_SET.md).

## Demo hardening notes

- Prompt-injection requests are blocked before retrieval/generation when matched by the input guard.
- The Streamlit UI shows a concise generated answer and minimal citation references only; retrieved chunks and internal diagnostics are not shown in the normal view.
- Redis is an optional session-history store, not vector memory or a cache. See the Redis hardening guide for limitations.

# TaxGuard AI — Implementation Notes

## What is actually implemented

| Feature | Current implementation |
|---|---|
| Corpus | 20 synthetic Markdown docs, 4 each for US, UK, India, Canada, Australia |
| Parsing | YAML frontmatter + Markdown body |
| Chunking | Heading-aware, character windows, 900-character maximum, 120-character overlap |
| Manifest | SHA-256 source-file hashes; changed/unchanged/removed source paths |
| Retrieval baseline | TF-IDF uni/bi-grams + BM25 + RRF rank fusion |
| Dense retrieval | Optional Chroma + `all-MiniLM-L6-v2`; only active with `TAXGUARD_USE_DENSE=true` and `requirements-ml.txt` installed |
| Citation | Chunk ID, document ID, version, section, effective dates and score metadata |
| Guardrails | Request validation, country/date filtering, PII redaction, injection signal detection, evidence threshold, citation-ID validation, abstention |
| Agents | Request Router Agent and Policy Research Agent as Python components |
| Planner | Deterministic rule-based planner, no LLM planning |
| Orchestration | LangGraph `StateGraph` when installed; deterministic fallback when not installed |
| LLM | None called by default. Optional OpenAI-compatible adapter exists but is not wired into default response generation |
| UI | Streamlit with answer/evidence, safety flags, plan, trace and retrieval mode |
| Evaluation | 50 labeled cases, report and per-case JSON |

## Chunking and ingestion

`src/taxguard/ingestion/pipeline.py` parses frontmatter, hashes each raw source document with SHA-256, splits on Markdown headings and windows long sections. With 900 chars and 120 overlap, adjacent full chunks advance about 780 chars. Metadata is attached to every chunk.

Why: smaller passages improve retrieval precision and citation granularity; overlap reduces context loss. Limits: character size is not token size; complex tables, footnotes and legal exception chains need structure-aware handling.

The manifest is not a vector database and is not full incremental vector ingestion. A future implementation should hash normalized chunks, embed only changed chunks, tombstone removed chunks, build a staged index, validate it, atomically promote it, and support rollback.

## Hybrid retrieval

`src/taxguard/retrieval/search.py` implements a lightweight dependency-free BM25 ranker and scikit-learn TF-IDF ranker. Both produce ranked candidate lists. RRF fuses their ranks. If enabled, Chroma/MiniLM contributes a third ranking. The display `score` is normalized RRF rank score; `relevance_score` is a separate heuristic used by the evidence gate. Neither is a calibrated probability.

Optional dense setup:

```bash
pip install -r requirements-ml.txt
# Linux/macOS
export TAXGUARD_USE_DENSE=true
# Windows PowerShell
$env:TAXGUARD_USE_DENSE="true"
python scripts/run_evaluation.py
```

The model weights for `sentence-transformers/all-MiniLM-L6-v2` download on first use. Chroma persistence is under `data/chroma/`. If offline or unable to download, keep the default lexical hybrid path and do not claim dense retrieval was measured.

## Agent state, nodes, edges and memory

- `src/taxguard/graph/state.py`: `TaxGuardState`, the shared per-request state schema.
- `src/taxguard/graph/builder.py`: StateGraph construction and conditional routing.
- `input_guard`: redact PII.
- `router_agent`: validate request and produce plan.
- `research_agent`: retrieve, assess evidence, validate citations and return excerpts.
- `finalize`: attach execution trace, plan and active retrieval mode.
- Normal edges connect fixed steps; conditional edges route invalid/injection-signaled requests to finalize without research.
- State is transient working data for one invocation. There is no persistent conversational memory/checkpointer. A checkpointer should only be added if the product needs resumable workflows or multi-turn context, and sensitive state requires retention controls.

## LLM and prompt

The default app returns evidence excerpts and does not call an LLM. The prompt in `prompts/tax_policy_answer.md` is a proposed grounded generation prompt. The adapter `src/taxguard/llm_provider.py` supports an OpenAI-compatible endpoint when explicitly configured with `TAXGUARD_LLM_BASE_URL`, `TAXGUARD_LLM_API_KEY`, and `TAXGUARD_LLM_MODEL`. Temperature defaults to 0.1 in the adapter, max tokens to 800, and timeout to 25 seconds. These settings apply only if someone explicitly calls the adapter; they are not current default-generation settings.

Do not assume hosted open-model inference is always free. Provider free tiers, model availability and data retention change; review the provider terms and never send sensitive real tax data without approval.

## Evaluation

`data/eval/questions.jsonl` contains 50 hand-labeled cases with expected status, expected answer text, expected document ID, expected chunk ID, expected planner route and case type. `python scripts/run_evaluation.py` writes:

- `reports/evaluation_report.md`: summary and case-level result table.
- `reports/evaluation_results.json`: detailed per-case retrieval, status, planner and citation metadata.

Metrics include decision accuracy, Recall@5, top-1 document accuracy, MRR, Precision@5, expected-chunk Recall@5/MRR, expected-answer evidence coverage, deterministic planner-route accuracy, abstention precision/recall/F1.

The 100% decision score is only a result on 50 synthetic, intentionally curated cases, including 10 straightforward no-eligible-date abstentions. It is not model accuracy, real-world accuracy or a statistical guarantee. Expected-answer evidence coverage checks whether the labeled supporting sentence appears in retrieved passages; it is not generated-answer factuality. Add harder paraphrases, in-date unsupported topics, stale/superseded conflicts, ambiguous questions, injection cases, false-positive PII cases and human-reviewed labels before making quality claims.

## Commands

```bash
pytest -q
python scripts/run_evaluation.py
streamlit run app.py
```

## Debug order

1. Confirm source file and frontmatter.
2. Inspect chunk IDs/text in ingestion output.
3. Check jurisdiction/date eligibility count.
4. Compare TF-IDF and BM25 rankings and RRF order.
5. Check expected chunk and score fields.
6. Inspect evidence threshold, abstention and citations.
7. Inspect graph `execution_trace`, `plan` and `retrieval_mode`.
8. Add a regression test before finalizing a fix.

# TaxGuard AI — VS Code LLM, Chroma and Agent Debugging

## 1. Configure a hosted LLM (PowerShell)

TaxGuard's `src/taxguard/llm_provider.py` is the integration point. It reads these environment variables:

- `TAXGUARD_LLM_BASE_URL`
- `TAXGUARD_LLM_API_KEY`
- `TAXGUARD_LLM_MODEL`
- optional: `TAXGUARD_LLM_TEMPERATURE` (default `0.1`)
- optional: `TAXGUARD_LLM_MAX_TOKENS` (default `800`)
- optional: `TAXGUARD_LLM_TIMEOUT_SECONDS` (default `45`)

A simple OpenAI-compatible hosted option to test is Groq, if the model is currently enabled for your account. Create a key at https://console.groq.com/keys and check model availability at https://console.groq.com/docs/models. Do not assume a free quota is guaranteed; rates and limits can change.

From the VS Code PowerShell terminal, inside the `taxguard-ai` root folder:

```powershell
$env:TAXGUARD_LLM_BASE_URL = "https://api.groq.com/openai/v1"
$env:TAXGUARD_LLM_MODEL = "openai/gpt-oss-20b"
$env:TAXGUARD_LLM_TEMPERATURE = "0.1"
$env:TAXGUARD_LLM_MAX_TOKENS = "800"

# Securely prompt for the key (input is not echoed to the terminal)
$secureKey = Read-Host "Paste your NEW Groq API key" -AsSecureString
$ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
try { $env:TAXGUARD_LLM_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr) }
finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr) }
```

These values apply only to the current terminal session. Do not commit API keys or paste them into source code. Do not send real tax documents or PII to a hosted provider unless its data-processing and retention terms have been approved.

Smoke test the LLM without RAG first. The script prints a useful provider error for invalid keys (401/403), wrong URLs/models (404), or quota/rate limits (429), without printing the key:

```powershell
python scripts/test_llm.py
```

**Important:** Do not use `Read-Host xxyyy...`; `Read-Host` takes a prompt, not the API key. If a real key was pasted into chat, source control, or a screenshot, revoke it and create a replacement. The environment variables last only for this terminal session.

After the smoke test says `PASS`, ask TaxGuard to retrieve eligible synthetic evidence and generate an answer:

```powershell
python scripts/taxguard.py generate --question "What records are required for business travel expenses?" --country US --as-of 2025-08-01
```

The LLM only answers if the evidence gate returns eligible citations. It uses the synthetic corpus; do not use real tax documents or personal data for this demo.

Then test generation with retrieved synthetic evidence:

```powershell
python scripts/taxguard.py generate --question "What records are required for business travel expenses?" --country US --as-of 2025-08-01
```

Then run the generated-answer checks:

```powershell
python scripts/taxguard.py evaluate-generation
```

The generation evaluator currently performs basic JSON and citation-ID checks. It is a heuristic, not a validated factual accuracy or groundedness metric. Review outputs manually before presenting any quality claim.

## 2. What code invokes the LLM?

- `src/taxguard/llm_provider.py`: sends the HTTP request to the configured OpenAI-compatible endpoint and loads `prompts/tax_policy_answer.md`.
- `prompts/tax_policy_answer.md`: grounding, jurisdiction, date, abstention and citation rules.
- `scripts/test_llm.py`: isolated LLM smoke test using only fictional sample evidence.
- `scripts/taxguard.py`: `generate` and `evaluate-generation` commands.

The default Streamlit workflow does not automatically invoke generation yet. The CLI is the explicit test path in this version.

## 3. Inspect ChromaDB

First ensure optional packages and model weights are available:

```powershell
pip install -r requirements-ml.txt
python scripts/taxguard.py ingest --dense
```

The default persistent directory is `data/chroma`, and the collection name is `taxguard_chunks`.

Inspect the stored count and sample chunks:

```powershell
python scripts/inspect_chroma.py
```

Run a semantic query against the stored collection:

```powershell
python scripts/inspect_chroma.py --query "travel receipts and business purpose" --limit 5
```

This is a read-only inspection script. It displays chunk IDs, metadata, text snippets and Chroma distances. Lower cosine distance usually means closer vectors, but it is not a probability. The project currently stores vectors, chunk text and metadata in Chroma; there is no separate GUI database browser bundled.

If you want a graphical browser, use a Chroma-compatible third-party viewer only after checking its maintenance and security. For the assignment, the CLI inspection is reproducible and avoids another dependency.

## 4. Understand state and debug LangGraph

State is the per-request working dictionary. Its TypedDict schema is in `src/taxguard/graph/state.py`. Typical fields include `question`, `country_code`, `as_of_date`, `redacted_question`, `pii_counts`, `safety_flags`, `request`, `plan`, `response` and `trace`.

The graph nodes and edges are in `src/taxguard/graph/builder.py`:

1. `input_guard`: redacts detected PII and updates counts/trace.
2. `router_agent`: validates the request, sets safety flags and creates a plan.
3. Conditional edge: routes invalid or injection-flagged requests to `finalize`; otherwise routes to `research_agent`.
4. `research_agent`: retrieves eligible evidence, checks the evidence gate and updates response.
5. `finalize`: attaches trace and plan and writes a safe audit summary.

Run a node-by-node trace:

```powershell
python scripts/debug_graph.py
```

Try a different question:

```powershell
python scripts/debug_graph.py --question "What records are required for business travel expenses?" --country US --as-of 2025-08-01
```

The script prints each node name and selected state updates rather than dumping the entire state or original question. If LangGraph is missing, it reports that the deterministic fallback is being used.

## 5. VS Code breakpoints

1. Open `src/taxguard/graph/builder.py`.
2. Click the gutter beside `input_guard`, `router_agent`, `research_agent` or `finalize` to set a breakpoint.
3. In VS Code, choose Run and Debug, create/select a Python launch configuration, and use `scripts/debug_graph.py` as the program.
4. Ensure the selected interpreter is the project `.venv` and the terminal working directory is the `taxguard-ai` project root.
5. Step over each node and inspect `state`, `update`, `request`, `plan`, `safety_flags` and `response` in the Variables panel.

State changes when a node returns a dictionary of updates. LangGraph merges those updates into the shared state and follows the next edge. A node should return only fields it intends to update; it does not need to reconstruct the whole state.

## 6. Important limitations to state honestly

- All included policy documents are fictional.
- Hosted LLM configuration is optional; no model is configured by default.
- Chroma/MiniLM is optional and requires local model files on first use.
- The current reranker is not implemented; RRF is fusion, not a cross-encoder reranker.
- Current generated-answer evaluation is only a structural heuristic. Human review and claim-level grounding evaluation are still needed.
- The graph uses two logical agent components with deterministic planning; it is not a pair of autonomous LLM agents.

# UI cleanup, Redis history and demo hardening

## What changed
- Removed the static example questions.
- Normal UI now displays only decision, concise answer, and minimal citations (document ID, version, section).
- Removed raw retrieved chunks, ranking scores, full citation manifest, planner trace, and audit payload from the normal user view.
- Added injection detection for requests to reveal confidential/private data and ignore prompts.
- Added a shared generation service used by Streamlit, so the UI now calls the configured LLM after the evidence gate.
- Added optional Redis-backed per-session history with 30-day default TTL and a 50-entry cap.
- Redis stores a PII-redacted question and answer, status, compact citations and safety flags.

## Start Redis locally (Docker Desktop required)
From the project root:
```powershell
docker compose -f docker-compose.redis.yml up -d
docker compose -f docker-compose.redis.yml ps
```

## Configure the app
```powershell
pip install -r requirements.txt
$env:TAXGUARD_REDIS_URL = "redis://localhost:6379/0"
$env:TAXGUARD_HISTORY_TTL_SECONDS = "2592000"
$env:TAXGUARD_LLM_BASE_URL = "https://api.groq.com/openai/v1"
$env:TAXGUARD_LLM_MODEL = "openai/gpt-oss-20b"
$env:TAXGUARD_LLM_API_KEY = "YOUR_REPLACEMENT_KEY"
$env:TAXGUARD_LLM_TEMPERATURE = "0.1"
$env:TAXGUARD_LLM_MAX_TOKENS = "800"
streamlit run app.py
```
Use a replacement API key; revoke any key previously pasted into chat or committed to files.

## Redis behavior
- The app creates a random session ID in Streamlit session state.
- History is isolated by that session ID.
- Redis persists the records and they survive app-process restarts, but this demo's session ID is held in Streamlit session state. A browser refresh/new browser session may create a new session ID, so cross-refresh/user identity is not guaranteed. Production needs authenticated user identity or a secure server-managed session cookie.
- Each session retains up to 50 records; default expiration is 30 days.
- Clear history removes only the current session's Redis list.
- If Redis is unavailable, the app continues without history and displays a connection notice.
- This is suitable for a local interviewer demo, not a multi-user production deployment. Production needs authentication, server-side user identity, authorization, encryption, managed Redis, audit retention policy, and tested session lifecycle.

## Important limitations / issues to address before claiming production readiness
1. Policy corpus is synthetic; it is not authoritative legal/tax data.
2. The lexical grounding heuristic is not a semantic entailment verifier.
3. Query rewriting, cross-encoder reranking, retrieval/LLM caching, and durable cross-device user identity are not implemented.
4. Dense retrieval is optional; configure and verify it explicitly.
5. Current policy metadata must be reviewed for effective dates and active/superseded statuses.
6. Redis history is not conversational memory. Prior turns are not automatically injected into later prompts.
7. Do not expose retrieved evidence, scores, internal traces, or audit details to ordinary users. Keep diagnostics behind an authenticated developer/admin mode in a real deployment.
8. The prompt-injection detector is pattern-based; it is a defense-in-depth signal, not a complete security boundary.
9. The generation path should be tested with a configured LLM. When no LLM is configured, the UI reports that generation is unavailable instead of presenting raw chunks as an answer.

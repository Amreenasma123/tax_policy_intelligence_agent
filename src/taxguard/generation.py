"""Shared concise answer-generation service used by CLI and Streamlit."""
from taxguard.llm_provider import OpenAICompatibleProvider
from taxguard.safety.response_validation import validate_llm_response
from taxguard.safety.grounding import validate_grounding


def generate_answer(question: str, country_code: str, as_of_date: str, workflow_result: dict, top_k: int = 5) -> dict:
    flags = list(workflow_result.get("safety_flags", []))
    if "prompt_injection_signal" in flags:
        return {
            "status": "HUMAN_REVIEW",
            "answer": "I can’t help with that request. It appears to contain instructions that conflict with the application’s safety rules.",
            "citations": [],
            "safety_flags": ["prompt_injection_signal"],
        }

    if workflow_result.get("status") != "answered" or not workflow_result.get("citations"):
        return {
            "status": str(workflow_result.get("status", "ABSTAINED")).upper(),
            "answer": workflow_result.get("answer") or "I could not find sufficient applicable policy evidence to answer this question.",
            "citations": [],
            "safety_flags": flags,
        }

    retrieved = [
        item["chunk"] for item in workflow_result.get("results", [])[:top_k]
        if isinstance(item, dict) and isinstance(item.get("chunk"), dict)
    ]
    if not retrieved:
        return {"status": "ABSTAINED", "answer": "I could not find sufficient applicable policy evidence to answer this question.", "citations": [], "safety_flags": ["no_retrieved_evidence"]}

    context = "\\n\\n".join(
        f"[chunk_id={c['chunk_id']} document_id={c['document_id']} version={c['version']} section={c['section']}]\\n{c['text']}"
        for c in retrieved
    )
    provider = OpenAICompatibleProvider()
    if not provider.configured:
        return {
            "status": "HUMAN_REVIEW",
            "answer": "Answer generation is unavailable because the LLM provider is not configured. Configure the provider and try again.",
            "citations": [],
            "safety_flags": ["llm_not_configured"],
        }

    safe_question = workflow_result.get("redacted_question") or question
    raw = provider.answer(safe_question, country_code, as_of_date, context)
    checked = validate_llm_response(raw, retrieved)
    if not checked["valid"]:
        return {
            "status": "HUMAN_REVIEW",
            "answer": "The generated response did not pass response validation and has been withheld.",
            "citations": [],
            "safety_flags": ["response_validation_failed"],
        }

    payload = checked["response"]
    if payload["status"] == "ANSWERED":
        grounding = validate_grounding(payload["answer"], payload["citations"], retrieved)
        if not grounding["grounded_heuristically"]:
            return {
                "status": "HUMAN_REVIEW",
                "answer": "The response could not be sufficiently matched to the retrieved policy evidence and has been withheld for review.",
                "citations": [],
                "safety_flags": ["grounding_check_failed"],
            }

    # Only return a compact, user-facing citation manifest. Never return retrieved chunk text.
    citations = []
    for c in payload.get("citations", []):
        citations.append({
            "document_id": c["document_id"],
            "version": c["version"],
            "section": c["section"],
        })
    return {
        "status": payload["status"],
        "answer": payload["answer"],
        "citations": citations,
        "safety_flags": sorted(set(flags + payload.get("safety_flags", []))),
    }

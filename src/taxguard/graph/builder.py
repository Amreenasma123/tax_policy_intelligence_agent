"""LangGraph orchestration for TaxGuard's two agents and safety/evidence nodes."""
from __future__ import annotations
from typing import Any
from taxguard.graph.state import TaxGuardState


def build_graph(workflow):
    """Build a real StateGraph when LangGraph is installed; otherwise return None.

    Nodes: input_guard -> router_agent -> research_agent -> finalize. Conditional
    routing exits early for invalid requests or detected prompt-injection signals.
    """
    try:
        from langgraph.graph import StateGraph, START, END
    except ImportError:
        return None

    def input_guard(state: TaxGuardState) -> dict[str, Any]:
        from taxguard.safety.pii import redact_pii
        result = redact_pii(state.get("question", ""))
        return {"redacted_question": result.text, "pii_counts": result.counts,
                "safety_flags": list(state.get("safety_flags", [])),
                "trace": list(state.get("trace", [])) + ["input_guard"]}

    def router_agent(state: TaxGuardState) -> dict[str, Any]:
        request, flags = workflow.router.route(state.get("redacted_question", ""), state.get("country_code", ""),
                                               state.get("as_of_date", ""), state.get("topic"))
        plan = workflow.planner.plan(state.get("redacted_question", ""), state.get("country_code", ""), state.get("as_of_date", ""))
        return {"request": request, "safety_flags": list(set(state.get("safety_flags", []) + flags)),
                "plan": {"steps": plan.steps, "route": plan.route, "rationale": plan.rationale},
                "trace": list(state.get("trace", [])) + ["router_agent"]}

    def research_agent(state: TaxGuardState) -> dict[str, Any]:
        request = state.get("request")
        if request is None:
            return {"response": {"status":"error", "answer":"Request validation failed.", "citations":[],
                    "safety_flags":state.get("safety_flags", []), "results":[]},
                    "trace": list(state.get("trace", [])) + ["research_agent_skipped"]}
        result = workflow.researcher.answer(request, int(state.get("top_k", 5)), workflow.audit_path)
        result["safety_flags"] = sorted(set(result.get("safety_flags", []) + state.get("safety_flags", [])))
        result["pii_redaction_applied"] = bool(state.get("pii_counts", {}))
        result["pii_redaction_counts"] = state.get("pii_counts", {})
        result["redacted_question"] = state.get("redacted_question", "")
        if "prompt_injection_signal" in result["safety_flags"]:
            result["status"] = "needs_review"
            result["answer"] = "Potential prompt-injection signal detected. Response withheld for review."
            result["citations"] = []
        return {"response": result, "trace": list(state.get("trace", [])) + ["research_agent"]}

    def finalize(state: TaxGuardState) -> dict[str, Any]:
        response = dict(state.get("response", {}))
        if not response:
            flags = state.get("safety_flags", [])
            if "prompt_injection_signal" in flags:
                response = {"status":"needs_review", "answer":"Potential prompt-injection signal detected. Response withheld for review.", "citations":[], "results":[], "safety_flags":flags}
            else:
                response = {"status":"error", "answer":"Request could not be processed.", "citations":[], "results":[], "safety_flags":flags}
            response.setdefault("country_code", state.get("country_code")); response.setdefault("as_of_date", state.get("as_of_date"))
            response.setdefault("request_id", __import__("taxguard.observability.audit", fromlist=["new_request_id"]).new_request_id())
            response.setdefault("limitations", ["Synthetic demo documents only; not legal or tax advice."])
            response.setdefault("pii_redaction_applied", bool(state.get("pii_counts", {})))
            response.setdefault("pii_redaction_counts", state.get("pii_counts", {}))
            response.setdefault("redacted_question", state.get("redacted_question", ""))
        response["execution_trace"] = list(state.get("trace", [])) + ["finalize"]
        response["plan"] = state.get("plan", {})
        response["retrieval_mode"] = workflow.retriever.mode
        # Safe audit summary only: never include the original question or matched identifier values.
        from taxguard.observability.audit import write_audit
        write_audit({"request_id": response.get("request_id"), "event":"pii_redaction_summary",
                     "pii_redaction_applied":bool(state.get("pii_counts", {})),
                     "pii_redaction_counts":state.get("pii_counts", {}),
                     "execution_trace":response["execution_trace"],
                     "retrieval_mode":workflow.retriever.mode}, workflow.audit_path)
        return {"response": response, "trace": list(state.get("trace", [])) + ["finalize"]}

    def after_router(state: TaxGuardState) -> str:
        flags = state.get("safety_flags", [])
        if state.get("request") is None or "prompt_injection_signal" in flags:
            return "finalize"
        return "research_agent"

    graph = StateGraph(TaxGuardState)
    graph.add_node("input_guard", input_guard)
    graph.add_node("router_agent", router_agent)
    graph.add_node("research_agent", research_agent)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "input_guard")
    graph.add_edge("input_guard", "router_agent")
    graph.add_conditional_edges("router_agent", after_router, {"research_agent":"research_agent", "finalize":"finalize"})
    graph.add_edge("research_agent", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()

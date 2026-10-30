from taxguard.ingestion.pipeline import load_documents, write_manifest
from taxguard.retrieval.search import PolicyRetriever
from taxguard.agents.router import RequestRouterAgent
from taxguard.agents.research import PolicyResearchAgent
from taxguard.agents.planner import TaxPolicyPlannerAgent
from taxguard.safety.pii import redact_pii
from taxguard.observability.audit import write_audit, new_request_id
from taxguard.graph.builder import build_graph

class TaxGuardWorkflow:
    def __init__(self, corpus_path="data/synthetic_policies", manifest_path="data/index/manifest.json",
                 audit_path="logs/audit.jsonl", use_dense=None, chroma_path="data/chroma"):
        self.chunks = load_documents(corpus_path)
        self.ingestion_delta = write_manifest(self.chunks, manifest_path)
        self.retriever = PolicyRetriever(self.chunks, use_dense=use_dense, chroma_path=chroma_path)
        self.router = RequestRouterAgent()
        self.researcher = PolicyResearchAgent(self.retriever)
        self.planner = TaxPolicyPlannerAgent()
        self.audit_path = audit_path
        self.graph = build_graph(self)

    def ask(self, question, country_code, as_of_date, topic=None, top_k=5):
        initial = {"question": question if isinstance(question, str) else "", "country_code": country_code,
                   "as_of_date": as_of_date, "topic": topic, "top_k": top_k, "trace": [], "safety_flags": []}
        if self.graph is not None:
            try:
                state = self.graph.invoke(initial)
                result = state.get("response", {})
                # Graph conditional path may terminate before the research node.
                if result.get("status") == "error":
                    result.setdefault("request_id", new_request_id())
                    result.setdefault("country_code", country_code); result.setdefault("as_of_date", as_of_date)
                    result.setdefault("limitations", ["Synthetic demo documents only; not legal or tax advice."])
                    result.setdefault("safety_flags", state.get("safety_flags", []))
                    result.setdefault("pii_redaction_applied", bool(state.get("pii_counts", {})))
                    result.setdefault("pii_redaction_counts", state.get("pii_counts", {}))
                    result.setdefault("redacted_question", state.get("redacted_question", ""))
                    write_audit({"request_id":result["request_id"],"status":"error","country_code":country_code,
                        "as_of_date":as_of_date,"result_count":0,"safety_flags":result["safety_flags"],
                        "pii_redaction_counts":result["pii_redaction_counts"],"citation_count":0},self.audit_path)
                return result
            except Exception as exc:
                # A graph/runtime failure must be visible but should not silently call a different path.
                return {"status":"error","answer":"Workflow execution failed. Inspect application logs.","citations":[],
                        "results":[],"safety_flags":["workflow_execution_error"],"error_type":type(exc).__name__,
                        "limitations":["Synthetic demo only."]}
        return self._ask_fallback(question, country_code, as_of_date, topic, top_k)

    def _ask_fallback(self, question, country_code, as_of_date, topic, top_k):
        pii_result = redact_pii(question if isinstance(question, str) else "")
        safe_question = pii_result.text
        request, route_flags = self.router.route(safe_question, country_code, as_of_date, topic)
        plan = self.planner.plan(safe_question, country_code, as_of_date)
        if request is None or "prompt_injection_signal" in route_flags:
            rid = new_request_id(); status = "error" if request is None else "needs_review"
            answer = "Request invalid. Check the question and applicable date." if request is None else "Potential prompt-injection signal detected. Response withheld for review."
            result = {"status":status,"answer":answer,"country_code":country_code,"as_of_date":as_of_date,
                "citations":[],"limitations":["Synthetic demo documents only; not legal or tax advice."],"safety_flags":route_flags,
                "request_id":rid,"results":[],"pii_redaction_applied":bool(pii_result.counts),"pii_redaction_counts":pii_result.counts,
                "redacted_question":safe_question,"execution_trace":["input_guard","router_agent","finalize"],
                "plan":{"steps":plan.steps,"route":plan.route,"rationale":plan.rationale},"retrieval_mode":self.retriever.mode}
            write_audit({"request_id":rid,"status":status,"country_code":country_code,"as_of_date":as_of_date,
                "result_count":0,"safety_flags":route_flags,"pii_redaction_counts":pii_result.counts,"citation_count":0},self.audit_path)
            return result
        result = self.researcher.answer(request, top_k, self.audit_path)
        result["safety_flags"] = sorted(set(result.get("safety_flags", []) + route_flags))
        result["pii_redaction_applied"] = bool(pii_result.counts); result["pii_redaction_counts"] = pii_result.counts
        result["redacted_question"] = safe_question
        result["execution_trace"] = ["input_guard","router_agent","research_agent","finalize"]
        result["plan"] = {"steps":plan.steps,"route":plan.route,"rationale":plan.rationale}
        result["retrieval_mode"] = self.retriever.mode
        write_audit({"request_id":result["request_id"],"event":"pii_redaction_summary",
            "pii_redaction_applied":bool(pii_result.counts),"pii_redaction_counts":pii_result.counts},self.audit_path)
        return result

"""Deterministic planner for bounded, auditable tax-policy tasks.

This deliberately avoids asking an LLM to plan a simple search. It emits an explicit
plan that LangGraph nodes execute and that can be evaluated with route/path tests.
"""
from dataclasses import dataclass

@dataclass
class Plan:
    steps: list[str]
    route: str
    rationale: str

class TaxPolicyPlannerAgent:
    def plan(self, question: str, country_code: str, as_of_date: str) -> Plan:
        q = question.lower()
        if any(x in q for x in ["compare countries", "compare jurisdictions", "across countries", "compare us and"]):
            return Plan(["validate", "retrieve_by_jurisdiction", "compare_evidence", "review_citations"], "multi_jurisdiction_review", "Cross-jurisdiction comparison requested")
        if any(x in q for x in ["exact tax rate", "exact penalty", "penalty amount", "legal advice"]):
            return Plan(["validate", "retrieve_evidence", "check_evidence", "human_review_if_unsupported"], "high_impact_review", "Question asks for a high-impact specific conclusion")
        return Plan(["validate", "retrieve_evidence", "assess_evidence", "validate_citations"], "policy_lookup", "Standard single-jurisdiction policy lookup")

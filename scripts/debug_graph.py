"""Run the LangGraph one node at a time and print safe, selected state updates."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from taxguard.workflow import TaxGuardWorkflow


def main():
    p = argparse.ArgumentParser(description="Trace TaxGuard LangGraph nodes and state updates")
    p.add_argument("--question", default="What records are required for business travel expenses?")
    p.add_argument("--country", default="US", choices=["US", "UK", "IN", "CA", "AU"])
    p.add_argument("--as-of", default="2025-08-01")
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()
    wf = TaxGuardWorkflow(audit_path=str(ROOT / "logs" / "debug_graph_audit.jsonl"), use_dense=False)
    initial = {"question": args.question, "country_code": args.country, "as_of_date": args.as_of,
               "topic": None, "top_k": args.top_k, "trace": [], "safety_flags": []}
    if wf.graph is None:
        print("LangGraph isn't installed/loaded; workflow is using its deterministic fallback.")
        result = wf.ask(args.question, args.country, args.as_of, top_k=args.top_k)
        print("Execution trace:", result.get("execution_trace", []))
        print("Status:", result.get("status"))
        print("Plan:", result.get("plan", {}))
        print("Citation count:", len(result.get("citations", [])))
        return 0
    print("Streaming graph node updates. Original question text is not printed from state.")
    for event in wf.graph.stream(initial, stream_mode="updates"):
        for node, update in event.items():
            print(f"\nNODE: {node}")
            print("Updated fields:", list(update.keys()))
            if "redacted_question" in update:
                print("Redacted question:", update["redacted_question"])
            if "pii_counts" in update:
                print("PII types/counts:", update["pii_counts"])
            if "safety_flags" in update:
                print("Safety flags:", update["safety_flags"])
            if "plan" in update:
                print("Plan:", update["plan"])
            if "request" in update:
                req = update["request"]
                print("Request accepted:", req is not None)
            if "response" in update:
                resp = update["response"] or {}
                print("Response status:", resp.get("status"))
                print("Citation count:", len(resp.get("citations", [])))
                print("Execution trace:", resp.get("execution_trace", []))
                print("Retrieval mode:", resp.get("retrieval_mode"))
    print("\nDebugging tip: inspect src/taxguard/graph/builder.py; each printed NODE maps to a node function.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

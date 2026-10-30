"""TaxGuard AI command-line interface."""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from taxguard.ingestion.pipeline import load_documents, write_manifest
from taxguard.retrieval.search import PolicyRetriever
from taxguard.workflow import TaxGuardWorkflow

def show(obj): print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
def workflow(args, audit="cli_audit.jsonl"):
    return TaxGuardWorkflow(corpus_path=str(ROOT/args.corpus), manifest_path=str(ROOT/args.manifest),
        audit_path=str(ROOT/"logs"/audit), use_dense=getattr(args,"dense",False), chroma_path=str(ROOT/args.chroma))
def ingest(args):
    chunks=load_documents(str(ROOT/args.corpus)); delta=write_manifest(chunks,str(ROOT/args.manifest))
    print(f"Documents: {len(set(c.source_path for c in chunks))}\nChunks: {len(chunks)}\nChanged: {len(delta['changed'])}\nUnchanged: {len(delta['unchanged'])}\nRemoved: {len(delta['removed'])}\nManifest: {ROOT/args.manifest}")
    if args.dense:
        r=PolicyRetriever(chunks,use_dense=True,chroma_path=str(ROOT/args.chroma))
        if not r.dense:
            print(f"Dense store unavailable: {r.dense_error}",file=sys.stderr); return 2
        print(f"Chroma synced using all-MiniLM-L6-v2 at {ROOT/args.chroma}")
    else: print("No vector store requested. Default is in-memory TF-IDF + BM25 + RRF.")
    return 0
def retrieve(args):
    wf=workflow(args)
    stages=wf.retriever.search_stages(args.question,args.country,args.as_of,args.top_k)
    print(f"Mode: {wf.retriever.mode} | eligible chunks={stages['eligible_count']}")
    if wf.retriever.dense_error: print(f"Dense warning: {wf.retriever.dense_error}")
    for name,label in [("tfidf","STAGE 1A — TF-IDF lexical ranking"),("bm25","STAGE 1B — BM25 keyword ranking"),("dense","STAGE 1C — Dense semantic ranking (MiniLM/Chroma)"),("fused","STAGE 2 — RRF fused ranking")]:
        items=stages[name]
        if name=="dense" and not stages.get("dense_enabled"):
            print(f"\n{label}: DISABLED (install requirements-ml.txt and pass --dense)"); continue
        print(f"\n{label}:")
        if not items: print("  No results"); continue
        for x in items:
            print(f"  {x['rank']}. {x['chunk_id']} | score={x['score']} | {x['title']}")
            print("     "+x['text'].replace("\n"," ")[:args.preview_chars])
    print("\nSTAGE 3 — Reranking: NOT IMPLEMENTED. RRF is rank fusion, not a cross-encoder reranker.")
    print("Scores are ranking signals, not calibrated probabilities. Raw dense cosine scores are not exposed by this version.")
    return 0

def generate(args):
    wf=workflow(args); result=wf.ask(args.question,args.country,args.as_of,topic=args.topic,top_k=args.top_k)
    if result.get("status") not in {"answered","needs_review"} or not result.get("citations"):
        show({"status":result.get("status"),"answer":result.get("answer"),"citations":result.get("citations",[]),"generation":"skipped: evidence gate did not supply eligible citations"}); return 0
    retrieved=[x["chunk"] for x in result.get("results",[])[:args.top_k] if isinstance(x,dict) and isinstance(x.get("chunk"),dict)]
    context="\n\n".join(f"[chunk_id={x['chunk_id']} document_id={x['document_id']} version={x['version']} section={x['section']}]\n{x['text']}" for x in retrieved)
    from taxguard.llm_provider import OpenAICompatibleProvider
    from taxguard.safety.response_validation import validate_llm_response
    from taxguard.safety.grounding import validate_grounding
    provider=OpenAICompatibleProvider()
    if not provider.configured:
        show({"status":result.get("status"),"evidence_answer":result.get("answer"),"citations":result.get("citations"),"generation":"not called; configure TAXGUARD_LLM_BASE_URL, TAXGUARD_LLM_API_KEY and TAXGUARD_LLM_MODEL"}); return 2
    try:
        raw=provider.answer(args.question,args.country,args.as_of,context)
        validation=validate_llm_response(raw,retrieved)
        if not validation["valid"]:
            show({"status":"HUMAN_REVIEW","answer":"The generated response failed validation and has been withheld.","citations":[],"missing_information":[],"safety_flags":["response_validation_failed"],"validation_errors":validation["validation_errors"],"limitations":["Synthetic demo documents only; not legal or tax advice.","Structural checks do not prove semantic correctness."]}); return 2
        payload=validation["response"]
        grounding=validate_grounding(payload["answer"],payload["citations"],retrieved)
        payload["validation"]={"json_schema_fields":"passed","citation_provenance":"passed","semantic_grounding":"heuristic_pass" if grounding["grounded_heuristically"] else "not_passed","grounding_details":grounding,"human_review_recommended":True}
        payload["limitations"]=["Synthetic demo documents only; not legal or tax advice.","Grounding uses lexical overlap and does not prove entailment.","Human review is required for consequential tax decisions."]
        if payload["status"]=="ANSWERED" and not grounding["grounded_heuristically"]:
            payload["status"]="HUMAN_REVIEW"; payload["safety_flags"]=sorted(set(payload["safety_flags"]+["grounding_check_failed"]))
            payload["answer"]="The generated response did not pass the heuristic evidence-grounding check and has been withheld for review."
        show(payload)
        return 0 if payload["status"] in {"ANSWERED","ABSTAINED","CLARIFICATION_REQUIRED"} else 2
    except Exception as e: print(f"Generation failed: {type(e).__name__}: {e}",file=sys.stderr); return 2

def evaluate(args):
    env=os.environ.copy(); env["TAXGUARD_USE_DENSE"]="true" if args.dense else "false"
    return subprocess.run([sys.executable,str(ROOT/"scripts/run_evaluation.py")],cwd=ROOT,env=env).returncode
def eval_generation(args):
    from taxguard.llm_provider import OpenAICompatibleProvider
    provider=OpenAICompatibleProvider()
    if not provider.configured:
        print("Configure TAXGUARD_LLM_BASE_URL, TAXGUARD_LLM_API_KEY and TAXGUARD_LLM_MODEL first.",file=sys.stderr); return 2
    cases=[json.loads(s) for s in (ROOT/args.dataset).read_text().splitlines() if s.strip()]
    wf=workflow(args); rows=[]
    for case in cases:
        if case.get("expected_status")!="answered": continue
        rs=wf.retriever.search(case["question"],case["country_code"],case["as_of_date"],args.top_k)
        ctx="\n\n".join(f"[chunk_id={r.chunk.chunk_id} document_id={r.chunk.document_id} version={r.chunk.version} section={r.chunk.section}]\n{r.chunk.text}" for r in rs)
        row={"id":case["id"],"question":case["question"],"expected_answer":case.get("expected_answer"),"retrieved_chunk_ids":[r.chunk.chunk_id for r in rs]}
        try:
            raw=provider.answer(case["question"],case["country_code"],case["as_of_date"],ctx); row["raw_output"]=raw
            parsed=json.loads(raw); row["parsed_json"]=parsed; answer=str(parsed.get("answer","")); cites=parsed.get("citations",[])
            cids=[str(c.get("chunk_id","")) for c in cites if isinstance(c,dict)]
            row.update(generation_status="COMPLETED",valid_json=True,has_answer=bool(answer.strip()),citation_ids=cids,citations_in_context=bool(cids) and all(x in row["retrieved_chunk_ids"] for x in cids),expected_answer_evidence_covered=bool(case.get("expected_answer","").lower() in ctx.lower()))
            row["heuristic_pass"]=row["valid_json"] and row["has_answer"] and row["citations_in_context"]
            display="PASS*" if row["heuristic_pass"] else "REVIEW"
        except Exception as e:
            # Keep provider/API failures separate from answer-quality review.
            is_provider_error = type(e).__name__ in {"LLMProviderError", "HTTPError", "URLError", "TimeoutError"}
            row.update(generation_status="ERROR" if is_provider_error else "REVIEW", error=f"{type(e).__name__}: {e}", heuristic_pass=False)
            display = "ERROR" if is_provider_error else "REVIEW"
        rows.append(row); print(f"{case['id']}: {display}")
    passed=sum(bool(r.get("heuristic_pass")) for r in rows)
    completed=sum(r.get("generation_status")=="COMPLETED" for r in rows)
    errors=sum(r.get("generation_status")=="ERROR" for r in rows)
    reviews=sum(r.get("generation_status")=="REVIEW" for r in rows)
    report={"note":"Heuristic structural/citation checks only; not validated answer accuracy. Human review required.","model":provider.model,"total_cases":len(rows),"completed_generations":completed,"provider_errors":errors,"format_or_quality_reviews":reviews,"heuristic_pass":passed,"heuristic_pass_pct_of_completed":round(100*passed/max(1,completed),2) if completed else None,"cases":rows}
    out=ROOT/args.output; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"\nCases: {len(rows)} | completed: {completed} | provider errors: {errors} | review: {reviews}")
    if completed: print(f"Heuristic pass: {passed}/{completed} completed generations ({report['heuristic_pass_pct_of_completed']}%)")
    else: print("No generations completed; answer-quality metrics are not available until the provider error is fixed.")
    print(f"Report: {out}\nDo not call this true accuracy; review answer correctness, groundedness, completeness and citations.")
    return 0 if errors == 0 else 2

def main():
    p=argparse.ArgumentParser(prog="taxguard",description="TaxGuard CLI: ingest, retrieve, generate and evaluate")
    sub=p.add_subparsers(dest="cmd",required=True)
    a=sub.add_parser("ingest",help="Parse documents, chunk, update manifest; optionally embed/store in Chroma")
    a.add_argument("--corpus",default="data/synthetic_policies"); a.add_argument("--manifest",default="data/index/manifest.json"); a.add_argument("--dense",action="store_true"); a.add_argument("--chroma",default="data/chroma"); a.set_defaults(func=ingest)
    for name,func,helptext in [("retrieve",retrieve,"Show TF-IDF/BM25 and fused top-k results"),("generate",generate,"Retrieve evidence and call configured hosted LLM")]:
        a=sub.add_parser(name,help=helptext); a.add_argument("--question",required=True); a.add_argument("--country",required=True,choices=["US","UK","IN","CA","AU"]); a.add_argument("--as-of",required=True,dest="as_of"); a.add_argument("--top-k",type=int,default=5); a.add_argument("--preview-chars",type=int,default=350); a.add_argument("--topic",default=None); a.add_argument("--corpus",default="data/synthetic_policies"); a.add_argument("--manifest",default=f"data/index/{name}_manifest.json"); a.add_argument("--dense",action="store_true"); a.add_argument("--chroma",default="data/chroma"); a.set_defaults(func=func)
    a=sub.add_parser("evaluate",help="Run retrieval/decision/abstention evaluation and write reports"); a.add_argument("--dense",action="store_true"); a.set_defaults(func=evaluate)
    a=sub.add_parser("evaluate-generation",help="Run configured LLM over answerable cases and emit review report"); a.add_argument("--dataset",default="data/eval/questions.jsonl"); a.add_argument("--top-k",type=int,default=5); a.add_argument("--dense",action="store_true"); a.add_argument("--corpus",default="data/synthetic_policies"); a.add_argument("--manifest",default="data/index/generation_eval_manifest.json"); a.add_argument("--chroma",default="data/chroma"); a.add_argument("--output",default="reports/generation_evaluation.json"); a.set_defaults(func=eval_generation)
    args=p.parse_args(); return args.func(args)
if __name__=="__main__": raise SystemExit(main())

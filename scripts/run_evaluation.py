"""Run 50 labeled test cases and produce reproducible retrieval/abstention metrics."""
import json, sys
from pathlib import Path
from statistics import mean
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from taxguard.workflow import TaxGuardWorkflow
from taxguard.agents.planner import TaxPolicyPlannerAgent

ROOT = Path(__file__).resolve().parents[1]
def main():
    cases = [json.loads(s) for s in (ROOT/"data/eval/questions.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    wf = TaxGuardWorkflow(corpus_path=str(ROOT/"data/synthetic_policies"), manifest_path=str(ROOT/"data/index/eval_manifest.json"),
                          audit_path=str(ROOT/"logs/evaluation_audit.jsonl"), use_dense=None,
                          chroma_path=str(ROOT/"data/chroma"))
    rows=[]; decision_correct=[]; top1=[]; recall5=[]; reciprocal=[]; precision5=[]; chunk_recall5=[]; chunk_mrr=[]; expected_answer_covered=[]; plan_correct=[]
    tp=fp=fn=tn=0
    for c in cases:
        # Evaluate retrieval directly, so metrics are not confused with answer generation.
        retrieved = wf.retriever.search(c['question'],c['country_code'],c['as_of_date'],top_k=5)
        r = wf.ask(c['question'],c['country_code'],c['as_of_date'],top_k=5)
        expected = c['expected_status']; actual = r.get('status','error')
        status_ok = actual == expected
        decision_correct.append(int(status_ok))
        is_abstain_expected = expected == 'abstained'; is_abstain_actual = actual == 'abstained'
        if is_abstain_expected and is_abstain_actual: tp += 1
        elif not is_abstain_expected and is_abstain_actual: fp += 1
        elif is_abstain_expected and not is_abstain_actual: fn += 1
        else: tn += 1
        relevant = c.get('expected_document_id')
        ranks = [i+1 for i,x in enumerate(retrieved) if relevant and x.chunk.document_id == relevant]
        chunk_ranks = [i+1 for i,x in enumerate(retrieved) if c.get('expected_chunk_id') and x.chunk.chunk_id == c['expected_chunk_id']]
        hit = bool(ranks); chunk_hit = bool(chunk_ranks)
        plan = TaxPolicyPlannerAgent().plan(c['question'], c['country_code'], c['as_of_date'])
        plan_ok = plan.route == c.get('expected_plan_route', 'policy_lookup'); plan_correct.append(int(plan_ok))
        answer_covered = bool(relevant and any(c['expected_answer'].lower() in x.chunk.text.lower() for x in retrieved))
        if relevant:
            recall5.append(int(hit)); top1.append(int(bool(retrieved and retrieved[0].chunk.document_id == relevant)))
            reciprocal.append(1/ranks[0] if ranks else 0.0); precision5.append(sum(x.chunk.document_id == relevant for x in retrieved)/5)
            chunk_recall5.append(int(chunk_hit)); chunk_mrr.append(1/chunk_ranks[0] if chunk_ranks else 0.0); expected_answer_covered.append(int(answer_covered))
        row={'id':c['id'],'case_type':c['case_type'],'expected_status':expected,'actual_status':actual,'status_correct':status_ok,
             'expected_document_id':relevant,'expected_chunk_id':c.get('expected_chunk_id'),
             'retrieved_document_ids':[x.chunk.document_id for x in retrieved],
             'retrieved_chunk_ids':[x.chunk.chunk_id for x in retrieved],
             'relevant_doc_rank':ranks[0] if ranks else None,'recall_at_5':hit if relevant else None,
             'expected_chunk_rank':chunk_ranks[0] if chunk_ranks else None,'expected_chunk_recall_at_5':chunk_hit if relevant else None,
             'expected_answer_evidence_covered':answer_covered if relevant else None,'planner_expected_route':c.get('expected_plan_route'),'planner_actual_route':plan.route,'planner_route_correct':plan_ok,
             'top1_correct':bool(retrieved and retrieved[0].chunk.document_id == relevant) if relevant else None,
             'expected_answer':c['expected_answer'],'actual_citations':[x.get('document_id') for x in r.get('citations',[])],
             'retrieval_mode':wf.retriever.mode,'safety_flags':r.get('safety_flags',[])}
        rows.append(row)
    total=len(cases); supported=sum(c['case_type']=='supported' for c in cases); abstention_cases=total-supported
    status_accuracy=sum(decision_correct)/max(1,total)
    metrics={
      'evaluation_name':'TaxGuard AI 50-case synthetic retrieval and abstention evaluation',
      'total_cases':total,'supported_cases':supported,'expected_abstention_cases':abstention_cases,
      'validated_cases':sum(1 for x in rows if x['status_correct']), 'decision_accuracy':status_accuracy,
      'decision_accuracy_pct':round(100*status_accuracy,2),
      'retrieval_recall_at_5':mean(recall5) if recall5 else 0,
      'retrieval_recall_at_5_pct':round(100*mean(recall5),2) if recall5 else 0,
      'retrieval_top1_accuracy_pct':round(100*mean(top1),2) if top1 else 0,
      'retrieval_mrr':round(mean(reciprocal),4) if reciprocal else 0,
      'retrieval_precision_at_5':round(mean(precision5),4) if precision5 else 0,
      'expected_chunk_recall_at_5_pct':round(100*mean(chunk_recall5),2) if chunk_recall5 else 0,
      'expected_chunk_mrr':round(mean(chunk_mrr),4) if chunk_mrr else 0,
      'expected_answer_evidence_coverage_pct':round(100*mean(expected_answer_covered),2) if expected_answer_covered else 0,
      'planner_route_accuracy_pct':round(100*mean(plan_correct),2) if plan_correct else 0,
      'abstention_precision':round(tp/max(1,tp+fp),4),
      'abstention_recall':round(tp/max(1,tp+fn),4),
      'abstention_f1':round((2*tp)/max(1,2*tp+fp+fn),4),
      'abstention_confusion':{'true_abstain':tp,'false_abstain':fp,'missed_abstain':fn,'correct_answerable':tn},
      'retrieval_mode':wf.retriever.mode,
      'dense_model_evaluated':bool(wf.retriever.dense),
      'important_scope_note':'These are measurements on synthetic, hand-labeled test cases. Retrieval metrics measure expected-document retrieval; decision accuracy measures expected answered/abstained status. No generative LLM was called, so this is not LLM answer accuracy.'
    }
    (ROOT/'reports').mkdir(exist_ok=True)
    (ROOT/'reports/evaluation_results.json').write_text(json.dumps({'metrics':metrics,'cases':rows},indent=2),encoding='utf-8')
    lines=['# TaxGuard AI — Evaluation Report','',f"**Validated status decisions:** {metrics['validated_cases']}/{total} ({metrics['decision_accuracy_pct']}%)",'',
      '## Results','',f"- Test cases: {total} ({supported} supported; {abstention_cases} expected abstentions)",
      f"- Decision accuracy: **{metrics['decision_accuracy_pct']}%**",
      f"- Retrieval Recall@5: **{metrics['retrieval_recall_at_5_pct']}%**",
      f"- Retrieval top-1 accuracy: **{metrics['retrieval_top1_accuracy_pct']}%**",
      f"- MRR: **{metrics['retrieval_mrr']}**",
      f"- Precision@5: **{metrics['retrieval_precision_at_5']}**",
      f"- Expected chunk Recall@5: **{metrics['expected_chunk_recall_at_5_pct']}%**",
      f"- Expected chunk MRR: **{metrics['expected_chunk_mrr']}**",
      f"- Expected-answer evidence coverage: **{metrics['expected_answer_evidence_coverage_pct']}%**",
      f"- Deterministic planner route accuracy: **{metrics['planner_route_accuracy_pct']}%**",
      f"- Abstention precision: **{metrics['abstention_precision']}**",
      f"- Abstention recall: **{metrics['abstention_recall']}**",
      f"- Abstention F1: **{metrics['abstention_f1']}**",'',
      f"**Retrieval mode actually run:** `{metrics['retrieval_mode']}`",'',
      '## Important interpretation','',metrics['important_scope_note'],'',
      'The default evaluation environment uses BM25 + TF-IDF candidate rankings fused by RRF. The optional Chroma + all-MiniLM-L6-v2 dense branch is implemented behind `TAXGUARD_USE_DENSE=true` and optional dependencies, but is only counted as evaluated when those packages and model weights are available. Do not describe these scores as dense-hybrid results unless `dense_model_evaluated` is true.', '',
      '## Case-level results','', '| Case | Type | Expected | Actual | Status | Doc rank | Chunk rank |', '|---|---|---|---|---:|---:|---:|']
    for x in rows:
        lines.append(f"| {x['id']} | {x['case_type']} | {x['expected_status']} | {x['actual_status']} | {'PASS' if x['status_correct'] else 'FAIL'} | {x['relevant_doc_rank'] or '—'} | {x['expected_chunk_rank'] or '—'} |")
    (ROOT/'reports/evaluation_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines[:18]))
    print(f"\nDetailed results: reports/evaluation_results.json and reports/evaluation_report.md")
    return 0 if metrics['validated_cases']==total else 1
if __name__=='__main__': raise SystemExit(main())

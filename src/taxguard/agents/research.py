from taxguard.safety.guards import assess_evidence, validate_citations
from taxguard.observability.audit import new_request_id, write_audit
class PolicyResearchAgent:
    def __init__(self,retriever): self.retriever=retriever
    def answer(self,request,top_k=5,audit_path="logs/audit.jsonl"):
        rid=new_request_id(); results=self.retriever.search(request.question,request.country_code,request.as_of_date,top_k)
        enough,flags=assess_evidence(results); citations=[]
        if enough:
            best=results[:min(3,len(results))]
            citations=[{"chunk_id":r.chunk.chunk_id,"document_id":r.chunk.document_id,"version":r.chunk.version,
                "title":r.chunk.title,"section":r.chunk.section,"effective_from":r.chunk.effective_from,
                "effective_to":r.chunk.effective_to,"source_path":r.chunk.source_path,"score":round(r.score,4)} for r in best]
            answer=("Potentially relevant synthetic policy evidence follows. This is not real tax guidance.\n\n"+
                "\n\n".join(f"- **{r.chunk.title}** ({r.chunk.document_id}, v{r.chunk.version}; {r.chunk.section}): {r.chunk.text}" for r in best))
            status="answered"
            if not validate_citations(citations,results):
                status="needs_review"; answer="Citation validation failed; response withheld."; citations=[]; flags.append("citation_validation_failed")
        else:
            answer="I can't provide a supported answer from the eligible synthetic policy corpus. Clarify the jurisdiction, applicable date, or topic, or consult a qualified tax professional."
            status="abstained"
        limitations=["Synthetic demo documents only; not legal or tax advice.","Ranking scores are not calibrated probabilities."]
        write_audit({"request_id":rid,"status":status,"country_code":request.country_code,"as_of_date":request.as_of_date,
            "result_count":len(results),"top_score":round(results[0].score,4) if results else None,
            "safety_flags":flags,"citation_count":len(citations)},audit_path)
        return {"status":status,"answer":answer,"country_code":request.country_code,"as_of_date":request.as_of_date,
            "citations":citations,"limitations":limitations,"safety_flags":flags,"request_id":rid,
            "results":[{"score":round(r.score,4),"rrf_score":round(getattr(r,"rrf_score",r.score),6),"relevance_score":round(getattr(r,"relevance_score",r.score),4),"tfidf_score":round(getattr(r,"tfidf_score",0),4),"bm25_score":round(getattr(r,"bm25_score",0),4),"lexical_overlap":round(r.lexical_overlap,4),"dense_enabled":getattr(r,"dense_enabled",False),"chunk":r.chunk.model_dump()} for r in results]}

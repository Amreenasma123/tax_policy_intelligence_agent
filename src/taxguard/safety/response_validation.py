"""Structural and citation-provenance validation for generated JSON responses."""
import json
ALLOWED_STATUSES = {"ANSWERED", "CLARIFICATION_REQUIRED", "ABSTAINED", "HUMAN_REVIEW"}

def validate_llm_response(raw, retrieved_chunks):
    try:
        if not isinstance(raw, str): raise ValueError("Response must be a string.")
        payload=json.loads(raw)
        if not isinstance(payload,dict): raise ValueError("Response must be a JSON object.")
        required={"status","answer","citations","missing_information","safety_flags"}
        if not required.issubset(payload): raise ValueError("Missing required response fields.")
        status=payload["status"]
        if not isinstance(status,str) or status not in ALLOWED_STATUSES: raise ValueError("status is not an allowed status.")
        if not isinstance(payload["answer"],str): raise ValueError("answer must be a string.")
        for field in ("citations","missing_information","safety_flags"):
            if not isinstance(payload[field],list): raise ValueError(f"{field} must be a list; received {type(payload[field]).__name__}.")
        allowed={}
        for chunk in retrieved_chunks:
            if isinstance(chunk,dict) and chunk.get("chunk_id"):
                allowed[chunk["chunk_id"]]={k:chunk.get(k) for k in ("document_id","version","section")}
        if status=="ANSWERED" and (not payload["answer"].strip() or not payload["citations"]):
            raise ValueError("ANSWERED requires a non-empty answer and citations.")
        if status in {"ABSTAINED","CLARIFICATION_REQUIRED","HUMAN_REVIEW"} and not payload["answer"].strip():
            raise ValueError(f"{status} requires a user-facing explanation.")
        for citation in payload["citations"]:
            if not isinstance(citation,dict): raise ValueError("Each citation must be an object.")
            for field in ("chunk_id","document_id","version","section"):
                if not isinstance(citation.get(field),str) or not citation[field].strip(): raise ValueError(f"Citation {field} must be a non-empty string.")
            expected=allowed.get(citation["chunk_id"])
            if expected is None: raise ValueError("Citation references an unretrieved chunk.")
            for field in ("document_id","version","section"):
                if citation[field] != expected[field]: raise ValueError(f"Citation metadata mismatch: {field}.")
        return {"valid":True,"response":payload,"validation_errors":[]}
    except (ValueError,TypeError,json.JSONDecodeError) as exc:
        return {"valid":False,"response":None,"validation_errors":[str(exc)]}

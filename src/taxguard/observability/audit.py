import json, uuid
from pathlib import Path
from datetime import datetime, timezone
def new_request_id(): return str(uuid.uuid4())
def write_audit(event, path="logs/audit.jsonl"):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    safe=dict(event); safe.setdefault("timestamp_utc",datetime.now(timezone.utc).isoformat())
    with p.open("a",encoding="utf-8") as f: f.write(json.dumps(safe,ensure_ascii=False)+"\n")

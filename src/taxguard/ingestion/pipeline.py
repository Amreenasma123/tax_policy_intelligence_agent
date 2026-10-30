import hashlib, json, re
from pathlib import Path
FRONT = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.S)
def parse_frontmatter(text):
    match = FRONT.match(text)
    if not match: raise ValueError("Document missing YAML frontmatter")
    import yaml
    return yaml.safe_load(match.group(1)) or {}, match.group(2).strip()
def chunk_text(text, max_chars=900, overlap=120):
    sections = re.split(r"(?m)^#{1,3}\s+", text)
    chunks = []
    for section in sections:
        section = section.strip()
        if not section: continue
        lines = section.splitlines()
        heading = lines[0].strip()[:120] if lines else "Body"
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else section
        if not body: continue
        start = 0
        while start < len(body):
            piece = body[start:start + max_chars].strip()
            if piece: chunks.append((heading, piece))
            if start + max_chars >= len(body): break
            start += max_chars - overlap
    return chunks
def load_documents(root):
    from taxguard.schemas import PolicyChunk
    output = []
    for path in sorted(Path(root).rglob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(raw)
        digest = hashlib.sha256(raw.encode()).hexdigest()
        for i, (section, chunk) in enumerate(chunk_text(body), 1):
            output.append(PolicyChunk(
                chunk_id=f"{meta['document_id']}:{meta['version']}:{i}",
                document_id=str(meta["document_id"]), country_code=str(meta["country_code"]),
                title=str(meta["title"]), topic=str(meta["topic"]).lower(), version=str(meta["version"]),
                effective_from=str(meta["effective_from"]), effective_to=str(meta["effective_to"]),
                status=str(meta.get("status", "active")), section=section, text=chunk,
                source_path=str(path), content_hash=digest, source_type=str(meta.get("source_type","synthetic"))))
    return output
def write_manifest(chunks, manifest_path):
    manifest_path = Path(manifest_path); manifest_path.parent.mkdir(parents=True, exist_ok=True)
    files = {c.source_path: c.content_hash for c in chunks}
    previous = json.loads(manifest_path.read_text())["files"] if manifest_path.exists() else {}
    changed = sorted(p for p,h in files.items() if previous.get(p) != h)
    removed = sorted(p for p in previous if p not in files)
    manifest_path.write_text(json.dumps({"files":files}, indent=2), encoding="utf-8")
    return {"changed":changed, "removed":removed, "unchanged":sorted(p for p,h in files.items() if previous.get(p)==h)}

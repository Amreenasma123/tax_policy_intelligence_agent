"""Optional Redis-backed, per-session interaction history for the demo."""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from taxguard.safety.pii import redact_pii


class InteractionHistory:
    def __init__(self, redis_url: str | None = None, ttl_seconds: int | None = None):
        self.redis_url = redis_url or os.getenv("TAXGUARD_REDIS_URL", "")
        self.ttl_seconds = ttl_seconds if ttl_seconds is not None else int(os.getenv("TAXGUARD_HISTORY_TTL_SECONDS", "2592000"))
        self.client = None
        self.error = None
        if self.redis_url:
            try:
                import redis
                client = redis.Redis.from_url(self.redis_url, decode_responses=True, socket_connect_timeout=2, socket_timeout=2)
                client.ping()
                self.client = client
            except Exception as exc:
                self.error = type(exc).__name__

    @property
    def available(self) -> bool:
        return self.client is not None

    def save(self, session_id: str, question: str, response: dict) -> bool:
        if not self.client:
            return False
        safe_question = redact_pii(question or "").text
        safe_answer = redact_pii(str(response.get("answer", ""))).text
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "question": safe_question,
            "answer": safe_answer,
            "status": str(response.get("status", "unknown")).upper(),
            "citations": response.get("citations", []),
            "safety_flags": response.get("safety_flags", []),
        }
        key = f"taxguard:history:{session_id}"
        self.client.lpush(key, json.dumps(record, ensure_ascii=False))
        self.client.ltrim(key, 0, 49)
        if self.ttl_seconds > 0:
            self.client.expire(key, self.ttl_seconds)
        return True

    def get(self, session_id: str, limit: int = 20) -> list[dict]:
        if not self.client:
            return []
        rows = self.client.lrange(f"taxguard:history:{session_id}", 0, max(0, limit - 1))
        output = []
        for row in rows:
            try:
                output.append(json.loads(row))
            except (TypeError, json.JSONDecodeError):
                continue
        return output

    def clear(self, session_id: str) -> bool:
        if not self.client:
            return False
        self.client.delete(f"taxguard:history:{session_id}")
        return True

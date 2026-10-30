from pydantic import BaseModel, Field
from typing import Literal

class UserRequest(BaseModel):
    question: str = Field(min_length=5, max_length=2000)
    country_code: Literal["US", "UK", "IN", "CA", "AU"]
    as_of_date: str
    topic: str | None = None

class PolicyChunk(BaseModel):
    chunk_id: str
    document_id: str
    country_code: str
    title: str
    topic: str
    version: str
    effective_from: str
    effective_to: str
    status: str
    section: str
    text: str
    source_path: str
    content_hash: str
    source_type: str = "synthetic"

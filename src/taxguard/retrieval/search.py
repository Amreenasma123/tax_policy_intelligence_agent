"""Hybrid retrieval: TF-IDF + BM25 with RRF fusion; optional Chroma/MiniLM dense branch.

The default installation remains lightweight and offline-capable. Install the optional
extras in requirements-ml.txt to enable the dense branch and persistent Chroma store.
"""
from __future__ import annotations
from collections import Counter
from datetime import date
import math
import os
import re
from types import SimpleNamespace
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)

def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())

class BM25Index:
    """Small dependency-free BM25 implementation suitable for this demo corpus."""
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [tokens(x) for x in documents]
        self.k1, self.b = k1, b
        self.lengths = [len(x) for x in self.docs]
        self.avgdl = sum(self.lengths) / max(1, len(self.lengths))
        self.term_freqs = [Counter(d) for d in self.docs]
        df = Counter()
        for doc in self.docs:
            df.update(set(doc))
        n = len(self.docs)
        self.idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def scores(self, query: str, indices: list[int]) -> dict[int, float]:
        qterms = set(tokens(query)); out = {}
        for i in indices:
            tf = self.term_freqs[i]; dl = self.lengths[i]; score = 0.0
            for term in qterms:
                f = tf.get(term, 0)
                if not f: continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / max(1, self.avgdl))
                score += self.idf.get(term, 0.0) * f * (self.k1 + 1) / denom
            out[i] = score
        return out

class OptionalChromaMiniLM:
    """Persistent Chroma store with all-MiniLM-L6-v2 embeddings; lazy-loaded on demand."""
    def __init__(self, persist_dir: str = "data/chroma", collection_name: str = "taxguard_chunks"):
        try:
            import chromadb
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        except ImportError as exc:
            raise RuntimeError("Dense retrieval requires optional packages in requirements-ml.txt") from exc
        self.client = chromadb.PersistentClient(path=persist_dir)
        ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.collection = self.client.get_or_create_collection(collection_name, embedding_function=ef, metadata={"hnsw:space": "cosine"})

    def sync(self, chunks) -> None:
        # Rebuild deterministically for this small demo. Production should stage and atomically swap indexes.
        existing = self.collection.get(include=[])
        ids = existing.get("ids", [])
        if ids: self.collection.delete(ids=ids)
        batch = list(chunks)
        for start in range(0, len(batch), 64):
            part = batch[start:start+64]
            self.collection.add(ids=[c.chunk_id for c in part], documents=[c.text for c in part],
                metadatas=[{"document_id":c.document_id,"country_code":c.country_code,"title":c.title,
                "topic":c.topic,"version":c.version,"effective_from":c.effective_from,
                "effective_to":c.effective_to,"status":c.status,"section":c.section,
                "content_hash":c.content_hash,"source_path":c.source_path} for c in part])

    def query(self, query: str, eligible_chunks, top_k: int = 10) -> list[str]:
        # Query broadly then enforce eligibility against the authoritative in-memory metadata.
        res = self.collection.query(query_texts=[query], n_results=min(max(top_k * 8, 20), max(1, self.collection.count())))
        found = (res.get("ids") or [[]])[0]
        eligible_ids = {c.chunk_id for c in eligible_chunks}
        return [x for x in found if x in eligible_ids][:top_k]

class PolicyRetriever:
    @property
    def mode(self):
        return "bm25+tfidf+rrf+dense-minilm" if self.dense else "bm25+tfidf+rrf (dense optional; disabled)"

    def __init__(self, chunks, use_dense: bool | None = None, chroma_path: str = "data/chroma"):
        self.chunks = chunks
        self.documents = [c.text + " " + c.title + " " + c.topic for c in chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(self.documents)
        self.bm25 = BM25Index(self.documents)
        self.dense = None; self.dense_error = None
        if use_dense is None:
            use_dense = os.getenv("TAXGUARD_USE_DENSE", "false").lower() in {"1", "true", "yes"}
        if use_dense:
            try:
                self.dense = OptionalChromaMiniLM(chroma_path)
                self.dense.sync(chunks)
            except Exception as exc:  # Keep app usable if optional model/provider is unavailable.
                self.dense_error = f"Dense retrieval unavailable: {type(exc).__name__}: {exc}"

    @staticmethod
    def eligible(c, country_code, as_of_date):
        try:
            d = date.fromisoformat(as_of_date); start = date.fromisoformat(c.effective_from); end = date.fromisoformat(c.effective_to)
        except (ValueError, TypeError): return False
        return c.country_code == country_code and start <= d < end and str(c.status).strip().lower() == "active"


    def search_stages(self, query, country_code, as_of_date, top_k=5):
        """Return intermediate lexical/dense rankings and final RRF ranking for debugging."""
        idx = [i for i,c in enumerate(self.chunks) if self.eligible(c,country_code,as_of_date)]
        if not idx:
            return {"eligible_count": 0, "tfidf": [], "bm25": [], "dense": [], "fused": []}
        tfidf = cosine_similarity(self.vectorizer.transform([query]), self.matrix[idx]).ravel()
        tfidf_scores = {i: float(v) for i,v in zip(idx,tfidf)}
        bm25_scores = self.bm25.scores(query,idx)
        qterms=set(tokens(query)); overlap={i:len(qterms & set(tokens(self.documents[i])))/max(1,len(qterms)) for i in idx}
        tfidf_rank=sorted(idx,key=lambda i:tfidf_scores[i],reverse=True)
        bm25_rank=sorted(idx,key=lambda i:bm25_scores[i],reverse=True)
        dense_rank=[]
        if self.dense:
            eligible_chunks=[self.chunks[i] for i in idx]
            ids=self.dense.query(query,eligible_chunks,top_k=min(20,len(idx)))
            id_to_idx={self.chunks[i].chunk_id:i for i in idx}
            dense_rank=[id_to_idx[x] for x in ids if x in id_to_idx]
        rank_lists=[tfidf_rank,bm25_rank]+([dense_rank] if dense_rank else [])
        rrf={i:0.0 for i in idx}; k=60
        for ranks in rank_lists:
            for rank,i in enumerate(ranks,start=1): rrf[i]+=1.0/(k+rank)
        fused_rank=sorted(idx,key=lambda i:(rrf[i],0.55*tfidf_scores[i]+0.25*(bm25_scores[i]/(max(bm25_scores.values()) or 1.0))+0.20*overlap[i]),reverse=True)
        def pack(order):
            out=[]
            for rank,i in enumerate(order[:top_k],start=1):
                c=self.chunks[i]
                out.append({"rank":rank,"chunk_id":c.chunk_id,"document_id":c.document_id,"title":c.title,
                    "score":round(tfidf_scores[i],6) if order is tfidf_rank else round(bm25_scores[i],6) if order is bm25_rank else round(rrf[i],8) if order is fused_rank else None,
                    "tfidf_score":round(tfidf_scores[i],6),"bm25_score":round(bm25_scores[i],6),
                    "text":c.text,"country_code":c.country_code,"effective_from":c.effective_from,"effective_to":c.effective_to})
            return out
        return {"eligible_count":len(idx),"tfidf":pack(tfidf_rank),"bm25":pack(bm25_rank),
                "dense":pack(dense_rank),"fused":pack(fused_rank),"dense_enabled":bool(self.dense),
                "reranker_enabled":False}

    def search(self, query, country_code, as_of_date, top_k=5):
        idx = [i for i,c in enumerate(self.chunks) if self.eligible(c,country_code,as_of_date)]
        if not idx: return []
        tfidf = cosine_similarity(self.vectorizer.transform([query]), self.matrix[idx]).ravel()
        tfidf_scores = {i: float(v) for i,v in zip(idx, tfidf)}
        bm25_scores = self.bm25.scores(query, idx)
        qterms = set(tokens(query)); overlap_scores = {}
        for i in idx:
            dterms = set(tokens(self.documents[i]))
            overlap_scores[i] = len(qterms & dterms) / max(1, len(qterms))
        # Candidate rankings for Reciprocal Rank Fusion (RRF).
        rank_lists = [sorted(idx, key=lambda i: tfidf_scores[i], reverse=True),
                      sorted(idx, key=lambda i: bm25_scores[i], reverse=True)]
        dense_rank_ids = []
        if self.dense:
            eligible_chunks = [self.chunks[i] for i in idx]
            dense_rank_ids = self.dense.query(query, eligible_chunks, top_k=min(20, len(idx)))
            id_to_idx = {self.chunks[i].chunk_id:i for i in idx}
            if dense_rank_ids:
                dense_rank_ids = [id_to_idx[x] for x in dense_rank_ids if x in id_to_idx]
                rank_lists.append(dense_rank_ids)
        rrf = {i: 0.0 for i in idx}; k = 60
        for ranks in rank_lists:
            for rank, i in enumerate(ranks, start=1): rrf[i] += 1.0 / (k + rank)
        max_rrf = max(rrf.values()) or 1.0
        bmmax = max(bm25_scores.values()) or 1.0
        results = []
        for i in idx:
            # Confidence proxy is separate from RRF; not a calibrated probability.
            relevance = 0.55 * tfidf_scores[i] + 0.25 * (bm25_scores[i] / bmmax) + 0.20 * overlap_scores[i]
            results.append(SimpleNamespace(chunk=self.chunks[i], score=rrf[i]/max_rrf,
                rrf_score=rrf[i], relevance_score=relevance, tfidf_score=tfidf_scores[i],
                bm25_score=bm25_scores[i], lexical_overlap=overlap_scores[i],
                dense_enabled=bool(self.dense)))
        return sorted(results, key=lambda x:(x.rrf_score, x.relevance_score), reverse=True)[:top_k]

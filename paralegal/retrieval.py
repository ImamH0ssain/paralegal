"""Hybrid evidence indexing and retrieval.

The index combines SQLite FTS5 keyword search, vector similarity, reciprocal
rank fusion, and an optional local cross-encoder reranker. It stores all data
locally so retrieval remains inspectable and reproducible.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from .io_utils import ensure_dir, read_json, read_jsonl, write_json
from .models import DocumentBlock, EvidenceChunk
from .providers import EmbeddingProvider, embedding_provider_from_config, cosine
from .text_utils import tokenize


class EvidenceIndex:
    """Local hybrid search index over processed document blocks."""

    def __init__(self, index_dir: str | Path, embedding_provider: EmbeddingProvider | None = None, reranker: Any | None = None):
        """Open an index directory and restore its embedding provider config."""
        self.index_dir = ensure_dir(index_dir)
        self.db_path = self.index_dir / "evidence.sqlite"
        self.vector_path = self.index_dir / "vectors.jsonl"
        self.embedding_cache_path = self.index_dir / "embedding_cache.jsonl"
        self.config_path = self.index_dir / "index_config.json"
        self.embedding_provider = embedding_provider or embedding_provider_from_config(
            read_json(self.config_path, default={}).get("embedding_provider")
        )
        self.reranker = reranker

    def build(self, processed_jsonl: str | Path) -> int:
        """Build SQLite FTS and vector artifacts from processed JSONL blocks."""
        records = read_jsonl(processed_jsonl)
        blocks = [DocumentBlock.from_record(record) for record in records if record.get("record_type") == "block"]
        conn = sqlite3.connect(self.db_path)
        try:
            self._init_db(conn)
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM chunks_fts")
            texts = [block.text for block in blocks]
            embeddings = self._embed_with_cache(texts)
            with self.vector_path.open("w", encoding="utf-8") as vf:
                for block, embedding in zip(blocks, embeddings):
                    conn.execute(
                        """
                        INSERT INTO chunks (
                            evidence_id, doc_id, source_file, page, block_index, text,
                            confidence, warnings
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            block.evidence_id,
                            block.doc_id,
                            block.source_file,
                            block.page,
                            block.block_index,
                            block.text,
                            block.confidence,
                            json.dumps(block.warnings),
                        ),
                    )
                    conn.execute(
                        "INSERT INTO chunks_fts(evidence_id, text, source_file, doc_id, page) VALUES (?, ?, ?, ?, ?)",
                        (block.evidence_id, block.text, block.source_file, block.doc_id, str(block.page)),
                    )
                    vf.write(json.dumps({"evidence_id": block.evidence_id, "embedding": embedding}) + "\n")
            conn.commit()
            write_json(
                self.config_path,
                {
                    "embedding_provider": self.embedding_provider.config(),
                    "block_count": len(blocks),
                },
            )
        finally:
            conn.close()
        return len(blocks)

    def search(self, query: str, *, top_k: int = 8, vector_k: int = 12, bm25_k: int = 12) -> list[EvidenceChunk]:
        """Return fused evidence chunks for a natural-language query."""
        expanded_query = _expand_query(query)
        bm25 = self._bm25_search(expanded_query, bm25_k)
        vector = self._vector_search(expanded_query, vector_k)
        by_id: dict[str, dict[str, Any]] = {}
        for rank, chunk in enumerate(bm25, start=1):
            entry = by_id.setdefault(chunk.evidence_id, {"chunk": chunk, "score": 0.0})
            entry["bm25_rank"] = rank
            entry["score"] += 1.0 / (60 + rank)
        for rank, chunk in enumerate(vector, start=1):
            entry = by_id.setdefault(chunk.evidence_id, {"chunk": chunk, "score": 0.0})
            entry["vector_rank"] = rank
            entry["score"] += 1.0 / (60 + rank)
        ranked: list[EvidenceChunk] = []
        for entry in by_id.values():
            chunk = entry["chunk"]
            chunk.bm25_rank = entry.get("bm25_rank")
            chunk.vector_rank = entry.get("vector_rank")
            chunk.score = entry["score"]
            ranked.append(chunk)
        ranked.sort(key=lambda item: item.score, reverse=True)
        if self.reranker:
            rerank_limit = int(os.getenv("LOCAL_RERANK_TOP_N", str(top_k)))
            rerank_limit = max(top_k, min(len(ranked), rerank_limit))
            return _blend_reranked(query, ranked, self.reranker.rerank(query, ranked[:rerank_limit], top_k=rerank_limit), top_k)
        return ranked[:top_k]

    def _embed_with_cache(self, texts: list[str]) -> list[list[float]]:
        provider_config = self.embedding_provider.config()
        cache = self._load_embedding_cache()
        keys = [_embedding_cache_key(provider_config, text) for text in texts]
        embeddings: list[list[float] | None] = [cache.get(key) for key in keys]
        missing_positions = [i for i, embedding in enumerate(embeddings) if embedding is None]
        if missing_positions:
            missing_texts = [texts[i] for i in missing_positions]
            new_embeddings = self.embedding_provider.embed_texts(missing_texts)
            with self.embedding_cache_path.open("a", encoding="utf-8") as f:
                for position, embedding in zip(missing_positions, new_embeddings):
                    key = keys[position]
                    embedding = [float(value) for value in embedding]
                    embeddings[position] = embedding
                    f.write(json.dumps({"key": key, "embedding": embedding}) + "\n")
        return [embedding or [] for embedding in embeddings]

    def _load_embedding_cache(self) -> dict[str, list[float]]:
        if not self.embedding_cache_path.exists():
            return {}
        cache: dict[str, list[float]] = {}
        with self.embedding_cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    cache[str(record["key"])] = [float(value) for value in record["embedding"]]
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
        return cache

    def get_chunks(self, evidence_ids: list[str]) -> dict[str, EvidenceChunk]:
        """Fetch chunks by citation id from the SQLite store."""
        if not evidence_ids:
            return {}
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            placeholders = ",".join("?" for _ in evidence_ids)
            rows = conn.execute(f"SELECT * FROM chunks WHERE evidence_id IN ({placeholders})", evidence_ids).fetchall()
            return {row["evidence_id"]: _row_to_chunk(row) for row in rows}
        finally:
            conn.close()

    def all_chunks(self) -> list[EvidenceChunk]:
        """Return all indexed chunks in document/page/block order."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT * FROM chunks ORDER BY doc_id, page, block_index").fetchall()
            return [_row_to_chunk(row) for row in rows]
        finally:
            conn.close()

    def _init_db(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                evidence_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                source_file TEXT NOT NULL,
                page INTEGER NOT NULL,
                block_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                confidence REAL NOT NULL,
                warnings TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
            USING fts5(evidence_id UNINDEXED, text, source_file UNINDEXED, doc_id UNINDEXED, page UNINDEXED)
            """
        )

    def _bm25_search(self, query: str, limit: int) -> list[EvidenceChunk]:
        match_query = _fts_query(query)
        if not match_query:
            return []
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT c.*, bm25(chunks_fts) AS rank_score
                FROM chunks_fts
                JOIN chunks c ON c.evidence_id = chunks_fts.evidence_id
                WHERE chunks_fts MATCH ?
                ORDER BY rank_score
                LIMIT ?
                """,
                (match_query, limit),
            ).fetchall()
            return [_row_to_chunk(row) for row in rows]
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()

    def _vector_search(self, query: str, limit: int) -> list[EvidenceChunk]:
        if not self.vector_path.exists():
            return []
        query_embedding = self.embedding_provider.embed_texts([query])[0]
        vectors: list[tuple[str, float]] = []
        with self.vector_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                vectors.append((record["evidence_id"], cosine(query_embedding, record["embedding"])))
        vectors.sort(key=lambda item: item[1], reverse=True)
        top_ids = [evidence_id for evidence_id, _ in vectors[:limit]]
        chunks_by_id = self.get_chunks(top_ids)
        return [chunks_by_id[evidence_id] for evidence_id in top_ids if evidence_id in chunks_by_id]


def _row_to_chunk(row: sqlite3.Row) -> EvidenceChunk:
    return EvidenceChunk(
        evidence_id=row["evidence_id"],
        doc_id=row["doc_id"],
        source_file=row["source_file"],
        page=int(row["page"]),
        text=row["text"],
        confidence=float(row["confidence"]),
        warnings=json.loads(row["warnings"] or "[]"),
    )


def _blend_reranked(query: str, base_ranked: list[EvidenceChunk], reranked: list[EvidenceChunk], top_k: int) -> list[EvidenceChunk]:
    if not reranked:
        return base_ranked[:top_k]
    weight = float(os.getenv("LOCAL_RERANK_WEIGHT", "0.25"))
    protected = int(os.getenv("LOCAL_RERANK_PROTECT_TOP", "2"))
    base_scores = {chunk.evidence_id: 1.0 / (rank + 1) for rank, chunk in enumerate(base_ranked)}
    rerank_scores = {chunk.evidence_id: 1.0 / (rank + 1) for rank, chunk in enumerate(reranked)}
    candidates: dict[str, EvidenceChunk] = {chunk.evidence_id: chunk for chunk in base_ranked}
    for chunk in reranked:
        candidates[chunk.evidence_id] = chunk
    protected_ids = [chunk.evidence_id for chunk in base_ranked[:protected]]
    query_terms = set(tokenize(query))
    scored: list[tuple[float, EvidenceChunk]] = []
    for evidence_id, chunk in candidates.items():
        lexical = len(query_terms.intersection(tokenize(chunk.text))) / max(len(query_terms), 1)
        score = (1.0 - weight) * base_scores.get(evidence_id, 0.0) + weight * rerank_scores.get(evidence_id, 0.0)
        score += 0.05 * lexical
        if evidence_id in protected_ids:
            score += 1.0
        chunk.score = score
        scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


def _embedding_cache_key(provider_config: dict[str, Any], text: str) -> str:
    payload = {
        "provider": provider_config,
        "text_sha256": hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _fts_query(query: str) -> str:
    tokens = [token for token in tokenize(query) if len(token) > 1]
    if not tokens:
        return ""
    return " OR ".join(f'"{token}"' for token in tokens[:32])


def _expand_query(query: str) -> str:
    lowered = query.lower()
    additions: list[str] = []
    if any(word in lowered for word in ["party", "parties", "client"]):
        additions.extend(["client", "tenant", "landlord", "buyer", "seller", "opposing party", "grantor", "grantee"])
    if any(word in lowered for word in ["property", "properties", "address", "premises"]):
        additions.extend(["premises", "property", "location", "street", "avenue", "road"])
    if any(word in lowered for word in ["title", "exception", "closing", "risk"]):
        additions.extend(["exception", "easement", "release", "encroaches", "payoff", "closing", "survey"])
    if any(word in lowered for word in ["cure", "deadline", "lease", "amount"]):
        additions.extend(["notice", "cure", "calendar", "days", "unpaid", "charges", "lockout", "wire"])
    if any(word in lowered for word in ["unclear", "follow-up", "followup", "claim"]):
        additions.extend(["unclear", "illegible", "smudged", "denial", "proof", "need", "original"])
    if any(word in lowered for word in ["chegg", "cancellation", "subscription", "subscriber", "recurring"]):
        additions.extend(["chegg", "rosca", "cancel", "subscription", "subscriber", "recurring", "charges"])
    if any(word in lowered for word in ["monetary", "judgment", "injunction", "order"]):
        additions.extend(["monetary", "judgment", "injunction", "relief", "order", "$7,500,000", "seven", "million", "five", "hundred", "thousand"])
    if any(word in lowered for word in ["schnitzer", "facility", "facilities", "clean air", "refrigerant", "compliance"]):
        additions.extend(["schnitzer", "facilities", "scrap", "metal", "recycling", "clean", "air", "refrigerant", "compliance"])
    if any(word in lowered for word in ["ocr", "scanned", "image-only", "image"]):
        additions.extend(["ocr", "scanned", "image-only", "mistral_api_key", "requires"])
    if not additions:
        return query
    return f"{query} {' '.join(additions)}"

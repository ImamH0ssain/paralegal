"""Shared data models used across ingestion, retrieval, and generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class DocumentBlock:
    """A normalized text block extracted from one document page.

    Blocks are the durable evidence units in the system. The generated
    ``evidence_id`` is stable for a given document ordering and page/block
    position, making it suitable for citations in memo drafts.
    """

    doc_id: str
    source_file: str
    page: int
    block_index: int
    text: str
    confidence: float = 1.0
    bbox: list[float] | None = None
    warnings: list[str] = field(default_factory=list)
    extraction_method: str = "native"

    @property
    def evidence_id(self) -> str:
        """Return the citation id used in generated drafts and evidence maps."""
        return f"{self.doc_id}:p{self.page}:b{self.block_index}"

    def to_record(self) -> dict[str, Any]:
        """Serialize the block as a JSONL-ready record."""
        record = asdict(self)
        record["record_type"] = "block"
        record["evidence_id"] = self.evidence_id
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "DocumentBlock":
        """Rehydrate a block from a processed JSONL record."""
        return cls(
            doc_id=record["doc_id"],
            source_file=record["source_file"],
            page=int(record["page"]),
            block_index=int(record["block_index"]),
            text=record["text"],
            confidence=float(record.get("confidence", 1.0)),
            bbox=record.get("bbox"),
            warnings=list(record.get("warnings") or []),
            extraction_method=record.get("extraction_method", "native"),
        )


@dataclass(slots=True)
class EvidenceChunk:
    """A retrievable evidence unit returned by the hybrid index."""

    evidence_id: str
    doc_id: str
    source_file: str
    page: int
    text: str
    confidence: float
    warnings: list[str] = field(default_factory=list)
    bm25_rank: int | None = None
    vector_rank: int | None = None
    score: float = 0.0

    def to_record(self) -> dict[str, Any]:
        """Serialize the chunk for UI and sample output files."""
        return asdict(self)

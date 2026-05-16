"""Text normalization, tokenization, citation, and lightweight scoring helpers."""

from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'$.-]*")
CITATION_RE = re.compile(r"\b[A-Z]{2,}\d+:p\d+:b\d+\b")


def normalize_space(text: str) -> str:
    """Collapse repeated whitespace while preserving word order."""
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Return lowercased alphanumeric tokens suitable for retrieval scoring."""
    return [t.lower() for t in TOKEN_RE.findall(text)]


def sentence_split(text: str) -> list[str]:
    """Split normalized text into simple sentence-like spans."""
    parts = re.split(r"(?<=[.!?])\s+", normalize_space(text))
    return [p.strip() for p in parts if p.strip()]


def extract_citations(text: str) -> set[str]:
    """Extract citation ids such as ``DOC2:p6:b1`` from text."""
    return set(CITATION_RE.findall(text))


def similarity(a: str, b: str) -> float:
    """Return a rough character-level similarity score between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def keywords(text: str, limit: int = 12) -> list[str]:
    """Return the most common non-stopword tokens for prompt/context matching."""
    stop = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "was",
        "were",
        "are",
        "has",
        "have",
        "had",
        "not",
        "but",
        "into",
        "over",
        "under",
        "client",
        "document",
    }
    counts = Counter(t for t in tokenize(text) if len(t) > 2 and t not in stop)
    return [word for word, _ in counts.most_common(limit)]

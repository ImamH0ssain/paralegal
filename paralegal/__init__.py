"""Paralegal's evidence-grounded legal memo RAG package.

The package is intentionally small and file-oriented: ingestion produces
normalized evidence records, retrieval builds a local hybrid index, generation
drafts cited memos from retrieved evidence only, and feedback captures operator
edits for future prompt context.
"""

__all__ = [
    "__version__",
]

__version__ = "0.1.0"

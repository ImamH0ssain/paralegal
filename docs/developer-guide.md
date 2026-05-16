# Developer Guide

This guide documents the code layout and runtime boundaries for Paralegal.

## Pipeline

1. Ingestion expands input paths, extracts text blocks, performs OCR when needed, and writes `processed_documents.jsonl`.
2. Structured extraction summarizes parties, dates, addresses, amounts, signatures, deadlines, and unclear spans per document.
3. Retrieval builds a local SQLite FTS5 index plus vector artifacts, then combines sparse and dense rankings with reciprocal-rank fusion.
4. Generation retrieves evidence, adds structured-field support chunks, drafts a memo, sanitizes citations, validates grounding, and writes sample outputs.
5. Feedback captures operator edits, classifies reusable changes, and writes learned patterns for future prompts.
6. Evaluation checks retrieval quality, extraction output, memo grounding, and edit-learning persistence.

## Module Map

| Module | Responsibility |
| --- | --- |
| `app.py` | Streamlit UI for document preparation, indexing, drafting, evidence review, feedback, and evaluation. |
| `paralegal.cli` | Command-line orchestration for individual stages and the complete `all` workflow. |
| `paralegal.ingestion` | Native PDF extraction, OCR routing, text normalization, extraction caching, and structured fields. |
| `paralegal.retrieval` | SQLite FTS5, embedding cache, vector search, reciprocal-rank fusion, and optional reranking. |
| `paralegal.generation` | Evidence selection, memo drafting, local/API model routing, citation sanitization, and validation. |
| `paralegal.feedback` | Operator edit storage, edit classification, learned pattern aggregation, and prompt-memory loading. |
| `paralegal.evaluation` | Retrieval, extraction, citation-grounding, and edit-learning report generation. |
| `paralegal.providers` | OpenAI, Ollama, Mistral OCR, embedding, and reranker adapters. |
| `paralegal.local_device` | CPU/CUDA selection for WSL local model runs. |
| `paralegal.models` | Shared `DocumentBlock` and `EvidenceChunk` data models. |
| `paralegal.real_data` | Public sample document download manifest and ground-truth queries. |
| `paralegal.sample_data` | Synthetic legal-style PDF generation for offline runs and tests. |
| `paralegal.text_utils` | Tokenization, citation extraction, sentence splitting, and lightweight text scoring. |

## Data Boundaries

Generated runtime artifacts are intentionally kept out of Git:

- `data/processed/`
- `data/feedback/`
- `data/uploads/`
- `evidence_index/`
- `sample_inputs/`
- local model caches and virtual environments

The repository keeps source code, tests, documentation, screenshots, setup files, and representative sample outputs. A fresh clone can regenerate inputs and indexes with the CLI.

## Provider Strategy

Provider classes expose small, synchronous methods so the rest of the pipeline does not depend on provider-specific request formats. The default path is safe for no-key runs:

- OpenAI generation and embeddings are used only when keys are configured and the caller enables them.
- Mistral OCR is used only when a key is configured and OCR routing selects it.
- Ollama is local and opt-in through `--use-ollama`.
- Hashing embeddings and deterministic memo generation keep tests and demos reproducible.

## Grounding Rules

Memo generation is constrained by these invariants:

- Claims must cite evidence ids from the retrieved evidence bundle.
- Unsupported facts should appear as open questions rather than assertions.
- LLM citations are sanitized before validation.
- Invalid citations, uncited factual bullets, missing sections, weak support, or likely truncation fail validation.
- Ollama drafts can fall back to the deterministic generator when validation fails.

## Local Hardware Policy

`scripts/run_wsl_local.sh` calls `python -m paralegal.local_device` before running local OCR, embedding, and reranking workloads. Auto mode enables CUDA only when PyTorch reports a supported GPU at or above `LOCAL_CUDA_MIN_CAPABILITY`. Unsupported visible GPUs are hidden from PyTorch workloads by setting `CUDA_VISIBLE_DEVICES=""`.

Override options:

- `LOCAL_ACCELERATOR=cpu`
- `LOCAL_ACCELERATOR=cuda`
- `LOCAL_CUDA_MIN_CAPABILITY=75`
- `LOCAL_OCR_DEVICE=cpu|cuda`
- `LOCAL_EMBEDDING_DEVICE=cpu|cuda`
- `LOCAL_RERANK_DEVICE=cpu|cuda`

## Verification

Run the fast test suite:

```powershell
venv\Scripts\python.exe -m pytest
```

Run the end-to-end local baseline:

```powershell
venv\Scripts\paralegal.exe all
```

Run the WSL local-model stack:

```bash
bash scripts/run_wsl_local.sh
```

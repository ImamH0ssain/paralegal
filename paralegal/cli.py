"""Command-line interface for the document RAG pipeline."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sqlite3
import sys
from pathlib import Path

from .evaluation import run_evaluation
from .feedback import save_operator_feedback
from .generation import DEFAULT_MEMO_QUERY, generate_memo
from .ingestion import ingest_paths
from .local_device import resolve_local_devices
from .providers import OllamaClient, OpenAIClient, make_embedding_provider, make_reranker
from .real_data import download_real_inputs
from .retrieval import EvidenceIndex
from .sample_data import create_sample_inputs


def main() -> None:
    """Parse CLI arguments and run the requested pipeline stage."""
    _load_dotenv(Path(".env"))
    parser = argparse.ArgumentParser(description="Evidence-grounded legal-document RAG demo")
    sub = parser.add_subparsers(dest="command", required=True)

    samples = sub.add_parser("make-samples", help="Create synthetic legal-style PDF sample inputs")
    samples.add_argument("--out", default="sample_inputs")

    sub.add_parser("doctor", help="Check local dependencies, API-key presence, and SQLite FTS5 support")

    real = sub.add_parser("download-real", help="Download public FTC/DOJ/SEC documents for the demo")
    real.add_argument("--out", default="sample_inputs")
    real.add_argument("--overwrite", action="store_true")

    ingest = sub.add_parser("ingest", help="Extract document blocks and structured fields")
    ingest.add_argument("inputs", nargs="+")
    ingest.add_argument("--out", default="data/processed")
    ingest.add_argument("--force-ocr", action="store_true")
    ingest.add_argument("--ocr-backend", choices=["auto", "mistral", "docling", "none"], default="auto")
    ingest.add_argument("--ingest-workers", type=int, default=None)
    ingest.add_argument("--no-mistral", action="store_true")
    ingest.add_argument("--no-openai", action="store_true")

    build = sub.add_parser("index", help="Build the hybrid evidence index")
    build.add_argument("--processed", default="data/processed/processed_documents.jsonl")
    build.add_argument("--index", default="evidence_index")
    build.add_argument("--embedding-backend", choices=["auto", "openai", "local", "hashing"], default="auto")

    draft = sub.add_parser("draft", help="Generate a grounded internal memo")
    draft.add_argument("--index", default="evidence_index")
    draft.add_argument("--processed", default="data/processed")
    draft.add_argument("--feedback", default="data/feedback")
    draft.add_argument("--out", default="sample_outputs")
    draft.add_argument("--query", default=DEFAULT_MEMO_QUERY)
    draft.add_argument("--no-openai", action="store_true")
    draft.add_argument("--use-ollama", action="store_true")
    draft.add_argument("--reranker-backend", choices=["none", "local"], default="none")

    fb = sub.add_parser("feedback", help="Save an operator edit and update learned patterns")
    fb.add_argument("--feedback", default="data/feedback")
    fb.add_argument("--original-file", required=True)
    fb.add_argument("--edited-file", required=True)
    fb.add_argument("--evidence-id", action="append", default=[])
    fb.add_argument("--task-type", default="internal_case_memo")
    fb.add_argument("--no-openai", action="store_true")

    evaluate = sub.add_parser("evaluate", help="Run retrieval, grounding, extraction, and edit-loop checks")
    evaluate.add_argument("--index", default="evidence_index")
    evaluate.add_argument("--processed", default="data/processed")
    evaluate.add_argument("--truth", default="sample_inputs/sample_ground_truth.json")
    evaluate.add_argument("--feedback", default="data/feedback")
    evaluate.add_argument("--out", default="sample_outputs")
    evaluate.add_argument("--use-openai", action="store_true")
    evaluate.add_argument("--use-ollama", action="store_true")
    evaluate.add_argument("--reranker-backend", choices=["none", "local"], default="none")

    all_cmd = sub.add_parser("all", help="Create samples, ingest, index, draft, and evaluate")
    all_cmd.add_argument("--sample-dir", default="sample_inputs")
    all_cmd.add_argument("--synthetic", action="store_true", help="Use generated synthetic documents instead of public real documents")
    all_cmd.add_argument("--overwrite-real", action="store_true")
    all_cmd.add_argument("--processed", default="data/processed")
    all_cmd.add_argument("--index", default="evidence_index")
    all_cmd.add_argument("--feedback", default="data/feedback")
    all_cmd.add_argument("--out", default="sample_outputs")
    all_cmd.add_argument("--use-openai", action="store_true")
    all_cmd.add_argument("--use-ollama", action="store_true")
    all_cmd.add_argument("--ocr-backend", choices=["auto", "mistral", "docling", "none"], default="auto")
    all_cmd.add_argument("--ingest-workers", type=int, default=None)
    all_cmd.add_argument("--embedding-backend", choices=["auto", "openai", "local", "hashing"], default="auto")
    all_cmd.add_argument("--reranker-backend", choices=["none", "local"], default="none")

    args = parser.parse_args()

    if args.command == "doctor":
        _doctor()
    elif args.command == "make-samples":
        paths = create_sample_inputs(args.out)
        print(f"Created {len(paths)} sample PDFs in {args.out}")
    elif args.command == "download-real":
        paths = download_real_inputs(args.out, overwrite=args.overwrite)
        print(f"Downloaded/found {len(paths)} real public input file(s) in {args.out}")
    elif args.command == "ingest":
        records = ingest_paths(
            args.inputs,
            args.out,
            force_ocr=args.force_ocr,
            ocr_backend=args.ocr_backend,
            use_mistral=not args.no_mistral,
            use_openai=not args.no_openai,
            workers=args.ingest_workers,
        )
        print(f"Wrote {len(records)} processed records to {Path(args.out) / 'processed_documents.jsonl'}")
    elif args.command == "index":
        count = EvidenceIndex(args.index, embedding_provider=make_embedding_provider(args.embedding_backend)).build(args.processed)
        print(f"Indexed {count} evidence blocks in {args.index}")
    elif args.command == "draft":
        result = generate_memo(
            args.index,
            args.processed,
            query=args.query,
            feedback_dir=args.feedback,
            output_dir=args.out,
            use_openai=not args.no_openai,
            use_ollama=args.use_ollama,
            reranker=make_reranker(args.reranker_backend),
        )
        print(f"Wrote memo to {Path(args.out) / 'generated_memo.md'}")
        for warning in result.get("generation_warnings", []):
            print(f"Generation warning: {warning}")
        print(f"Citation validation passed: {result['validation']['valid']}")
    elif args.command == "feedback":
        original = Path(args.original_file).read_text(encoding="utf-8")
        edited = Path(args.edited_file).read_text(encoding="utf-8")
        result = save_operator_feedback(
            args.feedback,
            task_type=args.task_type,
            original_draft=original,
            edited_draft=edited,
            evidence_ids=args.evidence_id,
            openai=OpenAIClient() if not args.no_openai else None,
        )
        print(f"Stored feedback. Learned patterns: {len(result['learned_patterns'].get('patterns', []))}")
    elif args.command == "evaluate":
        report = run_evaluation(
            index_dir=args.index,
            processed_dir=args.processed,
            ground_truth_path=args.truth,
            feedback_dir=args.feedback,
            output_dir=args.out,
            use_openai=args.use_openai,
            use_ollama=args.use_ollama,
            reranker=make_reranker(args.reranker_backend),
        )
        print(f"Wrote evaluation report to {Path(args.out) / 'evaluation_report.md'}")
        print(f"Mean term recall@5: {report['retrieval']['mean_term_recall_at_5']:.2f}")
    elif args.command == "all":
        if args.synthetic:
            paths = create_sample_inputs(args.sample_dir)
        else:
            paths = download_real_inputs(args.sample_dir, overwrite=args.overwrite_real)
            if not paths:
                print("Real-data download failed; falling back to generated synthetic samples.")
                paths = create_sample_inputs(args.sample_dir)
        records = ingest_paths(
            paths,
            args.processed,
            use_openai=args.use_openai,
            ocr_backend=args.ocr_backend,
            workers=args.ingest_workers,
        )
        embedding_provider = make_embedding_provider(args.embedding_backend)
        reranker = make_reranker(args.reranker_backend)
        count = EvidenceIndex(args.index, embedding_provider=embedding_provider).build(Path(args.processed) / "processed_documents.jsonl")
        draft_result = generate_memo(
            args.index,
            args.processed,
            feedback_dir=args.feedback,
            output_dir=args.out,
            use_openai=args.use_openai,
            use_ollama=args.use_ollama,
            reranker=reranker,
        )
        report = run_evaluation(
            index_dir=args.index,
            processed_dir=args.processed,
            ground_truth_path=Path(args.sample_dir) / "sample_ground_truth.json",
            feedback_dir=args.feedback,
            output_dir=args.out,
            use_openai=args.use_openai,
            use_ollama=args.use_ollama,
            reranker=reranker,
            draft_result=draft_result,
        )
        print(f"Created {len(paths)} samples, {len(records)} records, and {count} indexed blocks.")
        for warning in draft_result.get("generation_warnings", []):
            print(f"Generation warning: {warning}")
        print(f"Memo validation passed: {draft_result['validation']['valid']}")
        print(f"Mean term recall@5: {report['retrieval']['mean_term_recall_at_5']:.2f}")

def _doctor() -> None:
    print(f"Python: {sys.version.split()[0]}")
    for module in ["pypdf", "reportlab", "streamlit", "pytest"]:
        print(f"{module}: {'ok' if importlib.util.find_spec(module) else 'missing'}")
    print(f"OPENAI_API_KEY: {'set' if os.getenv('OPENAI_API_KEY') else 'not set'}")
    print(f"MISTRAL_API_KEY: {'set' if os.getenv('MISTRAL_API_KEY') else 'not set'}")
    ollama = OllamaClient()
    print(f"OLLAMA_BASE_URL: {ollama.base_url}")
    print(f"OLLAMA_MODEL: {ollama.model}")
    print(f"OLLAMA_NUM_THREAD: {os.getenv('OLLAMA_NUM_THREAD', 'default')}")
    print(f"OLLAMA_NUM_CTX: {os.getenv('OLLAMA_NUM_CTX', 'default')}")
    print(f"Ollama reachable: {'yes' if ollama.available else 'no'}")
    for module in ["sentence_transformers", "torch", "docling"]:
        print(f"{module}: {'ok' if importlib.util.find_spec(module) else 'missing'}")
    print(f"LOCAL_EMBEDDING_MODEL: {os.getenv('LOCAL_EMBEDDING_MODEL', 'Qwen/Qwen3-Embedding-0.6B')}")
    print(f"LOCAL_RERANK_MODEL: {os.getenv('LOCAL_RERANK_MODEL', 'BAAI/bge-reranker-base')}")
    local_devices = resolve_local_devices()
    print(f"LOCAL_ACCELERATOR_RESOLVED: {local_devices.accelerator} ({local_devices.reason})")
    print(f"LOCAL_OCR_DEVICE: {local_devices.ocr_device}")
    print(f"LOCAL_EMBEDDING_DEVICE: {local_devices.embedding_device}")
    print(f"LOCAL_RERANK_DEVICE: {local_devices.rerank_device}")
    print(f"OPENAI_TEXT_MODEL_FALLBACKS: {os.getenv('OPENAI_TEXT_MODEL_FALLBACKS', 'gpt-5.5,gpt-5.4,gpt-5.4-mini')}")
    print(f"OPENAI_JSON_MODEL_FALLBACKS: {os.getenv('OPENAI_JSON_MODEL_FALLBACKS', 'gpt-5.4-mini,gpt-5.4,gpt-5.5')}")
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE VIRTUAL TABLE test_fts USING fts5(text)")
        print("SQLite FTS5: ok")
    except sqlite3.OperationalError as exc:
        print(f"SQLite FTS5: missing ({exc})")
    finally:
        conn.close()


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


if __name__ == "__main__":
    main()

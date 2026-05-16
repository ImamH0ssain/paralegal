"""Evaluation routines for retrieval, extraction, grounding, and edit learning."""

from __future__ import annotations

from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from typing import Any

from .feedback import save_operator_feedback
from .generation import DEFAULT_MEMO_QUERY, generate_memo
from .io_utils import read_json, read_jsonl, write_json
from .providers import OpenAIClient
from .retrieval import EvidenceIndex


def run_evaluation(
    *,
    index_dir: str | Path,
    processed_dir: str | Path,
    ground_truth_path: str | Path,
    feedback_dir: str | Path,
    output_dir: str | Path,
    use_openai: bool = False,
    use_ollama: bool = False,
    reranker: Any | None = None,
    draft_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the built-in evaluation suite and write JSON/Markdown reports."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    index = EvidenceIndex(index_dir, reranker=reranker)
    ground_truth = read_json(ground_truth_path, default={"queries": []}) or {"queries": []}
    retrieval_results = [_evaluate_query(index, item) for item in ground_truth.get("queries", [])]
    processed_records = read_jsonl(Path(processed_dir) / "processed_documents.jsonl")
    extraction_result = _evaluate_extraction(processed_records)
    if draft_result is None:
        draft_result = generate_memo(
            index_dir,
            processed_dir,
            query=DEFAULT_MEMO_QUERY,
            feedback_dir=feedback_dir,
            output_dir=out,
            use_openai=use_openai,
            use_ollama=use_ollama,
            reranker=reranker,
        )
    with TemporaryDirectory(prefix="paralegal_eval_feedback_") as temp_feedback_dir:
        learning_result = _simulate_edit_learning(temp_feedback_dir, draft_result)
    report = {
        "retrieval": {
            "queries": retrieval_results,
            "mean_term_recall_at_5": _safe_mean([r["term_recall_at_5"] for r in retrieval_results]),
            "mean_source_recall_at_5": _safe_mean([r["source_recall_at_5"] for r in retrieval_results]),
            "mean_reciprocal_rank": _safe_mean([r["mrr"] for r in retrieval_results]),
        },
        "extraction": extraction_result,
        "draft_grounding": draft_result["validation"],
        "edit_learning": learning_result,
        "notes": [
            "Retrieval metrics are term-based because sample evidence IDs depend on PDF extraction block boundaries.",
            "Source recall checks whether the expected public document surfaced in the top five.",
            "The no-key baseline uses deterministic hashing embeddings; local Qwen or OpenAI embeddings improve semantic retrieval when configured.",
        ],
    }
    write_json(out / "evaluation_report.json", report)
    (out / "evaluation_report.md").write_text(_format_report(report), encoding="utf-8")
    return report


def _evaluate_query(index: EvidenceIndex, item: dict[str, Any]) -> dict[str, Any]:
    query = item["query"]
    expected_terms = item.get("expected_terms", [])
    expected_sources = item.get("expected_source_files", [])
    results = index.search(query, top_k=5)
    term_ranks: dict[str, int | None] = {}
    for term in expected_terms:
        lowered = term.lower()
        rank = None
        for i, chunk in enumerate(results, start=1):
            if lowered in chunk.text.lower():
                rank = i
                break
        term_ranks[term] = rank
    found = [rank for rank in term_ranks.values() if rank is not None]
    retrieved_sources = [chunk.source_file for chunk in results]
    matched_sources = sorted({source for source in expected_sources if source in retrieved_sources})
    return {
        "query": query,
        "expected_terms": expected_terms,
        "expected_source_files": expected_sources,
        "retrieved_evidence_ids": [chunk.evidence_id for chunk in results],
        "retrieved_source_files": retrieved_sources,
        "term_ranks": term_ranks,
        "term_recall_at_5": len(found) / max(len(expected_terms), 1),
        "source_recall_at_5": len(matched_sources) / max(len(expected_sources), 1) if expected_sources else 0.0,
        "mrr": _safe_mean([1.0 / rank for rank in found]),
    }


def _evaluate_extraction(records: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [r for r in records if r.get("record_type") == "document_summary"]
    blocks = [r for r in records if r.get("record_type") == "block"]
    field_counts: dict[str, int] = {}
    for summary in summaries:
        fields = summary.get("structured_fields", {})
        for key in ["parties", "dates", "addresses", "amounts", "signatures", "deadlines", "unclear_spans"]:
            field_counts[key] = field_counts.get(key, 0) + len(fields.get(key, []) or [])
    warning_count = sum(len(block.get("warnings", []) or []) for block in blocks)
    return {
        "document_count": len(summaries),
        "block_count": len(blocks),
        "field_counts": field_counts,
        "warning_count": warning_count,
        "usable_downstream": len(blocks) > 0 and field_counts.get("parties", 0) > 0,
    }


def _simulate_edit_learning(feedback_dir: str | Path, draft_result: dict[str, Any]) -> dict[str, Any]:
    memo = draft_result["memo"]
    evidence_ids = [item["evidence_id"] for item in draft_result.get("evidence", [])]
    first_evidence = evidence_ids[0] if evidence_ids else "DOC1:p1:b1"
    original = memo + "\n\n- The client likely has a strong merits position.\n"
    edited = memo + f"\n\n- Do not assess merits without source-backed legal analysis. [{first_evidence}]\n"
    saved = save_operator_feedback(
        feedback_dir,
        task_type="internal_case_memo",
        original_draft=original,
        edited_draft=edited,
        evidence_ids=evidence_ids,
        openai=OpenAIClient(),
    )
    return {
        "captured": True,
        "learned_patterns": saved["learned_patterns"].get("patterns", [])[:5],
        "stored_feedback_count": len(read_jsonl(Path(feedback_dir) / "feedback.jsonl")),
    }


def _safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _format_report(report: dict[str, Any]) -> str:
    lines = [
        "# Evaluation Report",
        "",
        "## Retrieval",
        f"- Mean term recall@5: {report['retrieval']['mean_term_recall_at_5']:.2f}",
        f"- Mean source recall@5: {report['retrieval']['mean_source_recall_at_5']:.2f}",
        f"- Mean reciprocal rank: {report['retrieval']['mean_reciprocal_rank']:.2f}",
        "",
        "## Extraction",
        f"- Documents processed: {report['extraction']['document_count']}",
        f"- Blocks extracted: {report['extraction']['block_count']}",
        f"- Field counts: {report['extraction']['field_counts']}",
        f"- Extraction warnings: {report['extraction']['warning_count']}",
        "",
        "## Draft Grounding",
        f"- Citation validation passed: {report['draft_grounding']['valid']}",
        f"- Invalid citations: {report['draft_grounding']['invalid_citations']}",
        f"- Uncited factual lines: {len(report['draft_grounding']['uncited_factual_lines'])}",
        f"- Weakly supported cited lines: {len(report['draft_grounding'].get('weakly_supported_lines', []))}",
        "",
        "## Edit Learning",
        f"- Feedback captured: {report['edit_learning']['captured']}",
        f"- Stored feedback count: {report['edit_learning']['stored_feedback_count']}",
    ]
    for item in report["edit_learning"]["learned_patterns"]:
        lines.append(f"- Learned: {item.get('pattern', item)}")
    lines.append("")
    return "\n".join(lines)

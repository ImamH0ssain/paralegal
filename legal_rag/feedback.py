"""Operator edit capture and lightweight preference learning."""

from __future__ import annotations

import difflib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import ensure_dir, read_json, read_jsonl, write_json
from .providers import OpenAIClient
from .text_utils import extract_citations, keywords, normalize_space


def save_operator_feedback(
    feedback_dir: str | Path,
    *,
    task_type: str,
    original_draft: str,
    edited_draft: str,
    evidence_ids: list[str],
    openai: OpenAIClient | None = None,
) -> dict[str, Any]:
    """Persist an operator edit and refresh the learned pattern summary.

    The saved record keeps the original draft, edited draft, evidence ids that
    were visible to the operator, and a structured analysis of reusable editing
    preferences. Future draft prompts consume the aggregate pattern file.
    """

    directory = ensure_dir(feedback_dir)
    analysis = classify_edit(original_draft, edited_draft, openai=openai)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task_type": task_type,
        "original_draft": original_draft,
        "edited_draft": edited_draft,
        "evidence_ids": evidence_ids,
        "analysis": analysis,
    }
    feedback_path = directory / "feedback.jsonl"
    with feedback_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    learned = update_learned_patterns(directory)
    return {"feedback_record": record, "learned_patterns": learned}


def classify_edit(original: str, edited: str, *, openai: OpenAIClient | None = None) -> dict[str, Any]:
    """Classify reusable differences between an original and edited memo."""
    if openai and openai.available:
        try:
            schema = {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "patterns": {"type": "array", "items": {"type": "string"}},
                    "removed_unsupported_claims": {"type": "array", "items": {"type": "string"}},
                    "added_missing_facts": {"type": "array", "items": {"type": "string"}},
                    "tone_preferences": {"type": "array", "items": {"type": "string"}},
                    "citation_preferences": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "patterns",
                    "removed_unsupported_claims",
                    "added_missing_facts",
                    "tone_preferences",
                    "citation_preferences",
                ],
            }
            return openai.generate_json(
                "Classify reusable editing patterns for future grounded legal memos.",
                f"ORIGINAL:\n{original}\n\nEDITED:\n{edited}",
                schema,
            )
        except Exception as exc:
            heuristic = _classify_edit_heuristic(original, edited)
            heuristic["patterns"].append(f"LLM edit classifier failed; heuristic fallback used: {exc}")
            return heuristic
    return _classify_edit_heuristic(original, edited)


def update_learned_patterns(feedback_dir: str | Path) -> dict[str, Any]:
    """Aggregate all feedback records into a compact prompt-memory file."""
    directory = ensure_dir(feedback_dir)
    records = read_jsonl(directory / "feedback.jsonl")
    pattern_counts: dict[str, int] = {}
    examples: list[dict[str, Any]] = []
    for record in records:
        analysis = record.get("analysis", {})
        for pattern in analysis.get("patterns", []):
            key = normalize_space(pattern)
            pattern_counts[key] = pattern_counts.get(key, 0) + 1
        examples.append(
            {
                "task_type": record.get("task_type", "memo"),
                "before_excerpt": record.get("original_draft", "")[:800],
                "after_excerpt": record.get("edited_draft", "")[:800],
                "patterns": analysis.get("patterns", [])[:5],
            }
        )
    learned = {
        "patterns": [
            {"pattern": pattern, "count": count}
            for pattern, count in sorted(pattern_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "examples": examples[-3:],
    }
    write_json(directory / "learned_patterns.json", learned)
    return learned


def load_learning_context(feedback_dir: str | Path, query: str = "") -> str:
    """Return a prompt-ready summary of learned edit patterns and examples."""
    learned = read_json(Path(feedback_dir) / "learned_patterns.json", default={}) or {}
    lines: list[str] = []
    for item in learned.get("patterns", [])[:6]:
        lines.append(f"- {item['pattern']}")
    query_terms = set(keywords(query, limit=8))
    examples = learned.get("examples", [])
    if examples:
        lines.append("\nRelevant prior edit examples:")
        scored = []
        for example in examples:
            text = " ".join(example.get("patterns", [])) + " " + example.get("after_excerpt", "")
            score = len(query_terms.intersection(keywords(text, limit=20)))
            scored.append((score, example))
        for _, example in sorted(scored, key=lambda item: item[0], reverse=True)[:2]:
            lines.append(f"Example before: {normalize_space(example.get('before_excerpt', ''))[:240]}")
            lines.append(f"Example after: {normalize_space(example.get('after_excerpt', ''))[:240]}")
    return "\n".join(lines).strip()


def _classify_edit_heuristic(original: str, edited: str) -> dict[str, Any]:
    diff = list(difflib.ndiff(original.splitlines(), edited.splitlines()))
    removed = [line[2:].strip() for line in diff if line.startswith("- ") and line[2:].strip()]
    added = [line[2:].strip() for line in diff if line.startswith("+ ") and line[2:].strip()]
    patterns: list[str] = []
    removed_unsupported: list[str] = []
    added_missing: list[str] = []
    citation_preferences: list[str] = []
    tone_preferences: list[str] = []
    for line in removed:
        if extract_citations(line) == set() and len(line) > 60:
            removed_unsupported.append(line[:240])
    for line in added:
        if extract_citations(line):
            added_missing.append(line[:240])
    if len(edited) < len(original) * 0.85:
        patterns.append("Prefer tighter memos with fewer speculative paragraphs.")
        tone_preferences.append("Concise, review-ready prose.")
    if "Open Questions" in edited and "Open Questions" not in original:
        patterns.append("Surface missing or uncertain facts in an Open Questions section.")
    if len(extract_citations(edited)) > len(extract_citations(original)):
        patterns.append("Use more explicit source citations on factual claims.")
        citation_preferences.append("Add citations to each factual bullet.")
    if removed_unsupported:
        patterns.append("Remove claims that do not carry source citations.")
    if added_missing:
        patterns.append("Carry operator-added cited facts into future drafts when similar evidence appears.")
    if not patterns:
        patterns.append("Preserve operator section wording and citation density in future drafts.")
    return {
        "patterns": patterns,
        "removed_unsupported_claims": removed_unsupported[:5],
        "added_missing_facts": added_missing[:5],
        "tone_preferences": tone_preferences,
        "citation_preferences": citation_preferences,
    }

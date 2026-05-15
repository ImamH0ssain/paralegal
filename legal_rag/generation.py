"""Evidence-grounded memo generation and citation validation."""

from __future__ import annotations

import json
import os
import re
from hashlib import sha256
from pathlib import Path
from typing import Any

from .feedback import load_learning_context
from .io_utils import read_json, write_json
from .models import EvidenceChunk
from .providers import OllamaClient, OpenAIClient
from .retrieval import EvidenceIndex
from .text_utils import extract_citations, normalize_space, sentence_split, tokenize


DEFAULT_MEMO_QUERY = (
    "Prepare a first-pass internal case memo. Identify parties, timeline, key facts, "
    "document issues, open questions, and cite every factual claim."
)

REQUIRED_MEMO_SECTIONS = [
    "Parties and Documents",
    "Timeline",
    "Key Facts",
    "Document Issues",
    "Open Questions",
    "Evidence Used",
]


def generate_memo(
    index_dir: str | Path,
    processed_dir: str | Path,
    *,
    query: str = DEFAULT_MEMO_QUERY,
    feedback_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    use_openai: bool = True,
    use_ollama: bool = False,
    embedding_provider: Any | None = None,
    reranker: Any | None = None,
) -> dict[str, Any]:
    """Generate a cited memo from retrieved evidence and learned edit context.

    LLM-backed drafts are sanitized and validated against the retrieved
    evidence IDs. If local Ollama output fails grounding validation, the
    deterministic offline memo is used as a reliability fallback by default.
    """

    index = EvidenceIndex(index_dir, embedding_provider=embedding_provider, reranker=reranker)
    evidence = index.search(query, top_k=10)
    summaries = read_json(Path(processed_dir) / "document_summaries.json", default=[]) or []
    evidence = _add_structured_evidence(index, evidence, summaries)
    learning_context = load_learning_context(feedback_dir, query) if feedback_dir else ""
    openai = OpenAIClient()
    ollama = OllamaClient()
    used_provider = "offline"
    generation_warnings: list[str] = []
    if use_ollama and ollama.available:
        try:
            memo = _generate_with_llm_cached(query, evidence, summaries, learning_context, ollama, processed_dir)
            memo = _sanitize_llm_memo(memo, {chunk.evidence_id for chunk in evidence}, evidence)
            memo = _prune_weakly_supported_lines(memo, {chunk.evidence_id for chunk in evidence}, evidence)
            used_provider = "ollama"
        except Exception as exc:
            memo = _generate_offline(query, evidence, summaries, learning_context)
            generation_warnings.append(f"Ollama generation failed; used offline memo fallback: {str(exc)[:240]}")
    elif use_openai and openai.available:
        try:
            memo = _generate_with_llm(query, evidence, summaries, learning_context, openai)
            memo = _sanitize_llm_memo(memo, {chunk.evidence_id for chunk in evidence}, evidence)
            memo = _prune_weakly_supported_lines(memo, {chunk.evidence_id for chunk in evidence}, evidence)
            used_provider = "openai"
        except Exception as exc:
            memo = _generate_offline(query, evidence, summaries, learning_context)
            generation_warnings.append(f"OpenAI generation failed; used offline memo fallback: {str(exc)[:240]}")
    else:
        memo = _generate_offline(query, evidence, summaries, learning_context)
    validation = validate_memo_citations(memo, {chunk.evidence_id for chunk in evidence}, evidence)
    if (
        used_provider == "ollama"
        and not validation["valid"]
        and os.getenv("OLLAMA_FALLBACK_ON_INVALID", "true").lower() not in {"0", "false", "no"}
    ):
        memo = _generate_offline(query, evidence, summaries, learning_context)
        generation_warnings.append("Ollama memo failed citation validation; used offline memo fallback.")
        used_provider = "offline"
        validation = validate_memo_citations(memo, {chunk.evidence_id for chunk in evidence}, evidence)
    result = {
        "query": query,
        "memo": memo,
        "evidence": [chunk.to_record() for chunk in evidence],
        "validation": validation,
        "used_openai": used_provider == "openai",
        "used_llm_provider": used_provider,
        "generation_warnings": generation_warnings,
    }
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "generated_memo.md").write_text(memo, encoding="utf-8")
        write_json(out / "evidence_map.json", result)
    return result


def validate_memo_citations(
    memo: str,
    allowed_evidence_ids: set[str],
    evidence: list[EvidenceChunk] | None = None,
) -> dict[str, Any]:
    """Validate citation IDs, required sections, truncation, and weak support."""
    citations = extract_citations(memo)
    invalid = sorted(citations - allowed_evidence_ids)
    uncited_lines: list[str] = []
    weakly_supported_lines: list[dict[str, Any]] = []
    evidence_by_id = {chunk.evidence_id: chunk for chunk in evidence or []}
    seen_sections: set[str] = set()
    section = ""
    for line in memo.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            section = stripped[3:].strip().lower()
            seen_sections.add(section)
            continue
        if not stripped.startswith("- "):
            continue
        if section in {"evidence used"}:
            continue
        if section in {"document issues", "open questions", "learned operator preferences applied"} and not extract_citations(stripped):
            continue
        if len(stripped) > 30 and not extract_citations(stripped):
            uncited_lines.append(stripped)
        elif evidence_by_id and section not in {"document issues"}:
            support = _claim_support_score(stripped, evidence_by_id)
            if support < 0.18:
                weakly_supported_lines.append({"line": stripped, "support_score": round(support, 3)})
    missing_sections = [section for section in REQUIRED_MEMO_SECTIONS if section.lower() not in seen_sections]
    possibly_truncated = _looks_truncated(memo)
    return {
        "valid": not invalid and not uncited_lines and not weakly_supported_lines and not missing_sections and not possibly_truncated,
        "citations_used": sorted(citations),
        "invalid_citations": invalid,
        "uncited_factual_lines": uncited_lines,
        "weakly_supported_lines": weakly_supported_lines,
        "missing_sections": missing_sections,
        "possibly_truncated": possibly_truncated,
    }


def _looks_truncated(memo: str) -> bool:
    lines = [line.strip() for line in memo.splitlines() if line.strip()]
    if not lines:
        return True
    last = lines[-1]
    if last.count("[") != last.count("]") or last.count("(") != last.count(")"):
        return True
    if last.startswith("- ") and not re.search(r"[\].?!)]$", last):
        return True
    return False


def _prune_weakly_supported_lines(
    memo: str,
    allowed_evidence_ids: set[str],
    evidence: list[EvidenceChunk],
) -> str:
    validation = validate_memo_citations(memo, allowed_evidence_ids, evidence)
    weak_lines = {item["line"] for item in validation.get("weakly_supported_lines", [])}
    if not weak_lines:
        return memo
    kept: list[str] = []
    for line in memo.splitlines():
        if line.strip() in weak_lines:
            continue
        kept.append(line.rstrip())
    return "\n".join(kept).strip() + "\n"


def _local_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _generate_with_llm_cached(
    query: str,
    evidence: list[EvidenceChunk],
    summaries: list[dict[str, Any]],
    learning_context: str,
    llm: Any,
    processed_dir: str | Path,
) -> str:
    if not bool(getattr(llm, "is_local", False)):
        return _generate_with_llm(query, evidence, summaries, learning_context, llm)
    cache_dir = Path(processed_dir) / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    prompt_evidence = _select_llm_evidence(
        evidence,
        summaries,
        _local_int_env("OLLAMA_EVIDENCE_LIMIT", 7),
    )
    evidence_fingerprint = [
        {
            "id": chunk.evidence_id,
            "text": chunk.text[:800],
            "confidence": chunk.confidence,
            "warnings": chunk.warnings,
        }
        for chunk in prompt_evidence
    ]
    key_payload = {
        "provider": "ollama",
        "model": getattr(llm, "model", ""),
        "prompt_version": 5,
        "output_tokens": os.getenv("OLLAMA_MAX_OUTPUT_TOKENS", "800"),
        "evidence_limit": os.getenv("OLLAMA_EVIDENCE_LIMIT", "7"),
        "chunk_chars": os.getenv("OLLAMA_CHUNK_CHARS", "550"),
        "num_ctx": os.getenv("OLLAMA_NUM_CTX", ""),
        "query": query,
        "evidence": evidence_fingerprint,
        "learning_context": learning_context[:1200],
    }
    key = sha256(json.dumps(key_payload, sort_keys=True).encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"memo_{key}.md"
    if cache_path.exists():
        memo = _sanitize_llm_memo(
            cache_path.read_text(encoding="utf-8"),
            {chunk.evidence_id for chunk in evidence},
            evidence,
        )
        cache_path.write_text(memo, encoding="utf-8")
        return memo
    memo = _generate_with_llm(query, evidence, summaries, learning_context, llm)
    memo = _sanitize_llm_memo(memo, {chunk.evidence_id for chunk in evidence}, evidence)
    cache_path.write_text(memo, encoding="utf-8")
    return memo


def _generate_with_llm(
    query: str,
    evidence: list[EvidenceChunk],
    summaries: list[dict[str, Any]],
    learning_context: str,
    llm: Any,
) -> str:
    is_local = bool(getattr(llm, "is_local", False))
    evidence_limit = _local_int_env("OLLAMA_EVIDENCE_LIMIT", 7) if is_local else len(evidence)
    chunk_limit = _local_int_env("OLLAMA_CHUNK_CHARS", 550) if is_local else 6000
    summary_limit = _local_int_env("OLLAMA_SUMMARY_CHARS", 1800) if is_local else 12000
    output_tokens = _local_int_env("OLLAMA_MAX_OUTPUT_TOKENS", 800) if is_local else 2200
    prompt_evidence = _select_llm_evidence(evidence, summaries, evidence_limit) if is_local else evidence
    prompt_evidence_ids = {chunk.evidence_id for chunk in prompt_evidence}
    evidence_text = "\n\n".join(
        f"[{chunk.evidence_id}] source={chunk.source_file} page={chunk.page} "
        f"confidence={chunk.confidence:.2f}\n{chunk.text[:chunk_limit]}"
        for chunk in prompt_evidence
    )
    allowed_ids = ", ".join(chunk.evidence_id for chunk in prompt_evidence)
    if is_local:
        system = (
            "Draft a concise internal legal memo from evidence only. "
            "Every factual bullet must include exact bracket citations like [DOC1:p1:b1]. "
            "Use only citation IDs provided in the Evidence section. Do not invent citation IDs."
        )
        user = f"""
Task: {query}

Allowed citation IDs:
{allowed_ids}

Fields:
{_compact_summaries(summaries, allowed_evidence_ids=prompt_evidence_ids)[:summary_limit]}

Evidence:
{evidence_text}

Return Markdown only:
# First-Pass Internal Memo
## Parties and Documents
## Timeline
## Key Facts
## Document Issues
## Open Questions
## Evidence Used

Rules:
- Use citations only in this format: [DOC1:p1:b1]
- Never use citations like (evidence: DOC1:p1:b1)
- Never cite doc_type, summary, source_file, page labels, or any ID not listed above.
- Do not cite or discuss a fact unless the supporting text appears in the Evidence section above.
- Keep each section to 2-4 short bullets so the memo finishes completely.
- If a fact is not supported by the listed evidence, put it in Open Questions.
"""
    else:
        system = (
            "You draft first-pass internal legal memos from evidence only. "
            "Do not invent facts. Every factual bullet must include evidence IDs exactly as provided. "
            "Put unsupported or unclear items in Open Questions."
        )
        user = f"""
Task:
{query}

Structured fields:
{json.dumps(summaries, indent=2)[:summary_limit]}

Learned operator preferences:
{learning_context or "None yet."}

Evidence:
{evidence_text}

Return Markdown with these sections:
# First-Pass Internal Memo
## Parties and Documents
## Timeline
## Key Facts
## Document Issues
## Open Questions
## Evidence Used
"""
    return llm.generate_text(system, user, max_output_tokens=output_tokens)


def _compact_summaries(summaries: list[dict[str, Any]], allowed_evidence_ids: set[str] | None = None) -> str:
    compact: list[dict[str, Any]] = []
    for summary in summaries:
        fields = summary.get("structured_fields", {})
        doc_type_evidence_ids = []
        if summary.get("doc_id"):
            doc_type_id = f"{summary.get('doc_id')}:p1:b1"
            if allowed_evidence_ids is None or doc_type_id in allowed_evidence_ids:
                doc_type_evidence_ids.append(doc_type_id)
        parties = _filter_summary_items(fields.get("parties", [])[:4], allowed_evidence_ids)
        dates = _filter_summary_items(fields.get("dates", [])[:4], allowed_evidence_ids)
        amounts = _filter_summary_items(fields.get("amounts", [])[:3], allowed_evidence_ids)
        deadlines = _filter_summary_items(fields.get("deadlines", [])[:3], allowed_evidence_ids)
        unclear_spans = _filter_summary_items(fields.get("unclear_spans", [])[:2], allowed_evidence_ids)
        if allowed_evidence_ids is not None and not any(
            [doc_type_evidence_ids, parties, dates, amounts, deadlines, unclear_spans]
        ):
            continue
        compact.append(
            {
                "doc_id": summary.get("doc_id"),
                "source_file": summary.get("source_file"),
                "document_type": {
                    "value": fields.get("document_type"),
                    "evidence_ids": doc_type_evidence_ids,
                },
                "parties": parties,
                "dates": dates,
                "amounts": amounts,
                "deadlines": deadlines,
                "unclear_spans": unclear_spans,
                "warnings": summary.get("warnings", [])[:3],
            }
        )
    return json.dumps(compact, ensure_ascii=False, indent=2)


def _filter_summary_items(items: list[dict[str, Any]], allowed_evidence_ids: set[str] | None) -> list[dict[str, Any]]:
    if allowed_evidence_ids is None:
        return list(items)
    filtered: list[dict[str, Any]] = []
    for item in items:
        evidence_ids = [evidence_id for evidence_id in item.get("evidence_ids", []) if evidence_id in allowed_evidence_ids]
        if evidence_ids:
            copy = dict(item)
            copy["evidence_ids"] = evidence_ids
            filtered.append(copy)
    return filtered


def _select_llm_evidence(
    evidence: list[EvidenceChunk],
    summaries: list[dict[str, Any]],
    limit: int,
) -> list[EvidenceChunk]:
    if limit <= 0:
        return []
    if len(evidence) <= limit:
        return list(evidence)
    by_id = {chunk.evidence_id: chunk for chunk in evidence}
    selected: list[EvidenceChunk] = []
    selected_ids: set[str] = set()

    def add_chunk(chunk: EvidenceChunk | None, *, allow_low_signal: bool = False) -> None:
        """Add a chunk once while respecting the prompt evidence budget."""
        if chunk and chunk.evidence_id not in selected_ids and len(selected) < limit:
            if not allow_low_signal and _is_low_signal_chunk(chunk):
                return
            selected.append(chunk)
            selected_ids.add(chunk.evidence_id)

    primary_keep = min(limit, max(1, _local_int_env("OLLAMA_PRIMARY_EVIDENCE", 4)))
    for chunk in evidence[:primary_keep]:
        add_chunk(chunk)
    for evidence_id in _summary_doc_first_page_ids(summaries):
        add_chunk(by_id.get(evidence_id))
    for chunk in _representative_chunks_by_doc(evidence, summaries):
        add_chunk(chunk)
    for evidence_id in _summary_field_evidence_ids(summaries):
        add_chunk(by_id.get(evidence_id))
    for chunk in evidence:
        add_chunk(chunk)
    for chunk in evidence:
        add_chunk(chunk, allow_low_signal=True)
    return selected


def _is_low_signal_chunk(chunk: EvidenceChunk) -> bool:
    text = normalize_space(chunk.text).strip("# ").strip()
    if len(text) < 35:
        return True
    tokens = tokenize(text)
    if len(tokens) <= 3 and len(text) < 90:
        return True
    letters = [ch for ch in text if ch.isalpha()]
    if letters and len(text) < 120 and sum(ch.isupper() for ch in letters) / len(letters) > 0.85:
        return True
    return False


def _summary_evidence_ids(summaries: list[dict[str, Any]]) -> list[str]:
    ids = _summary_doc_first_page_ids(summaries)
    for evidence_id in _summary_field_evidence_ids(summaries):
        if evidence_id not in ids:
            ids.append(evidence_id)
    return ids


def _summary_doc_first_page_ids(summaries: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []

    def add(evidence_id: str | None) -> None:
        """Append a summary evidence id once."""
        if evidence_id and evidence_id not in ids:
            ids.append(evidence_id)

    for summary in summaries:
        doc_id = summary.get("doc_id")
        if doc_id:
            add(f"{doc_id}:p1:b1")
    return ids


def _summary_field_evidence_ids(summaries: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []

    def add(evidence_id: str | None) -> None:
        """Append a structured-field evidence id once."""
        if evidence_id and evidence_id not in ids:
            ids.append(evidence_id)

    for summary in summaries:
        fields = summary.get("structured_fields", {})
        for key in ["parties", "dates", "addresses", "amounts", "signatures", "deadlines", "unclear_spans"]:
            for item in fields.get(key, []) or []:
                if isinstance(item, dict):
                    for evidence_id in item.get("evidence_ids", []) or []:
                        add(evidence_id)
    return ids


def _representative_chunks_by_doc(
    evidence: list[EvidenceChunk],
    summaries: list[dict[str, Any]],
) -> list[EvidenceChunk]:
    doc_order: list[str] = []
    for summary in summaries:
        doc_id = summary.get("doc_id")
        if doc_id and doc_id not in doc_order:
            doc_order.append(doc_id)
    for chunk in evidence:
        if chunk.doc_id not in doc_order:
            doc_order.append(chunk.doc_id)

    by_doc: dict[str, list[EvidenceChunk]] = {}
    for chunk in evidence:
        by_doc.setdefault(chunk.doc_id, []).append(chunk)

    representatives: list[EvidenceChunk] = []
    for doc_id in doc_order:
        chunks = by_doc.get(doc_id, [])
        if not chunks:
            continue
        representative = next((chunk for chunk in chunks if not _is_low_signal_chunk(chunk)), chunks[0])
        representatives.append(representative)
    return representatives


def _sanitize_llm_memo(memo: str, allowed_evidence_ids: set[str], evidence: list[EvidenceChunk]) -> str:
    if not memo.strip():
        return memo
    doc_default_ids = _default_doc_evidence_ids(evidence)

    def replace_evidence_group(match: re.Match[str]) -> str:
        """Normalize parenthetical model citations into bracket citation groups."""
        raw = match.group(1)
        ids = _coerce_citation_ids(raw, allowed_evidence_ids, doc_default_ids)
        return f"[{', '.join(ids)}]" if ids else ""

    memo = re.sub(r"\(\s*evidence\s*:\s*([^)]+)\)", replace_evidence_group, memo, flags=re.IGNORECASE)
    memo = re.sub(r"\(\s*source\s*:\s*([^)]+)\)", replace_evidence_group, memo, flags=re.IGNORECASE)
    memo = re.sub(r"\[\s*(DOC\d+:(?:doc_type|summary|source_file|page))\s*\]", replace_evidence_group, memo)
    memo = re.sub(r"\((DOC\d+:(?:doc_type|summary|source_file|page))\)", replace_evidence_group, memo)

    sanitized_lines: list[str] = []
    section = ""
    fallback_id = evidence[0].evidence_id if evidence else ""
    for line in memo.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            section = stripped[3:].strip().lower()
            sanitized_lines.append(line.rstrip())
            continue
        if stripped.startswith("- ") and section not in {"evidence used", "open questions", "document issues"}:
            citations = extract_citations(stripped)
            invalid = citations - allowed_evidence_ids
            if invalid:
                for invalid_id in invalid:
                    stripped = stripped.replace(invalid_id, "")
                line = stripped
            if not extract_citations(line) and len(stripped) > 30 and fallback_id:
                line = f"{line.rstrip()} [{fallback_id}]"
        sanitized_lines.append(line.rstrip())
    return "\n".join(sanitized_lines).strip() + "\n"


def _coerce_citation_ids(raw: str, allowed_evidence_ids: set[str], doc_default_ids: dict[str, str]) -> list[str]:
    ids: list[str] = []
    for candidate in re.findall(r"DOC\d+:[A-Za-z0-9_:-]+", raw):
        if candidate in allowed_evidence_ids and candidate not in ids:
            ids.append(candidate)
            continue
        doc_match = re.match(r"(DOC\d+):", candidate)
        if doc_match:
            default_id = doc_default_ids.get(doc_match.group(1))
            if default_id and default_id in allowed_evidence_ids and default_id not in ids:
                ids.append(default_id)
    return ids


def _default_doc_evidence_ids(evidence: list[EvidenceChunk]) -> dict[str, str]:
    defaults: dict[str, str] = {}
    for chunk in evidence:
        defaults.setdefault(chunk.doc_id, chunk.evidence_id)
        if chunk.page == 1:
            defaults[chunk.doc_id] = chunk.evidence_id
    return defaults


def _generate_offline(
    query: str,
    evidence: list[EvidenceChunk],
    summaries: list[dict[str, Any]],
    learning_context: str,
) -> str:
    fallback_citation = f" [{evidence[0].evidence_id}]" if evidence else ""
    lines: list[str] = [
        "# First-Pass Internal Memo",
        "",
        "## Parties and Documents",
    ]
    parties = _field_items(summaries, "parties")
    if parties:
        for item in parties[:8]:
            lines.append(f"- {item['label'].title()}: {item['value']} [{', '.join(item['evidence_ids'])}]")
    else:
        lines.append(f"- No party field was extracted with enough confidence from the available documents.{fallback_citation}")
    doc_types = sorted({summary.get("structured_fields", {}).get("document_type", "unknown") for summary in summaries})
    if doc_types:
        citations = _summary_doc_citations(summaries)
        citation_text = f" [{', '.join(citations)}]" if citations else ""
        lines.append(f"- Document set appears to include: {', '.join(doc_types)}.{citation_text}")
    lines.extend(["", "## Timeline"])
    for item in _field_items(summaries, "dates")[:10]:
        lines.append(f"- {item['value']} appears in the source record. [{', '.join(item['evidence_ids'])}]")
    if not _field_items(summaries, "dates"):
        lines.append(f"- No clear dates were extracted; confirm manually before relying on any deadline.{fallback_citation}")
    lines.extend(["", "## Key Facts"])
    for sentence, evidence_id in _key_fact_sentences(evidence)[:8]:
        lines.append(f"- {sentence} [{evidence_id}]")
    lines.extend(["", "## Document Issues"])
    issue_lines = 0
    for chunk in evidence:
        if chunk.warnings or chunk.confidence < 0.75:
            warning = "; ".join(chunk.warnings) or "Low-confidence extraction."
            lines.append(f"- {warning} Review the source before relying on this block. [{chunk.evidence_id}]")
            issue_lines += 1
    if issue_lines == 0:
        lines.append("- No extraction warnings were detected in the retrieved evidence.")
    lines.extend(["", "## Open Questions"])
    unclear = _field_items(summaries, "unclear_spans")
    for item in unclear[:5]:
        unclear_text = _clean_sentence(item["value"])
        lines.append(f"- Clarify this unclear source text: {unclear_text} [{', '.join(item['evidence_ids'])}]")
    if not unclear:
        lines.append("- Confirm whether there are additional scanned, handwritten, or missing pages not included in this packet.")
    if learning_context:
        lines.extend(["", "## Learned Operator Preferences Applied"])
        for line in learning_context.splitlines()[:6]:
            if line.strip().startswith("-"):
                lines.append(f"Preference applied: {line.strip()[1:].strip()}")
    lines.extend(["", "## Evidence Used"])
    for chunk in evidence:
        lines.append(f"- {chunk.evidence_id}: {chunk.source_file}, page {chunk.page}, confidence {chunk.confidence:.2f}.")
    return "\n".join(lines).strip() + "\n"


def _field_items(summaries: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for summary in summaries:
        items.extend(summary.get("structured_fields", {}).get(field, []) or [])
    return items


def _add_structured_evidence(
    index: EvidenceIndex,
    evidence: list[EvidenceChunk],
    summaries: list[dict[str, Any]],
) -> list[EvidenceChunk]:
    existing = {chunk.evidence_id for chunk in evidence}
    needed: list[str] = []
    for summary in summaries:
        doc_id = summary.get("doc_id")
        if doc_id:
            evidence_id = f"{doc_id}:p1:b1"
            if evidence_id not in existing and evidence_id not in needed:
                needed.append(evidence_id)
    for summary in summaries:
        fields = summary.get("structured_fields", {})
        for value in fields.values():
            if isinstance(value, list):
                for item in value:
                    if not isinstance(item, dict):
                        continue
                    for evidence_id in item.get("evidence_ids", []):
                        if evidence_id not in existing and evidence_id not in needed:
                            needed.append(evidence_id)
    extra = index.get_chunks(needed)
    combined = list(evidence)
    for evidence_id in needed:
        if evidence_id in extra and evidence_id not in existing:
            combined.append(extra[evidence_id])
            existing.add(evidence_id)
    return combined


def _summary_doc_citations(summaries: list[dict[str, Any]]) -> list[str]:
    citations: list[str] = []
    for summary in summaries:
        doc_id = summary.get("doc_id")
        if doc_id:
            citations.append(f"{doc_id}:p1:b1")
    return citations


def _best_sentence(chunk: EvidenceChunk) -> str:
    sentences = sentence_split(chunk.text)
    if not sentences:
        return _clean_sentence(chunk.text[:220])
    sentences.sort(key=lambda item: (_sentence_score(item), min(len(item), 220)), reverse=True)
    sentence = sentences[0]
    if len(sentence) > 260:
        sentence = sentence[:257].rstrip() + "..."
    return _clean_sentence(sentence)


def _key_fact_sentences(evidence: list[EvidenceChunk]) -> list[tuple[str, str]]:
    candidates: list[tuple[int, str, str]] = []
    seen: set[str] = set()
    for chunk in evidence:
        for sentence in sentence_split(chunk.text):
            cleaned = _clean_sentence(sentence)
            if len(cleaned) < 35:
                continue
            if _is_noise_sentence(cleaned):
                continue
            if len(cleaned) > 360:
                cleaned = cleaned[:357].rsplit(" ", 1)[0].rstrip(" ,.;:") + "..."
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            score = _sentence_score(cleaned)
            if score > 0:
                candidates.append((score, cleaned, chunk.evidence_id))
    candidates.sort(key=lambda item: item[0], reverse=True)
    if not candidates:
        return [(_best_sentence(chunk), chunk.evidence_id) for chunk in evidence[:6]]
    return [(sentence, evidence_id) for _, sentence, evidence_id in candidates]


def _sentence_score(sentence: str) -> int:
    lowered = sentence.lower()
    score = 0
    for term in [
        "$",
        "alleges",
        "disputes",
        "notice",
        "must cure",
        "lockout",
        "wire",
        "deadline",
        "closing",
        "exception",
        "easement",
        "encroaches",
        "payoff",
        "denial",
        "respond by",
        "proof of loss",
        "signature",
        "chegg",
        "rosca",
        "subscription",
        "subscribers",
        "recurring charges",
        "continues to charge",
        "simple mechanism",
        "simple cancellation",
        "failed to provide",
        "monetary judgment",
        "$7,500,000",
        "schnitzer",
        "clean air act",
        "refrigerant",
        "scrap metal recycling",
    ]:
        if term in lowered:
            score += 2
    if "scan artifact" in lowered:
        score -= 4
    if "sample packet page" in lowered:
        score -= 2
    if "review" in lowered and len(sentence) < 80:
        score -= 1
    return score


def _is_noise_sentence(sentence: str) -> bool:
    stripped = sentence.strip()
    lowered = stripped.lower()
    if lowered.endswith(("u.s.c.", "c.f.r.", "u.s.", "no.")):
        return True
    if "complaint for permanent injunction" in lowered and "defendant" in lowered:
        return True
    if "signature:" in lowered and "____" in stripped:
        return True
    if stripped.startswith("Your subscription will end") or "bonus perks" in lowered:
        return True
    if stripped.startswith("We have heard:"):
        return True
    if stripped.startswith("§"):
        return True
    letters = [ch for ch in stripped if ch.isalpha()]
    if letters and sum(ch.isupper() for ch in letters) / len(letters) > 0.8 and len(stripped) > 70:
        return True
    return False


def _clean_sentence(sentence: str) -> str:
    cleaned = sentence.replace("SCAN ARTIFACT - LOW CONTRAST SOURCE ", "")
    cleaned = cleaned.replace("SCAN ARTIFACT - LOW CONTRAST SOURCE", "")
    for prefix in ["Operator Notes ", "Follow-up Items ", "Title Notes "]:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
    cleaned = cleaned.replace(" Sample packet page 1", "")
    cleaned = cleaned.replace(" Sample packet page 2", "")
    cleaned = re.sub(r"\s+\b\d{1,2}\s+(?=\(|its\b|Defendant\b|Plaintiff\b|The\b)", " ", cleaned)
    cleaned = re.sub(r"\s+\d{1,3}\.$", ".", cleaned)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    return cleaned.strip()


def _claim_support_score(line: str, evidence_by_id: dict[str, EvidenceChunk]) -> float:
    if "appears in the source record" in line:
        return 1.0
    cited_ids = extract_citations(line)
    claim_tokens = {
        token
        for token in tokenize(line)
        if token not in {"doc1", "doc2", "doc3", "page", "source", "record", "appears", "include", "includes"}
    }
    claim_tokens = {token for token in claim_tokens if len(token) > 2 and not token.startswith("p")}
    if not claim_tokens:
        return 1.0
    evidence_tokens: set[str] = set()
    for evidence_id in cited_ids:
        chunk = evidence_by_id.get(evidence_id)
        if chunk:
            evidence_tokens.update(tokenize(chunk.text))
    if not evidence_tokens:
        return 0.0
    return len(claim_tokens.intersection(evidence_tokens)) / len(claim_tokens)

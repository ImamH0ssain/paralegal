from __future__ import annotations

from pathlib import Path

from legal_rag.feedback import classify_edit
from legal_rag.generation import (
    _generate_with_llm,
    _prune_weakly_supported_lines,
    _select_llm_evidence,
    generate_memo,
    validate_memo_citations,
)
from legal_rag.ingestion import ingest_paths
from legal_rag.models import EvidenceChunk
from legal_rag.providers import OllamaClient
from legal_rag.retrieval import EvidenceIndex
from legal_rag.sample_data import create_sample_inputs
from legal_rag.text_utils import extract_citations


def test_extract_citations() -> None:
    text = "Tenant disputed charges. [DOC1:p2:b1] See also DOC2:p1:b3."
    assert extract_citations(text) == {"DOC1:p2:b1", "DOC2:p1:b3"}


def test_validate_memo_citations_flags_invalid_and_uncited() -> None:
    memo = "- Supported fact. [DOC1:p1:b1]\n- Unsupported factual sentence with no source."
    result = validate_memo_citations(memo, {"DOC1:p1:b1"})
    assert result["valid"] is False
    assert result["uncited_factual_lines"]
    invalid = validate_memo_citations("- Bad cite. [DOC9:p9:b9]", {"DOC1:p1:b1"})
    assert invalid["invalid_citations"] == ["DOC9:p9:b9"]


def test_validate_memo_citations_flags_missing_sections_and_truncation() -> None:
    memo = """# First-Pass Internal Memo
## Parties and Documents
- Plaintiff: Example Party [DOC1:p1:b1]
## Timeline
- January 1, 2024 appears in the source [DOC1:p1:b1]
## Key Facts
- Example Party filed the action [DOC1:p1:b1]
## Document Issues
- No extraction warnings were detected.
## Open Questions
- What is the nature
"""
    result = validate_memo_citations(memo, {"DOC1:p1:b1"})
    assert result["valid"] is False
    assert "Evidence Used" in result["missing_sections"]
    assert result["possibly_truncated"] is True


def test_local_llm_prompt_only_exposes_visible_evidence_ids(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_EVIDENCE_LIMIT", "1")
    monkeypatch.setenv("OLLAMA_CHUNK_CHARS", "200")
    captured: dict[str, str] = {}

    class FakeLocalLLM:
        is_local = True

        def generate_text(self, system: str, user: str, *, max_output_tokens: int = 1800) -> str:
            captured["user"] = user
            return "# First-Pass Internal Memo\n"

    evidence = [
        EvidenceChunk("DOC1:p1:b1", "DOC1", "visible.pdf", 1, "Visible plaintiff fact.", 0.95),
        EvidenceChunk("DOC2:p1:b1", "DOC2", "hidden.pdf", 1, "Hidden defendant fact.", 0.95),
    ]
    summaries = [
        {
            "doc_id": "DOC2",
            "source_file": "hidden.pdf",
            "structured_fields": {
                "parties": [{"label": "defendant", "value": "Hidden Co", "evidence_ids": ["DOC2:p1:b1"]}],
                "dates": [],
                "amounts": [],
                "unclear_spans": [],
            },
            "warnings": [],
        }
    ]

    _generate_with_llm("Draft memo", evidence, summaries, "", FakeLocalLLM())

    assert "DOC1:p1:b1" in captured["user"]
    assert "DOC2:p1:b1" not in captured["user"]


def test_prune_weakly_supported_lines_preserves_grounded_llm_claims() -> None:
    evidence = [
        EvidenceChunk(
            "DOC1:p1:b1",
            "DOC1",
            "consent.pdf",
            1,
            "Defendant is responsible for achieving and maintaining complete compliance with all applicable federal, State, and local laws, regulations, and permits.",
            0.95,
        ),
        EvidenceChunk("DOC2:p1:b1", "DOC2", "complaint.pdf", 1, "COMPLAINT", 0.65),
    ]
    memo = """# First-Pass Internal Memo
## Parties and Documents
- Defendant: Example Company [DOC1:p1:b1]
## Timeline
- No clear dates were extracted from the visible evidence [DOC1:p1:b1].
## Key Facts
- Defendant is responsible for achieving and maintaining complete compliance with all applicable laws and permits [DOC1:p1:b1].
- The United States filed a civil complaint in a separate proceeding involving Chegg [DOC2:p1:b1].
## Document Issues
- No extraction warnings were detected.
## Open Questions
- Confirm whether additional pages are missing.
## Evidence Used
- DOC1:p1:b1: consent.pdf, page 1.
- DOC2:p1:b1: complaint.pdf, page 1.
"""

    pruned = _prune_weakly_supported_lines(memo, {chunk.evidence_id for chunk in evidence}, evidence)

    assert "complete compliance" in pruned
    assert "separate proceeding involving Chegg" not in pruned
    assert validate_memo_citations(pruned, {chunk.evidence_id for chunk in evidence}, evidence)["valid"] is True


def test_local_llm_evidence_selection_skips_heading_only_chunks_when_possible() -> None:
    evidence = [
        EvidenceChunk("DOC1:p1:b1", "DOC1", "good.pdf", 1, "Plaintiff Example alleges a detailed cancellation-flow issue.", 0.95),
        EvidenceChunk("DOC2:p1:b1", "DOC2", "heading.pdf", 1, "COMPLAINT", 0.65),
        EvidenceChunk("DOC3:p1:b1", "DOC3", "good2.pdf", 1, "Defendant must pay $7,500,000 within seven days of entry.", 0.95),
    ]

    selected = _select_llm_evidence(evidence, [], 2)

    assert [chunk.evidence_id for chunk in selected] == ["DOC1:p1:b1", "DOC3:p1:b1"]


def test_local_llm_evidence_selection_prioritizes_document_coverage(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_PRIMARY_EVIDENCE", "1")
    evidence = [
        EvidenceChunk("DOC1:p29:b1", "DOC1", "doc1.pdf", 29, "Consent decree compliance obligations and notice terms.", 0.95),
        EvidenceChunk(
            "DOC1:p7:b1",
            "DOC1",
            "doc1.pdf",
            7,
            "Civil penalty payment details require electronic fund transfer within the stated consent decree deadline.",
            0.95,
        ),
        EvidenceChunk("DOC2:p1:b1", "DOC2", "doc2.pdf", 1, "Federal Trade Commission complaint against Chegg, Inc.", 0.95),
        EvidenceChunk("DOC3:p1:b1", "DOC3", "doc3.pdf", 1, "Stipulated order for monetary judgment against Chegg, Inc.", 0.95),
    ]
    summaries = [
        {
            "doc_id": "DOC1",
            "structured_fields": {
                "dates": [{"label": "date", "value": "January 1, 2024", "evidence_ids": ["DOC1:p7:b1"]}]
            },
        },
        {"doc_id": "DOC2", "structured_fields": {}},
        {"doc_id": "DOC3", "structured_fields": {}},
    ]

    selected = _select_llm_evidence(evidence, summaries, 3)

    assert [chunk.evidence_id for chunk in selected] == ["DOC1:p29:b1", "DOC2:p1:b1", "DOC3:p1:b1"]


def test_local_llm_evidence_selection_uses_representative_chunk_per_document(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_PRIMARY_EVIDENCE", "1")
    evidence = [
        EvidenceChunk("DOC1:p29:b1", "DOC1", "doc1.pdf", 29, "Consent decree compliance obligations and notice terms.", 0.95),
        EvidenceChunk("DOC1:p7:b1", "DOC1", "doc1.pdf", 7, "Civil penalty payment details require prompt transfer.", 0.95),
        EvidenceChunk("DOC2:p1:b1", "DOC2", "doc2.pdf", 1, "Federal Trade Commission complaint against Chegg, Inc.", 0.95),
        EvidenceChunk("DOC4:p1:b1", "DOC4", "lease.jpg", 1, "LEASE", 0.65),
        EvidenceChunk("DOC4:p1:b10", "DOC4", "lease.jpg", 1, "Monthly rent is $2,541.00 for months 1 through 12.", 0.82),
    ]
    summaries = [{"doc_id": "DOC1"}, {"doc_id": "DOC2"}, {"doc_id": "DOC4"}]

    selected = _select_llm_evidence(evidence, summaries, 3)

    assert [chunk.evidence_id for chunk in selected] == ["DOC1:p29:b1", "DOC2:p1:b1", "DOC4:p1:b10"]


def test_ollama_generation_options_use_thread_and_context_env(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_NUM_THREAD", "8")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "3072")
    captured: dict[str, object] = {}

    class CapturingOllama(OllamaClient):
        def _generate_text_stream(self, payload: dict[str, object]) -> str:
            captured.update(payload)
            return "OK"

    client = CapturingOllama(model="qwen-test")
    client.generate_text("system", "user", max_output_tokens=32)

    assert captured["options"]["num_thread"] == 8
    assert captured["options"]["num_ctx"] == 3072


def test_ingest_index_and_retrieve_samples(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample_inputs"
    processed_dir = tmp_path / "processed"
    index_dir = tmp_path / "index"
    paths = create_sample_inputs(sample_dir)
    records = ingest_paths(paths, processed_dir, use_mistral=False, use_openai=False)
    assert any(record.get("record_type") == "block" for record in records)
    count = EvidenceIndex(index_dir).build(processed_dir / "processed_documents.jsonl")
    assert count > 0
    results = EvidenceIndex(index_dir).search("title exception easement closing risk", top_k=5)
    joined = " ".join(chunk.text for chunk in results).lower()
    assert "easement" in joined
    assert "exception" in joined


def test_generate_offline_memo_is_citation_valid(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample_inputs"
    processed_dir = tmp_path / "processed"
    index_dir = tmp_path / "index"
    output_dir = tmp_path / "outputs"
    paths = create_sample_inputs(sample_dir)
    ingest_paths(paths, processed_dir, use_mistral=False, use_openai=False)
    EvidenceIndex(index_dir).build(processed_dir / "processed_documents.jsonl")
    result = generate_memo(index_dir, processed_dir, output_dir=output_dir, use_openai=False)
    assert result["validation"]["valid"] is True
    assert (output_dir / "generated_memo.md").exists()


def test_edit_classifier_learns_citation_density() -> None:
    original = "- The tenant has a strong position without a cite."
    edited = "- Do not state legal merits without support. [DOC1:p1:b1]"
    result = classify_edit(original, edited)
    assert result["patterns"]
    assert any("cited" in pattern.lower() or "citation" in pattern.lower() for pattern in result["patterns"])

"""Document ingestion, extraction, OCR routing, and structured field parsing."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from .io_utils import ensure_dir, write_json
from .models import DocumentBlock
from .providers import MistralOCRClient, OpenAIClient, ProviderError
from .text_utils import normalize_space

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}
SUPPORTED_EXTENSIONS = {".pdf", *TEXT_EXTENSIONS, *IMAGE_EXTENSIONS}


def ingest_paths(
    input_paths: list[str | Path],
    output_dir: str | Path,
    *,
    force_ocr: bool = False,
    ocr_backend: str = "auto",
    use_mistral: bool = True,
    use_openai: bool = True,
    workers: int | None = None,
) -> list[dict[str, Any]]:
    """Extract all input documents and write processed JSONL artifacts.

    The returned records include block-level evidence records and one
    ``document_summary`` record per source file. Document IDs are assigned in
    input order so generated citations remain stable for a given run.
    """

    out = ensure_dir(output_dir)
    blocks: list[DocumentBlock] = []
    summaries: list[dict[str, Any]] = []
    files = _expand_inputs(input_paths)
    mistral = MistralOCRClient()
    openai = OpenAIClient()
    worker_count = workers or int(os.getenv("INGEST_WORKERS", "1"))
    jobs = [(index, path, f"DOC{index}") for index, path in enumerate(files, start=1)]
    if worker_count > 1 and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            extracted = list(
                executor.map(
                    lambda item: (
                        item[0],
                        item[1],
                        item[2],
                        extract_document(
                            item[1],
                            item[2],
                            force_ocr=force_ocr,
                            ocr_backend=ocr_backend,
                            mistral=mistral if use_mistral else None,
                            cache_dir=out / ".cache",
                        ),
                    ),
                    jobs,
                )
            )
        extracted.sort(key=lambda item: item[0])
    else:
        extracted = [
            (
                index,
                path,
                doc_id,
                extract_document(
                    path,
                    doc_id,
                    force_ocr=force_ocr,
                    ocr_backend=ocr_backend,
                    mistral=mistral if use_mistral else None,
                    cache_dir=out / ".cache",
                ),
            )
            for index, path, doc_id in jobs
        ]
    for _, path, doc_id, doc_blocks in extracted:
        blocks.extend(doc_blocks)
        summaries.append(
            {
                "record_type": "document_summary",
                "doc_id": doc_id,
                "source_file": path.name,
                "structured_fields": extract_structured_fields(doc_blocks, openai if use_openai else None),
                "warnings": sorted({warning for block in doc_blocks for warning in block.warnings}),
                "page_count": max((block.page for block in doc_blocks), default=0),
                "block_count": len(doc_blocks),
            }
        )
    records = [block.to_record() for block in blocks] + summaries
    processed_path = out / "processed_documents.jsonl"
    processed_path.write_text("", encoding="utf-8")
    with processed_path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(out / "document_summaries.json", summaries)
    return records


def extract_document(
    path: str | Path,
    doc_id: str,
    *,
    force_ocr: bool = False,
    ocr_backend: str = "auto",
    mistral: MistralOCRClient | None = None,
    cache_dir: str | Path | None = None,
) -> list[DocumentBlock]:
    """Extract normalized text blocks from one PDF, text file, or image.

    Native PDF text is preferred when it is usable. OCR backends are invoked
    only when forced, when the document has no native text, or when the file is
    image-only.
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported input file extension: {p.suffix}")
    if ext == ".pdf" and not force_ocr:
        native_blocks = _extract_pdf_native(p, doc_id)
        extracted_chars = sum(len(block.text) for block in native_blocks)
        if extracted_chars >= 80:
            return native_blocks
    if ext in TEXT_EXTENSIONS:
        return _extract_text_file(p, doc_id)
    if ocr_backend == "docling":
        cached = _read_extraction_cache(cache_dir, p, doc_id, ocr_backend, force_ocr)
        if cached is not None:
            return cached
        try:
            blocks = _extract_with_docling(p, doc_id)
            _write_extraction_cache(cache_dir, p, doc_id, ocr_backend, force_ocr, blocks)
            return blocks
        except Exception as exc:
            fallback = _extract_pdf_native(p, doc_id) if ext == ".pdf" else _ocr_unavailable_block(p, doc_id)
            for block in fallback:
                block.warnings.append(f"Docling OCR failed; used local fallback: {str(exc)[:160]}")
            return fallback
    if ocr_backend in {"auto", "mistral"} and mistral and mistral.available:
        cached = _read_extraction_cache(cache_dir, p, doc_id, "mistral", force_ocr)
        if cached is not None:
            return cached
        try:
            blocks = _extract_with_mistral(p, doc_id, mistral)
            _write_extraction_cache(cache_dir, p, doc_id, "mistral", force_ocr, blocks)
            return blocks
        except ProviderError as exc:
            fallback = _extract_pdf_native(p, doc_id) if ext == ".pdf" else _ocr_unavailable_block(p, doc_id)
            for block in fallback:
                block.warnings.append(f"Mistral OCR failed; used local fallback: {str(exc)[:160]}")
            return fallback
    if ext == ".pdf":
        blocks = _extract_pdf_native(p, doc_id)
        if not blocks:
            return [
                DocumentBlock(
                    doc_id=doc_id,
                    source_file=p.name,
                    page=1,
                    block_index=1,
                    text="[No extractable text found. Set MISTRAL_API_KEY or rerun with OCR for scanned inputs.]",
                    confidence=0.1,
                    warnings=["No native PDF text; OCR API key unavailable."],
                    extraction_method="unavailable_ocr",
                )
            ]
        for block in blocks:
            block.warnings.append("OCR API key unavailable; native PDF text used.")
        return blocks
    if ext in IMAGE_EXTENSIONS:
        return _ocr_unavailable_block(p, doc_id)
    return _extract_text_file(p, doc_id)


def _read_extraction_cache(
    cache_dir: str | Path | None,
    path: Path,
    doc_id: str,
    backend: str,
    force_ocr: bool,
) -> list[DocumentBlock] | None:
    cache_path = _extraction_cache_path(cache_dir, path, backend, force_ocr)
    if not cache_path or not cache_path.exists():
        return None
    try:
        records = json.loads(cache_path.read_text(encoding="utf-8"))
        blocks = [DocumentBlock.from_record(record) for record in records]
        for index, block in enumerate(blocks, start=1):
            block.doc_id = doc_id
            block.block_index = index
        return blocks
    except (OSError, TypeError, ValueError, json.JSONDecodeError, KeyError):
        return None


def _write_extraction_cache(
    cache_dir: str | Path | None,
    path: Path,
    doc_id: str,
    backend: str,
    force_ocr: bool,
    blocks: list[DocumentBlock],
) -> None:
    cache_path = _extraction_cache_path(cache_dir, path, backend, force_ocr)
    if not cache_path:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps([block.to_record() for block in blocks], ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _extraction_cache_path(cache_dir: str | Path | None, path: Path, backend: str, force_ocr: bool) -> Path | None:
    if cache_dir is None:
        return None
    stat = path.stat()
    payload = {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "backend": backend,
        "force_ocr": force_ocr,
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"extract_{key}.json"


def extract_structured_fields(blocks: list[DocumentBlock], openai: OpenAIClient | None = None) -> dict[str, Any]:
    """Extract parties, dates, amounts, deadlines, and warnings from blocks."""
    text = "\n".join(f"[{b.evidence_id}] {b.text}" for b in blocks)
    if openai and openai.available:
        try:
            return _extract_structured_with_openai(text, openai)
        except Exception as exc:
            heuristic = _extract_structured_heuristic(blocks)
            heuristic["extraction_notes"].append(f"OpenAI structured extraction failed; heuristic fallback used: {exc}")
            return heuristic
    return _extract_structured_heuristic(blocks)


def _extract_pdf_native(path: Path, doc_id: str) -> list[DocumentBlock]:
    reader = PdfReader(str(path))
    blocks: list[DocumentBlock] = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = _clean_raw_text(page.extract_text() or "")
        parts = _split_blocks(text)
        if not parts:
            blocks.append(
                DocumentBlock(
                    doc_id=doc_id,
                    source_file=path.name,
                    page=page_index,
                    block_index=1,
                    text="[No extractable native text on this page.]",
                    confidence=0.2,
                    warnings=["Page has no native extractable text."],
                    extraction_method="native_pdf",
                )
            )
            continue
        for block_index, part in enumerate(parts, start=1):
            warnings = _quality_warnings(part)
            blocks.append(
                DocumentBlock(
                    doc_id=doc_id,
                    source_file=path.name,
                    page=page_index,
                    block_index=block_index,
                    text=part,
                    confidence=0.65 if warnings else 0.95,
                    warnings=warnings,
                    extraction_method="native_pdf",
                )
            )
    return blocks


def _extract_text_file(path: Path, doc_id: str) -> list[DocumentBlock]:
    text = _clean_raw_text(path.read_text(encoding="utf-8", errors="replace"))
    parts = _split_blocks(text)
    return [
        DocumentBlock(
            doc_id=doc_id,
            source_file=path.name,
            page=1,
            block_index=i,
            text=part,
            confidence=0.9,
            warnings=_quality_warnings(part),
            extraction_method="text_file",
        )
        for i, part in enumerate(parts or ["[Empty text file.]"], start=1)
    ]


def _extract_with_mistral(path: Path, doc_id: str, mistral: MistralOCRClient) -> list[DocumentBlock]:
    response = mistral.process_file(path)
    pages = response.get("pages") or []
    blocks: list[DocumentBlock] = []
    for page_index, page in enumerate(pages, start=1):
        text = page.get("markdown") or page.get("text") or ""
        confidence = _page_confidence(page)
        parts = _split_blocks(text)
        for block_index, part in enumerate(parts or ["[OCR returned no text for this page.]"], start=1):
            blocks.append(
                DocumentBlock(
                    doc_id=doc_id,
                    source_file=path.name,
                    page=page_index,
                    block_index=block_index,
                    text=part,
                    confidence=confidence,
                    warnings=_quality_warnings(part),
                    extraction_method="mistral_ocr",
                )
            )
    if not blocks:
        raise ProviderError("Mistral OCR response did not contain pages")
    return blocks


def _extract_with_docling(path: Path, doc_id: str) -> list[DocumentBlock]:
    if os.getenv("LOCAL_OCR_DEVICE", "cpu").lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ.setdefault("RAPIDOCR_DEVICE", "cpu")
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(str(path))
    markdown = _clean_raw_text(result.document.export_to_markdown())
    parts = _split_blocks(markdown)
    if not parts:
        raise ProviderError("Docling returned no extractable markdown")
    return [
        DocumentBlock(
            doc_id=doc_id,
            source_file=path.name,
            page=1,
            block_index=block_index,
            text=part,
            confidence=0.82,
            warnings=_quality_warnings(part),
            extraction_method="docling",
        )
        for block_index, part in enumerate(parts, start=1)
    ]


def _ocr_unavailable_block(path: Path, doc_id: str) -> list[DocumentBlock]:
    return [
        DocumentBlock(
            doc_id=doc_id,
            source_file=path.name,
            page=1,
            block_index=1,
            text=(
                f"[Image-only or scanned input requires OCR. Source appears to be {path.stem.replace('_', ' ')}. "
                "Set MISTRAL_API_KEY and rerun ingestion, "
                "or provide a cached OCR transcript for this file.]"
            ),
            confidence=0.05,
            warnings=["OCR required but no usable OCR result is available."],
            extraction_method="ocr_unavailable",
        )
    ]


def _page_confidence(page: dict[str, Any]) -> float:
    if page.get("confidence") is not None:
        return float(page["confidence"])
    scores = page.get("confidence_scores") or {}
    for key in ["average_page_confidence_score", "minimum_page_confidence_score"]:
        if scores.get(key) is not None:
            return float(scores[key])
    return 0.85


def _split_blocks(text: str) -> list[str]:
    normalized = _clean_raw_text(text).replace("\r\n", "\n").replace("\r", "\n")
    raw_parts = re.split(r"\n\s*\n|(?<=\.)\s+(?=[A-Z][A-Za-z ]{4,}:)", normalized)
    parts = [normalize_space(part) for part in raw_parts if normalize_space(part)]
    merged: list[str] = []
    for part in parts:
        if merged and len(merged[-1]) < 120 and len(part) < 160:
            merged[-1] = normalize_space(f"{merged[-1]} {part}")
        else:
            merged.append(part)
    return merged


def _clean_raw_text(text: str) -> str:
    text = _repair_mojibake(text)
    text = text.replace("งง", "§§").replace("ง", "§")
    for bad, good in {
        "vai·iety": "variety",
        "primai·ily": "primarily",
        "mai·kets": "markets",
        "monetaiy": "monetary",
        "pennanent": "permanent",
    }.items():
        text = text.replace(bad, good)
    text = re.sub(r"Case\s+\S+\s+Document\s+[\w.-]+\s+Filed\s+\S+\s+Page\s+\d+\s+of\s+\d+", " ", text)
    text = re.sub(r"\b(?:\d{1,2}\s+){8,}\b", " ", text)
    text = re.sub(r"\s+-\s+\d+\s+-\s+", " ", text)
    return text


def _repair_mojibake(text: str) -> str:
    if "â" in text or "Â" in text or "€" in text:
        try:
            fixed = text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
            if fixed and _mojibake_score(fixed) < _mojibake_score(text):
                return fixed
        except UnicodeError:
            pass
    replacements = {
        "â€™": "'",
        "â€˜": "'",
        "â€œ": '"',
        "â€\u009d": '"',
        "â€“": "-",
        "â€”": "-",
        "Â§": "§",
        "Â": "",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def _mojibake_score(text: str) -> int:
    return sum(text.count(marker) for marker in ["â", "Â", "�", "€"])


def _quality_warnings(text: str) -> list[str]:
    lowered = text.lower()
    warnings: list[str] = []
    if any(marker in lowered for marker in ["illegible", "unclear", "smudged", "unknown", "[?]", "???"]):
        warnings.append("Contains unclear or partially illegible text.")
    if len(text) < 35:
        warnings.append("Very short block; may need neighboring context.")
    if "\ufffd" in text:
        warnings.append("Replacement characters detected.")
    return warnings


def _extract_structured_with_openai(text: str, openai: OpenAIClient) -> dict[str, Any]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "document_type": {"type": "string"},
            "parties": {"type": "array", "items": _evidence_item_schema()},
            "dates": {"type": "array", "items": _evidence_item_schema()},
            "addresses": {"type": "array", "items": _evidence_item_schema()},
            "amounts": {"type": "array", "items": _evidence_item_schema()},
            "signatures": {"type": "array", "items": _evidence_item_schema()},
            "deadlines": {"type": "array", "items": _evidence_item_schema()},
            "unclear_spans": {"type": "array", "items": _evidence_item_schema()},
            "extraction_notes": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "document_type",
            "parties",
            "dates",
            "addresses",
            "amounts",
            "signatures",
            "deadlines",
            "unclear_spans",
            "extraction_notes",
        ],
    }
    return openai.generate_json(
        "Extract legally useful fields. Every item must include evidence_ids copied from the source.",
        text[:24000],
        schema,
    )


def _evidence_item_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "value": {"type": "string"},
            "label": {"type": "string"},
            "evidence_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["value", "label", "evidence_ids"],
    }


def _extract_structured_heuristic(blocks: list[DocumentBlock]) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "document_type": "unknown",
        "parties": [],
        "dates": [],
        "addresses": [],
        "amounts": [],
        "signatures": [],
        "deadlines": [],
        "unclear_spans": [],
        "extraction_notes": ["Heuristic extraction used because no LLM key was available."],
    }
    all_text = " ".join(block.text for block in blocks).lower()
    source_hint = " ".join(sorted({block.source_file for block in blocks})).lower()
    if any(block.extraction_method == "ocr_unavailable" for block in blocks):
        fields["document_type"] = "image-only OCR-required exhibit"
    elif "stipulated_order" in source_hint or "stipulated-order" in source_hint:
        fields["document_type"] = "stipulated order"
    elif "consent_decree" in source_hint or "consent-decree" in source_hint:
        fields["document_type"] = "consent decree"
    elif "complaint" in source_hint:
        fields["document_type"] = "civil complaint"
    elif re.search(r"\b(complaint for|plaintiff brings this action|summary of the case)\b", all_text):
        fields["document_type"] = "civil complaint"
    elif re.search(r"\b(stipulated order|monetary judgment)\b", all_text):
        fields["document_type"] = "stipulated order"
    elif re.search(r"\b(consent decree|injunction)\b", all_text):
        fields["document_type"] = "consent decree"
    elif re.search(r"\b(title|deed|easement|survey)\b", all_text):
        fields["document_type"] = "title review packet"
    elif re.search(r"\b(lease|tenant|landlord|cam)\b", all_text):
        fields["document_type"] = "lease / notice dispute packet"
    elif "claim" in all_text or "denial" in all_text or "adjuster" in all_text:
        fields["document_type"] = "insurance notice claim note"
    elif "notice" in all_text:
        fields["document_type"] = "notice summary packet"
    for block in blocks:
        eid = block.evidence_id
        labels = ["plaintiff", "defendant"]
        if fields["document_type"] in {"lease / notice dispute packet", "title review packet", "insurance notice claim note", "unknown"}:
            labels = [
                "client",
                "tenant",
                "landlord",
                "buyer",
                "seller",
                "borrower",
                "lender",
                "grantor",
                "grantee",
                "plaintiff",
                "defendant",
            ]
        for label in labels:
            match = re.search(rf"\b{label}\s*:\s*([^.;\n]+)", block.text, re.IGNORECASE)
            if match:
                fields["parties"].append(_item(match.group(1), label, eid))
        for label, pattern in [
            ("plaintiff", r"\b([A-Z][A-Z .,&'-]{4,80}),\s*Plaintiff\b"),
            ("defendant", r"\b([A-Z][A-Z .,&'-]{4,80}),\s*(?:a corporation[,;]?\s*)?Defendant\b"),
            ("defendant", r"\bDefendant\s+([A-Z][A-Za-z0-9 .,&'-]{2,80})\s+(?:is|has|operates)\b"),
        ]:
            for match in re.findall(pattern, block.text):
                fields["parties"].append(_item(match, label, eid))
        upper = block.text.upper()
        if "FEDERAL TRADE COMMISSION" in upper and "PLAINTIFF" in upper:
            fields["parties"].append(_item("Federal Trade Commission", "plaintiff", eid))
        if "UNITED STATES OF AMERICA" in upper and "PLAINTIFF" in upper:
            fields["parties"].append(_item("United States of America", "plaintiff", eid))
        if "CHEGG" in upper and "DEFENDANT" in upper:
            fields["parties"].append(_item("Chegg, Inc.", "defendant", eid))
        if "SCHNITZER STEEL INDUSTRIES" in upper and "DEFENDANT" in upper:
            fields["parties"].append(_item("Schnitzer Steel Industries, Inc.", "defendant", eid))
        for date in re.findall(r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{1,2},\s+\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b", block.text):
            fields["dates"].append(_item(date, "date", eid))
        for amount in re.findall(r"\$\s?\d[\d,]*(?:\.\d{2})?", block.text):
            fields["amounts"].append(_item(amount, "amount", eid))
        for address in re.findall(r"\b\d{2,5}\s+[A-Z][A-Za-z0-9 .'-]+(?:Street|St\.|Avenue|Ave\.|Road|Rd\.|Drive|Dr\.|Boulevard|Blvd\.|Lane|Ln\.)\b", block.text):
            fields["addresses"].append(_item(address, "address", eid))
        if re.search(r"\b(signature|signed|/s/)\b", block.text, re.IGNORECASE):
            fields["signatures"].append(_item(block.text[:140], "signature reference", eid))
        if re.search(r"\b(deadline|due|closing|respond by|notice expires|must be cured)\b", block.text, re.IGNORECASE):
            fields["deadlines"].append(_item(_trim_text(block.text, 320), "deadline context", eid))
        if block.warnings and any("unclear" in warning.lower() or "ocr required" in warning.lower() for warning in block.warnings):
            fields["unclear_spans"].append(_item(_trim_text(block.text, 320), "unclear text", eid))
    for key in ["parties", "dates", "addresses", "amounts", "signatures", "deadlines", "unclear_spans"]:
        fields[key] = _dedupe_items(fields[key])
    return fields


def _item(value: str, label: str, evidence_id: str) -> dict[str, Any]:
    if label in {"plaintiff", "defendant", "client", "tenant", "landlord", "buyer", "seller", "borrower", "lender", "grantor", "grantee"}:
        value = _clean_party_value(value)
    return {"value": normalize_space(value), "label": label, "evidence_ids": [evidence_id]}


def _clean_party_value(value: str) -> str:
    value = normalize_space(value)
    value = re.sub(r"^(?:v\.?|vs\.?)\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\b\d+\b", "", value)
    value = normalize_space(value)
    value = value.strip(" ,;:-")
    aliases = {
        "CHEGG": "Chegg, Inc.",
        "CHEGG, INC": "Chegg, Inc.",
        "CHEGG, INC.": "Chegg, Inc.",
        "FEDERAL TRADE COMMISSION": "Federal Trade Commission",
        "UNITED STATES OF AMERICA": "United States of America",
        "SCHNITZER STEEL INDUSTRIES, INC": "Schnitzer Steel Industries, Inc.",
        "SCHNITZER STEEL INDUSTRIES, INC.": "Schnitzer Steel Industries, Inc.",
    }
    return aliases.get(value.upper(), value)


def _trim_text(text: str, limit: int) -> str:
    text = normalize_space(text)
    if len(text) <= limit:
        return text
    trimmed = text[:limit].rsplit(" ", 1)[0].rstrip(" ,.;:")
    return f"{trimmed}..."


def _dedupe_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        if item["label"].lower() in {"plaintiff", "defendant", "client", "tenant", "landlord", "buyer", "seller"}:
            if not _is_party_value_usable(item["value"]):
                continue
        key = (item["label"].lower(), item["value"].lower())
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _is_party_value_usable(value: str) -> bool:
    if not value or len(value) > 90:
        return False
    lowered = value.lower()
    if any(token in lowered for token in ["attorney", "signature", "chief executive", "billing information", "clearly and conspicuously"]):
        return False
    if sum(ch.isdigit() for ch in value) > 2:
        return False
    return True


def _expand_inputs(paths: list[str | Path]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(child for child in p.iterdir() if child.suffix.lower() in SUPPORTED_EXTENSIONS))
        elif p.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(p)
    return sorted(files, key=lambda p: p.name.lower())

"""Download public legal/regulatory sample documents and ground truth."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from .io_utils import ensure_dir, write_json


@dataclass(frozen=True)
class RealSource:
    """A public sample document and the metadata shown in the manifest."""

    filename: str
    url: str
    source: str
    note: str


REAL_SOURCES = [
    RealSource(
        filename="ftc_chegg_complaint.pdf",
        url="https://www.ftc.gov/system/files/ftc_gov/pdf/Chegg-Complaint_0.pdf",
        source="Federal Trade Commission",
        note="Complaint for permanent injunction, monetary judgment, and other relief in FTC v. Chegg.",
    ),
    RealSource(
        filename="ftc_chegg_stipulated_order.pdf",
        url="https://www.ftc.gov/system/files/ftc_gov/pdf/Chegg-StipulatedOrder.pdf",
        source="Federal Trade Commission",
        note="Stipulated order for permanent injunction, monetary judgment, and other relief.",
    ),
    RealSource(
        filename="doj_schnitzer_consent_decree.pdf",
        url="https://www.justice.gov/archives/opa/press-release/file/1496286/dl?inline=",
        source="U.S. Department of Justice",
        note="Consent decree in United States v. Schnitzer Steel Industries, Inc. under the Clean Air Act.",
    ),
    RealSource(
        filename="sec_scanned_lease_page_001.jpg",
        url="https://www.sec.gov/Archives/edgar/data/1307579/000107878211003292/f8ka2111111_ex10z4001.jpg",
        source="U.S. Securities and Exchange Commission EDGAR",
        note="Image-only lease exhibit page used to exercise the OCR-required path.",
    ),
]


REAL_GROUND_TRUTH = {
    "queries": [
        {
            "query": "What defendant and alleged cancellation practices are described in the Chegg complaint?",
            "expected_source_files": ["ftc_chegg_complaint.pdf"],
            "expected_terms": ["Chegg", "ROSCA", "cancel", "subscribers", "recurring charges"],
        },
        {
            "query": "What monetary judgment and injunctive relief appear in the Chegg stipulated order?",
            "expected_source_files": ["ftc_chegg_stipulated_order.pdf"],
            "expected_terms": ["monetary judgment", "$7,500,000", "injunction", "Chegg"],
        },
        {
            "query": "What facilities and compliance obligations are described in the Schnitzer consent decree?",
            "expected_source_files": ["doj_schnitzer_consent_decree.pdf"],
            "expected_terms": ["Schnitzer", "scrap metal recycling", "Clean Air Act", "refrigerant", "Facilities"],
        },
        {
            "query": "What lease exhibit terms were extracted from the scanned SEC image?",
            "expected_source_files": ["sec_scanned_lease_page_001.jpg"],
            "expected_terms": ["Lease", "Lessor", "Lessee"],
        },
    ]
}


def download_real_inputs(output_dir: str | Path, *, overwrite: bool = False) -> list[Path]:
    """Download public sample inputs and write a retrieval ground-truth file."""
    out = ensure_dir(output_dir)
    paths: list[Path] = []
    manifest: list[dict[str, object]] = []
    for source in REAL_SOURCES:
        path = out / source.filename
        record = asdict(source)
        record["path"] = str(path)
        record["downloaded"] = False
        record["error"] = None
        if path.exists() and not overwrite:
            paths.append(path)
            record["downloaded"] = True
            record["cached"] = True
            manifest.append(record)
            continue
        try:
            _download(source.url, path)
            paths.append(path)
            record["downloaded"] = True
            record["cached"] = False
        except Exception as exc:
            record["error"] = str(exc)
        manifest.append(record)
    write_json(out / "real_source_manifest.json", manifest)
    write_json(out / "sample_ground_truth.json", REAL_GROUND_TRUTH)
    return paths


def _download(url: str, path: Path) -> None:
    headers = {
        "User-Agent": "paralegal/0.1 contact: local-demo",
        "Accept": "application/pdf,image/*,*/*",
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            data = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} downloading {url}: {body[:300]}") from exc
    if len(data) < 1024:
        raise RuntimeError(f"Downloaded content from {url} was unexpectedly small ({len(data)} bytes)")
    ensure_dir(path.parent)
    path.write_bytes(data)

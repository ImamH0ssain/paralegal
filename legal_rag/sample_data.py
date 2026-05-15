"""Synthetic legal-style sample packets used for offline demos and tests."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

from .io_utils import ensure_dir, write_json


SAMPLES: dict[str, list[list[str]]] = {
    "clean_case_packet.pdf": [
        [
            "Notice Packet - Lease Cure Review",
            "Client: Northstar Bistro LLC.",
            "Landlord: Harbor Point Properties LP.",
            "Tenant: Northstar Bistro LLC.",
            "Premises: 118 Market Street, Suite 4, Boston.",
            "Lease date: March 3, 2023.",
            "Notice date: April 22, 2026.",
            "The landlord alleges unpaid common-area maintenance charges of $18,450.00.",
            "The notice states the tenant must cure within ten calendar days after service.",
            "Signed: Maria Voss, Property Manager, Harbor Point Properties LP.",
        ],
        [
            "Operator Notes",
            "The tenant disputes the CAM balance and says a March 2026 wire was not credited.",
            "Payment ledger shows a $7,500.00 wire on March 31, 2026 labeled 'CAM partial'.",
            "Email from landlord counsel dated April 25, 2026 says lockout will not occur before May 6, 2026.",
            "Open item: confirm whether the lease requires business-day or calendar-day cure periods.",
        ],
    ],
    "handwritten_notice_note.pdf": [
        [
            "Handwritten Intake Note - Partially Legible",
            "Client: Avalon Repair Group.",
            "Opposing party: Metro Surety Company.",
            "Claim number: MS-7719-26.",
            "Loss location: 44 Cedar Avenue, Trenton.",
            "Date of loss: 02/18/2026.",
            "Adjuster call note: denial letter maybe sent 03/04/2026.",
            "The note says 'respond by 3/18?' but the last digit is unclear.",
            "[illegible] reference to roof invoice amount, possibly $12,900.00.",
        ],
        [
            "Follow-up Items",
            "Signature line appears incomplete.",
            "The scanned margin includes smudged handwriting next to 'proof of loss'.",
            "Need original denial letter and a clean copy of the contractor estimate.",
        ],
    ],
    "noisy_title_packet.pdf": [
        [
            "Title Review Packet - Low Quality Scan",
            "Buyer: Elara Holdings Inc.",
            "Seller: Wynn Family Trust.",
            "Property: 705 Pine Road, Lot 12, King County.",
            "Proposed closing: May 15, 2026.",
            "Deed reference: Book 4412 Page 117.",
            "Exception 7 references an easement recorded on 09/14/1998.",
            "Exception 9 appears to contain a handwritten release note but it is unclear.",
        ],
        [
            "Title Notes",
            "Tax certificate shows $4,210.55 due for the first half of 2026.",
            "Survey note says fence encroaches approximately 1.8 feet over the north boundary.",
            "Seller counsel email dated May 8, 2026 states payoff letter will follow.",
            "Open item: obtain recorded easement image and verify release status.",
        ],
    ],
}


GROUND_TRUTH = {
    "queries": [
        {
            "query": "What is the lease cure deadline and disputed amount?",
            "expected_terms": ["cure", "$18,450.00", "ten calendar days", "May 6, 2026"],
        },
        {
            "query": "What unclear facts need follow-up for the notice claim?",
            "expected_terms": ["illegible", "respond by 3/18", "denial letter", "proof of loss"],
        },
        {
            "query": "What title exceptions or closing risks should be flagged?",
            "expected_terms": ["Exception 7", "easement", "Exception 9", "fence encroaches", "payoff letter"],
        },
        {
            "query": "Which parties and properties are involved?",
            "expected_terms": ["Northstar Bistro", "Harbor Point", "Elara Holdings", "Wynn Family Trust", "705 Pine Road"],
        },
    ]
}


def create_sample_inputs(output_dir: str | Path) -> list[Path]:
    """Create small synthetic PDFs and a matching retrieval ground-truth file."""
    out = ensure_dir(output_dir)
    paths: list[Path] = []
    for filename, pages in SAMPLES.items():
        path = out / filename
        _write_pdf(path, pages, noisy="noisy" in filename or "handwritten" in filename)
        paths.append(path)
    write_json(out / "sample_ground_truth.json", GROUND_TRUTH)
    return paths


def _write_pdf(path: Path, pages: list[list[str]], *, noisy: bool = False) -> None:
    c = canvas.Canvas(str(path), pagesize=LETTER)
    width, height = LETTER
    for page_num, lines in enumerate(pages, start=1):
        if noisy:
            c.setFillColor(colors.lightgrey)
            for y in range(80, int(height) - 80, 54):
                c.line(40, y, width - 40, y + 7)
            c.setFillColor(colors.grey)
            c.setFont("Courier", 9)
            c.drawString(52, height - 36, "SCAN ARTIFACT - LOW CONTRAST SOURCE")
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(64, height - 72, lines[0])
        c.setFont("Helvetica", 10)
        y = height - 110
        for line in lines[1:]:
            for wrapped in _wrap(line, width=92):
                c.drawString(72, y, wrapped)
                y -= 18
            y -= 6
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.darkgrey)
        c.drawRightString(width - 64, 40, f"Sample packet page {page_num}")
        c.showPage()
    c.save()


def _wrap(text: str, width: int) -> Iterable[str]:
    words = text.split()
    line: list[str] = []
    current = 0
    for word in words:
        extra = 1 if line else 0
        if current + len(word) + extra > width:
            yield " ".join(line)
            line = [word]
            current = len(word)
        else:
            line.append(word)
            current += len(word) + extra
    if line:
        yield " ".join(line)

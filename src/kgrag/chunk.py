"""Split sections into extraction-sized chunks with deterministic ids.

The chunk id is the load-bearing decision of Phase 1. Phase 2 embeds these same
chunks into pgvector and joins on this key; Phase 4 cites it. So it is a hash of
(accession, section_path, ordinal) rather than a UUID or a row number — the same
document chunked next year produces byte-identical ids, which is what makes the
whole pipeline idempotent instead of merely re-runnable.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from typing import Any

from . import jsonl
from .config import CHUNKS, DOCS

#: ~1000 tokens of text per chunk. Big enough that a relation and its evidence span
#: usually survive intact, small enough that the model does not start summarising.
TARGET_CHARS = 4000
OVERLAP_CHARS = 800
MIN_CHARS = 300

#: DEF 14A prefilter. A proxy runs ~330k chars and the great majority of it is
#: compensation discussion and beneficial-ownership tables, which contain no extractable
#: relation. Sending them costs tokens and, worse, dilutes the chunks that do carry board
#: seats with hundreds of near-identical pages about equity incentive plans.
#:
#: A first pass matching any governance word kept 2126 of 2126 proxy chunks — "director"
#: and "officer" appear on nearly every page of a proxy, so the filter did nothing.
#: Requiring an actual statement of board *service* keeps 971. The dropped chunks are
#: compensation tables; the kept ones are director bios and nominee tables, which is
#: where directors' seats at OTHER companies live.
GOVERNANCE = re.compile(
    r"(?:serve[sd]?|has served|currently serves)[^.]{0,60}"
    r"(?:on the board|as a director|as director|board of directors)"
    r"|\bdirector of\b|\bboard member of\b|\bboards? of\b"
    r"|director nominee|election of directors|continuing directors",
    re.IGNORECASE,
)


def chunk_id(accession: str, section_path: str, ordinal: int) -> str:
    return hashlib.sha256(f"{accession}|{section_path}|{ordinal}".encode()).hexdigest()[:16]


def _split(text: str) -> Iterator[str]:
    """Accumulate paragraphs up to TARGET_CHARS, carrying OVERLAP_CHARS between chunks.

    Splitting on paragraph boundaries rather than a fixed stride keeps sentences whole,
    which matters because the model must quote a verbatim evidence span and a span cut
    in half fails the substring check for no good reason.
    """
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    buffer: list[str] = []
    size = 0

    for para in paragraphs:
        # A single oversized paragraph (dense tables, run-on risk factors) is hard-split.
        while len(para) > TARGET_CHARS:
            head, para = para[:TARGET_CHARS], para[TARGET_CHARS - OVERLAP_CHARS :]
            if buffer:
                yield "\n\n".join(buffer)
                buffer, size = [], 0
            yield head
        if size + len(para) > TARGET_CHARS and buffer:
            chunk = "\n\n".join(buffer)
            yield chunk
            tail = chunk[-OVERLAP_CHARS:]
            buffer, size = [tail], len(tail)
        buffer.append(para)
        size += len(para) + 2

    if buffer:
        yield "\n\n".join(buffer)


def _keep(doc: dict[str, Any], text: str) -> bool:
    if len(text) < MIN_CHARS:
        return False
    if doc["section_path"] != "DEF 14A":
        return True
    if not GOVERNANCE.search(text):
        return False
    # Compensation and ownership tables match "officer" but are mostly digits.
    dense = sum(c.isdigit() for c in text) / max(sum(not c.isspace() for c in text), 1)
    return dense < 0.25


def run() -> None:
    docs = list(jsonl.read(DOCS))
    if not docs:
        raise SystemExit("data/docs.jsonl is empty — run `kgrag fetch` first.")

    rows, skipped = [], 0
    for doc in docs:
        for ordinal, text in enumerate(_split(doc["text"])):
            if not _keep(doc, text):
                skipped += 1
                continue
            rows.append(
                {
                    "chunk_id": chunk_id(doc["accession"], doc["section_path"], ordinal),
                    "ticker": doc["ticker"],
                    "company": doc["company"],
                    "cik": doc["cik"],
                    "form": doc["form"],
                    "accession": doc["accession"],
                    "filing_date": doc["filing_date"],
                    "section_path": doc["section_path"],
                    "text": text,
                }
            )

    assert len({r["chunk_id"] for r in rows}) == len(rows), "chunk id collision"
    n = jsonl.write(CHUNKS, rows)
    chars = sum(len(r["text"]) for r in rows)
    print(f"chunks.jsonl: {n} chunks, {chars:,} chars (~{chars // 4:,} tokens)")
    print(f"prefiltered out: {skipped} chunks")

    by_form: dict[str, int] = {}
    for r in rows:
        by_form[r["section_path"].split("/")[0]] = by_form.get(r["section_path"].split("/")[0], 0) + 1
    for form, count in sorted(by_form.items()):
        print(f"  {form:10} {count:5}")

"""Pull the corpus from SEC EDGAR into data/docs.jsonl.

Each row is one *section* of one filing, not one filing — the section is the natural
unit here because it decides which relations are worth looking for and it gives every
downstream chunk a stable `section_path`.

Reproducibility: EDGAR keeps filing, so "the latest 10-K" is a moving target and a
rerun six months from now would silently ingest a different corpus. The accession
numbers resolved on the first run are pinned in data/manifest.jsonl and reused
thereafter, so the graph is rebuildable from the same documents forever.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

from edgar import Company, set_identity

from . import jsonl
from .config import DOCS, MANIFEST, require

#: 24 US semiconductor filers. Dense supply chains, heavily overlapping boards, and
#: a decade of real acquisitions between them. GlobalFoundries and ASML are excluded:
#: foreign private issuers file 20-F, which has different section structure.
UNIVERSE: tuple[str, ...] = (
    "NVDA", "AMD", "INTC", "AVGO", "QCOM", "MU", "AMAT", "LRCX",
    "KLAC", "TXN", "ADI", "MRVL", "NXPI", "ON", "MCHP", "SWKS",
    "QRVO", "TER", "ENTG", "WOLF", "MPWR", "CRUS", "SLAB", "ALGM",
)

#: 10-K items worth the tokens. Item 1 names competitors, suppliers, and products;
#: 1A carries the risk exposures; 3 the legal proceedings; 7 the acquisitions and
#: segment geography. Items 8 and 15 are financial statements — XBRL already has
#: those structured, and extracting them with an LLM would be strictly worse.
TENK_ITEMS = ("Item 1", "Item 1A", "Item 3", "Item 7")

#: 8-K items that carry edges. 1.01 material agreements, 2.01 completed acquisitions,
#: 5.02 director and officer changes. The rest are earnings pressers and exhibits.
EIGHTK_ITEMS = ("Item 1.01", "Item 2.01", "Item 5.02")

EIGHTK_LOOKBACK = 8  # most recent 8-Ks scanned per company


def _clean(text: str | None) -> str:
    """EDGAR text is HTML-derived: ragged whitespace and page furniture."""
    if not text:
        return ""
    text = re.sub(r"\n?Table of Contents\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _doc(filing, section_path: str, text: str, ticker: str) -> dict[str, Any] | None:
    text = _clean(text)
    if len(text) < 200:  # cross-references and stub sections carry no extractable facts
        return None
    return {
        "ticker": ticker,
        # The filer's own name. 10-Ks say "we" and "our Company" on every page, so the
        # extraction prompt has to be told who the filer is or first-person references
        # become unresolvable mentions.
        "company": filing.company,
        "cik": int(filing.cik),
        "form": filing.form,
        "accession": filing.accession_no,
        "filing_date": str(filing.filing_date),
        "section_path": section_path,
        "text": text,
    }


def _sections(filing, ticker: str) -> Iterator[dict[str, Any]]:
    form = filing.form
    if form == "10-K":
        obj = filing.obj()
        for item in TENK_ITEMS:
            if doc := _doc(filing, f"10-K/{item}", obj[item], ticker):
                yield doc
        # Exhibit 21 is a plain list of subsidiaries: the highest-precision, lowest-cost
        # edges in the whole corpus. It is an attachment, not an item.
        for att in filing.attachments:
            if str(att.document_type).startswith("EX-21"):
                if doc := _doc(filing, "10-K/EX-21", att.text(), ticker):
                    yield doc
                break
    elif form == "DEF 14A":
        # The proxy is ~330k chars, and most of it is compensation tables that yield
        # nothing. chunk.py prefilters; sending the whole thing here keeps fetch dumb.
        if doc := _doc(filing, "DEF 14A", filing.text(), ticker):
            yield doc
    elif form == "8-K":
        obj = filing.obj()
        for item in getattr(obj, "items", []) or []:
            if item in EIGHTK_ITEMS and (doc := _doc(filing, f"8-K/{item}", obj[item], ticker)):
                yield doc


def _pin(ticker: str) -> Iterator[Any]:
    """Resolve which filings this company contributes. Called once; then pinned.

    `amendments=False` is load-bearing. EDGAR form matching is prefix-based, so "10-K"
    also matches "10-K/A", and an amendment contains *only* the items it amends — AMD's
    most recent 10-K/A carries Items 7 and 15, Skyworks' carries Items 10 through 15.
    Taking the latest match silently cost those two companies their entire Business and
    Risk Factors sections, which reads as a parser bug and is not one.
    """
    company = Company(ticker)
    for form, n in (("10-K", 1), ("DEF 14A", 1), ("8-K", EIGHTK_LOOKBACK)):
        yield from company.get_filings(form=form, amendments=False).head(n)


def run(tickers: tuple[str, ...] = UNIVERSE) -> None:
    set_identity(require("SEC_USER_AGENT"))

    pinned = {r["ticker"]: r["accessions"] for r in jsonl.read(MANIFEST)}
    done = {r["ticker"] for r in jsonl.read(DOCS)}

    for ticker in tickers:
        if ticker in done:
            print(f"{ticker:6} cached")
            continue
        try:
            filings = list(_pin(ticker))
            if accessions := pinned.get(ticker):
                filings = [f for f in filings if f.accession_no in accessions]
            docs = [d for f in filings for d in _sections(f, ticker)]
        except Exception as exc:  # noqa: BLE001 — one bad filer must not kill a 24-company run
            print(f"{ticker:6} FAILED {type(exc).__name__}: {exc}")
            continue

        jsonl.append(DOCS, docs)
        if ticker not in pinned:
            jsonl.append(MANIFEST, [{"ticker": ticker, "accessions": [f.accession_no for f in filings]}])
        chars = sum(len(d["text"]) for d in docs)
        print(f"{ticker:6} {len(docs):3} sections  {chars:>9,} chars")

    total = list(jsonl.read(DOCS))
    print(f"\ndocs.jsonl: {len(total)} sections, {sum(len(d['text']) for d in total):,} chars")

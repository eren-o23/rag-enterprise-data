"""Paths and environment. Small on purpose — everything else takes paths as arguments."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CACHE = ROOT / "cache"
EVAL = ROOT / "eval"

DOCS = DATA / "docs.jsonl"
MANIFEST = DATA / "manifest.jsonl"
CHUNKS = DATA / "chunks.jsonl"
EXTRACTIONS = DATA / "extractions.jsonl"
ENTITIES = DATA / "entities.jsonl"
FAILURES = DATA / "failures.jsonl"
OVERRIDES = DATA / "overrides.jsonl"
#: Aliases the filings declare about themselves, mined by `kgrag mine-pairs`. Kept apart
#: from OVERRIDES so the hand-decided count stays a quality signal (see docs/decisions.md).
ALIASES = DATA / "aliases.jsonl"


def require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is not set. Copy .env.example to .env and fill it in.")
    return value

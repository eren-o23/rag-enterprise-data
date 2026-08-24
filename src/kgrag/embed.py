"""`kgrag embed` — the same chunks the graph was built from, as vectors in pgvector.

The point of this stage is the join key, not the vectors. Every row is keyed on the
`chunk_id` minted in chunk.py, which is the same value carried in every Neo4j edge's
`chunk_ids` list. That shared key is what lets Phase 3 cross from a graph path back to
the passage that justified it, and from a retrieved passage into the graph neighbourhood
around it. Two stores, one identifier, no mapping table.

Three embedding widths are written rather than one. `qwen3-embedding-8b` emits 4096 dims
natively and pgvector will not HNSW-index anything above 2000, so the production column
has to be a truncation -- and `kgrag recall` measures what that truncation costs instead
of asserting it is free.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Iterator
from typing import Any

import psycopg

from . import fireworks, jsonl
from .config import CHUNKS, ENTITIES, ROOT, require

#: Native width first so a budget stop leaves the cheap widths done rather than the
#: expensive one half-written.
WIDTHS = (1024, 2000, 4096)

#: Embedding requests are paced to 9 RPM, so batch size sets wall-clock time almost
#: entirely: 2,743 chunks is 43 requests at 64, or ~5 minutes per width.
BATCH = 64

#: How many rows go into one UPDATE round trip. Unrelated to BATCH.
WRITE_BATCH = 200


def connect() -> psycopg.Connection:
    return psycopg.connect(require("PG_DSN"))


def apply_sql(conn: psycopg.Connection, name: str) -> None:
    """Apply a versioned .sql file, the way load.apply_schema applies the Cypher one.

    psycopg3 executes multi-statement SQL in a single execute() as long as no parameters
    are bound, so this needs none of the comment-stripping the Cypher version does.
    """
    conn.execute((ROOT / "sql" / name).read_text())
    conn.commit()


def chunk_entities(entities: Iterable[dict[str, Any]] | None = None) -> dict[str, list[str]]:
    """chunk_id -> canonical entity ids mentioned in it.

    entities.jsonl stores the forward direction (entity -> mention_chunks) because that is
    what resolution produces; there is no stored reverse index, so invert it here. Cheap:
    4,496 entities over 2,741 chunks. Takes an iterable so it is testable without the
    gitignored data file.
    """
    index: dict[str, list[str]] = {}
    for entity in jsonl.read(ENTITIES) if entities is None else entities:
        for chunk_id in entity["mention_chunks"]:
            index.setdefault(chunk_id, []).append(entity["canonical_id"])
    return {k: sorted(v) for k, v in index.items()}


UPSERT = """
INSERT INTO chunks (chunk_id, ticker, company, form, accession, section_path,
                    filing_date, text, entity_ids)
VALUES (%(chunk_id)s, %(ticker)s, %(company)s, %(form)s, %(accession)s,
        %(section_path)s, %(filing_date)s, %(text)s, %(entity_ids)s)
ON CONFLICT (chunk_id) DO UPDATE SET
    ticker = EXCLUDED.ticker, company = EXCLUDED.company, form = EXCLUDED.form,
    accession = EXCLUDED.accession, section_path = EXCLUDED.section_path,
    filing_date = EXCLUDED.filing_date, text = EXCLUDED.text,
    entity_ids = EXCLUDED.entity_ids
"""


def _batched(rows: list[Any], size: int) -> Iterator[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def run(assume_yes: bool = False, limit: int | None = None) -> None:
    chunks = list(jsonl.read(CHUNKS))
    if not chunks:
        raise SystemExit("data/chunks.jsonl is empty — run `kgrag chunk` first.")
    if limit:
        chunks = chunks[:limit]

    # All 2,743 chunks, not the 2,741 in extractions.jsonl. The 2 quarantined chunks have
    # no graph entities but are still real passages and must stay retrievable by vector.
    entities = chunk_entities()
    print(f"{len(chunks)} chunks, {len(entities)} of them mention at least one entity")

    with connect() as conn:
        apply_sql(conn, "schema.sql")

        rows = [
            {**{k: c[k] for k in
                ("chunk_id", "ticker", "company", "form", "accession", "section_path",
                 "filing_date", "text")},
             "entity_ids": entities.get(c["chunk_id"], [])}
            for c in chunks
        ]
        with conn.cursor() as cur:
            cur.executemany(UPSERT, rows)
        conn.commit()
        print(f"upserted {len(rows)} metadata rows (entity_ids refreshed, vectors untouched)")

        todo = {w: _pending(conn, w, {c["chunk_id"] for c in chunks}) for w in WIDTHS}
        outstanding = sum(len(v) for v in todo.values())
        if not outstanding:
            print("every width already embedded — nothing to do")
            _summarise(conn)
            return

        # Cost is projectable without a probe call: embedding is priced on input tokens
        # only, and the input is text we already have. extract.py has to probe because
        # output length is unknowable in advance; here it is simply not.
        by_id = {c["chunk_id"]: c for c in chunks}
        tokens = sum(len(by_id[i]["text"]) for w in WIDTHS for i in todo[w]) // 4
        rate = fireworks.PRICES[fireworks.EMBED_MODEL][0]
        projected = tokens / 1e6 * rate
        print(
            f"\n{outstanding} chunk-widths to embed (~{tokens:,} tokens), "
            f"projected ${projected:.2f}"
        )
        for width in WIDTHS:
            print(f"  {width:>4} dims  {len(todo[width]):>5} pending")
        if not assume_yes:
            print(f"\nRerun with --yes to proceed (and --budget above ${projected:.2f}).")
            return

        for width in WIDTHS:
            _embed_width(conn, width, todo[width], by_id)
        apply_sql(conn, "index.sql")
        print("\nHNSW indexes built on emb_1024 and emb_2000 (emb_4096 is not indexable)")
        _summarise(conn)


def _pending(conn: psycopg.Connection, width: int, wanted: set[str]) -> list[str]:
    """chunk_ids that still have no vector at this width — how the stage resumes."""
    rows = conn.execute(
        f"SELECT chunk_id FROM chunks WHERE emb_{width} IS NULL"
    ).fetchall()
    return [r[0] for r in rows if r[0] in wanted]


def _embed_width(
    conn: psycopg.Connection, width: int, pending: list[str], by_id: dict[str, Any]
) -> None:
    if not pending:
        print(f"\n{width} dims: already complete")
        return
    print(f"\n{width} dims: {len(pending)} chunks")
    done = 0
    for group in _batched(pending, BATCH):
        vectors = fireworks.embed(
            [by_id[i]["text"] for i in group], dimensions=width, use_cache=False
        )
        # Written per API batch rather than at the end: a budget stop or a dropped
        # connection then costs the batch in flight, not the whole width.
        with conn.cursor() as cur:
            cur.executemany(
                f"UPDATE chunks SET emb_{width} = %s WHERE chunk_id = %s",
                [(str(v), i) for i, v in zip(group, vectors)],
            )
        conn.commit()
        done += len(group)
        if done % (BATCH * 8) == 0 or done == len(pending):
            now = dt.datetime.now().strftime("%H:%M:%S")
            print(f"  [{now}] {done}/{len(pending)}  ${fireworks.METER.usd:.3f}")


def _summarise(conn: psycopg.Connection) -> None:
    total = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    print(f"\nchunks table: {total:,} rows")
    for width in WIDTHS:
        n = conn.execute(
            f"SELECT count(emb_{width}) FROM chunks"
        ).fetchone()[0]
        print(f"  emb_{width:<5} {n:>6,} embedded")
    tagged = conn.execute(
        "SELECT count(*) FROM chunks WHERE cardinality(entity_ids) > 0"
    ).fetchone()[0]
    print(f"  entity_ids  {tagged:>6,} chunks tagged")

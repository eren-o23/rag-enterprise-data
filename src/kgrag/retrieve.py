"""`kgrag recall` — measure retrieval before anything is built on top of it.

Two different measurements live here, and conflating them is the easy mistake:

(a) **ANN recall against exact search.** Purely an index-quality question: does the HNSW
    graph return what a brute-force scan would? Needs no labels, because exact search IS
    the answer key. Swept across `ef_search`.

(b) **Retrieval recall@k against eval/questions.jsonl.** A retrieval-quality question:
    when someone asks something, do the chunks that answer it come back? Needs labels,
    and this is the number that decides whether Phase 3's router can trust the vector path.

They are reported separately because a system can ace (a) and fail (b) -- a perfectly
faithful index over embeddings that do not capture the question is still useless.

(b) runs exact for every width, so the 1024/2000/4096 comparison is not confounded by
4096 being the one width pgvector cannot index.
"""

from __future__ import annotations

import statistics
import time
from typing import Any

import psycopg

from . import fireworks, jsonl
from .embed import WIDTHS, connect
from .questions import QUESTIONS

INDEXED = (1024, 2000)
EF_SEARCH = (10, 40, 100, 200, 400)
KS = (1, 5, 10, 20)


def _topk(conn: psycopg.Connection, width: int, vector: list[float], k: int, exact: bool) -> list[str]:
    with conn.cursor() as cur:
        # enable_indexscan=off is what forces the exact baseline: pgvector's HNSW is an
        # index scan, so switching index scans off leaves a full scan with true distances.
        cur.execute("SET LOCAL enable_indexscan = %s", ("off" if exact else "on",))
        cur.execute(
            f"SELECT chunk_id FROM chunks WHERE emb_{width} IS NOT NULL "
            f"ORDER BY emb_{width} <=> %s::vector LIMIT %s",
            (str(vector), k),
        )
        return [r[0] for r in cur.fetchall()]


def ann_vs_exact(conn: psycopg.Connection, samples: int = 60, k: int = 10) -> None:
    """(a) Does the index agree with a brute-force scan, and how fast is each?"""
    print("=" * 74)
    print("(a) ANN recall vs exact search — index quality, no labels needed")
    print("=" * 74)

    # Existing corpus vectors are reused as queries: this measures the index, not the
    # embedding model, so a query drawn from the same distribution is exactly right and
    # costs nothing. (Each query trivially retrieves itself at rank 1, in both arms, so it
    # cancels out of the comparison.)
    rows = conn.execute(
        f"SELECT emb_{INDEXED[0]}, emb_{INDEXED[1]} FROM chunks "
        f"WHERE emb_{INDEXED[0]} IS NOT NULL ORDER BY random() LIMIT %s",
        (samples,),
    ).fetchall()
    queries = {w: [r[i] for r in rows] for i, w in enumerate(INDEXED)}

    for width in INDEXED:
        vectors = queries[width]
        exact, exact_ms = [], []
        for v in vectors:
            start = time.perf_counter()
            exact.append(set(_topk(conn, width, v, k, exact=True)))
            exact_ms.append((time.perf_counter() - start) * 1000)
        print(f"\nemb_{width}  (exact baseline: {statistics.median(exact_ms):.1f} ms median)")
        print(f"  {'ef_search':>10} {'recall@' + str(k):>10} {'median ms':>11} {'p95 ms':>9}")
        for ef in EF_SEARCH:
            conn.execute("SET hnsw.ef_search = %s", (ef,))
            hits, ms = [], []
            for v, gold in zip(vectors, exact):
                start = time.perf_counter()
                got = set(_topk(conn, width, v, k, exact=False))
                ms.append((time.perf_counter() - start) * 1000)
                hits.append(len(got & gold) / len(gold))
            p95 = sorted(ms)[int(len(ms) * 0.95) - 1]
            print(
                f"  {ef:>10} {statistics.mean(hits):>10.3f} "
                f"{statistics.median(ms):>11.1f} {p95:>9.1f}"
            )
    conn.execute("SET hnsw.ef_search = 40")
    print(
        "\n  At 2,743 chunks an exact scan is already fast, so HNSW here is a demonstration\n"
        "  of the technique rather than a necessity. The recall/ef_search curve is the real\n"
        "  result; the latency column mostly shows there is nothing yet to speed up."
    )


def recall_at_k(conn: psycopg.Connection) -> None:
    """(b) Do the chunks that answer a question actually come back?"""
    questions = [q for q in jsonl.read(QUESTIONS) if q["gold_chunk_ids"]]
    refusals = [q for q in jsonl.read(QUESTIONS) if not q["gold_chunk_ids"]]
    if not questions:
        raise SystemExit(f"{QUESTIONS} has no labelled questions — run `kgrag mine-questions`.")

    print("\n" + "=" * 74)
    print("(b) Retrieval recall@k on eval/questions.jsonl — the number that matters")
    print("=" * 74)
    print(
        f"{len(questions)} answerable questions "
        f"({sum(1 for q in questions if q['source'] == 'hand')} hand-written), "
        f"{len(refusals)} out-of-scope.\n"
        "Gold sets come from the graph: every edge carries the chunk_ids whose text\n"
        "justified it, and the evidence-span check means those chunks demonstrably contain\n"
        "the supporting sentence. They are a lower bound, not exhaustive -- other chunks may\n"
        "also answer a question -- so these are floors. The bound applies identically to all\n"
        "three widths, which is what keeps the comparison between them fair.\n"
    )

    for width in WIDTHS:
        vectors = fireworks.embed(
            [q["question"] for q in questions], dimensions=width, use_cache=True
        )
        # Exact for every width: 4096 has no index, and comparing an indexed width against
        # an unindexed one would measure the index, not the embedding.
        results = [_topk(conn, width, v, max(KS), exact=True) for v in vectors]

        print(f"emb_{width}")
        print(f"  {'slice':14} {'n':>4} " + " ".join(f"{'R@' + str(k):>7}" for k in KS))
        for label, subset in _slices(questions):
            if not subset:
                continue
            idx = [questions.index(q) for q in subset]
            scores = []
            for k in KS:
                per_q = [
                    len(set(results[i][:k]) & set(questions[i]["gold_chunk_ids"]))
                    / len(questions[i]["gold_chunk_ids"])
                    for i in idx
                ]
                scores.append(statistics.mean(per_q))
            print(f"  {label:14} {len(subset):>4} " + " ".join(f"{s:>7.3f}" for s in scores))
        print()


def _slices(questions: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    out = [(f"{h}-hop", [q for q in questions if q["hops"] == h]) for h in (1, 2, 3)]
    out.append(("hand-written", [q for q in questions if q["source"] == "hand"]))
    out.append(("all", questions))
    return out


def run() -> None:
    with connect() as conn:
        missing = conn.execute(
            f"SELECT count(*) FROM chunks WHERE emb_{WIDTHS[0]} IS NULL"
        ).fetchone()[0]
        if missing:
            raise SystemExit(f"{missing} chunks have no embedding — run `kgrag embed --yes` first.")
        ann_vs_exact(conn)
        recall_at_k(conn)

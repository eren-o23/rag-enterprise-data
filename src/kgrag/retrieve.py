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
from .embed import PRODUCTION_WIDTH, WIDTHS, connect
from .questions import QUESTIONS

INDEXED = (1024, 2000)
EF_SEARCH = (1, 2, 4, 10, 40, 100, 400)
KS = (1, 5, 10, 20)

#: A catastrophe floor on 1-hop R@10 at the production width, NOT a quality threshold.
#: Measured at 0.586; set to 0.35 deliberately loose. The 1-hop slice is 30 questions, and
#: a paired bootstrap over the (larger) 52-question set already showed that differences of
#: ~0.03 do not survive resampling -- so any threshold tight enough to detect real quality
#: drift would flap on noise instead. What this DOES catch is the class of failure that
#: does not drift, it collapses: querying the wrong embedding column, vectors written
#: against the wrong chunk_ids, or the eval set decoupling from the corpus after a
#: re-chunk. Those land near zero, not near 0.55.
#:
#: It lives here rather than in `kgrag verify` on purpose. verify is a gate and needs only
#: the two databases -- no API key, no network, no spend. Scoring recall means embedding 57
#: questions, and `cache/` is gitignored, so on a fresh clone that would turn the gate into
#: 57 paid Fireworks calls. `kgrag recall` already pays that cost, so the check is free here.
MIN_1HOP_RECALL_AT_10 = 0.35


def topk(conn: psycopg.Connection, width: int, vector: list[float], k: int, exact: bool) -> list[str]:
    with conn.cursor() as cur:
        # Both arms have to be forced, not just one. At 2,743 rows the planner often
        # costs a full scan cheaper than an HNSW probe and picks it even when the index is
        # available -- which silently turns the "ANN" arm into a second exact scan and
        # reports recall 1.000 at every ef_search. The tell was non-monotonicity (ef=4
        # scoring above ef=40) and ANN latencies sitting exactly on the exact baseline.
        # SET takes no bind parameters, hence interpolation of literals we control.
        if exact:
            cur.execute("SET LOCAL enable_indexscan = off")
            cur.execute("SET LOCAL enable_seqscan = on")
        else:
            cur.execute("SET LOCAL enable_indexscan = on")
            cur.execute("SET LOCAL enable_seqscan = off")
        cur.execute(
            f"SELECT chunk_id FROM chunks WHERE emb_{width} IS NOT NULL "
            f"ORDER BY emb_{width} <=> %s::vector LIMIT %s",
            (str(vector), k),
        )
        return [r[0] for r in cur.fetchall()]


def ann_vs_exact(conn: psycopg.Connection, k: int = 10) -> None:
    """(a) Does the index agree with a brute-force scan, and how fast is each?"""
    print("=" * 74)
    print("(a) ANN recall vs exact search — index quality, no labels needed")
    print("=" * 74)

    # The eval questions are the queries, not sampled corpus vectors. A corpus vector used
    # as its own query sits exactly on a graph node and retrieves itself at rank 1, which
    # makes the neighbourhood trivially easy to walk and flatters HNSW. Real question
    # embeddings land off-manifold, which is the case that actually stresses the index.
    questions = [q["question"] for q in jsonl.read(QUESTIONS)]
    print(f"{len(questions)} eval questions as queries, recall@{k} against a full scan\n")

    for width in INDEXED:
        vectors = fireworks.embed(questions, dimensions=width, use_cache=True)
        exact, exact_ms = [], []
        for v in vectors:
            start = time.perf_counter()
            exact.append(set(topk(conn, width, v, k, exact=True)))
            exact_ms.append((time.perf_counter() - start) * 1000)
        print(f"\nemb_{width}  (exact baseline: {statistics.median(exact_ms):.1f} ms median)")
        print(f"  {'ef_search':>10} {'recall@' + str(k):>10} {'median ms':>11} {'p95 ms':>9}")
        for ef in EF_SEARCH:
            conn.execute(f"SET hnsw.ef_search = {int(ef)}")
            hits, ms = [], []
            for v, gold in zip(vectors, exact):
                start = time.perf_counter()
                got = set(topk(conn, width, v, k, exact=False))
                ms.append((time.perf_counter() - start) * 1000)
                hits.append(len(got & gold) / len(gold))
            p95 = sorted(ms)[int(len(ms) * 0.95) - 1]
            print(
                f"  {ef:>10} {statistics.mean(hits):>10.3f} "
                f"{statistics.median(ms):>11.1f} {p95:>9.1f}"
            )
    conn.execute("SET hnsw.ef_search = 40")
    print(
        "\n  ef_search is swept down to 1 because the usual range does not bend at this\n"
        "  scale -- 2,743 vectors is a small graph and HNSW saturates by ef=100.\n"
        "  Both arms are planner-forced: left alone, Postgres costs a full scan cheaper\n"
        "  than an HNSW probe on a table this small and silently answers the ANN arm\n"
        "  exactly, which reports recall 1.000 everywhere and measures nothing.\n"
        "  The honest caveat stands: an exact scan is single-digit milliseconds here, so\n"
        "  the index is a demonstration of the technique rather than a necessity."
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

    per_width: dict[int, list[float]] = {}
    for width in WIDTHS:
        vectors = fireworks.embed(
            [q["question"] for q in questions], dimensions=width, use_cache=True
        )
        # Exact for every width: 4096 has no index, and comparing an indexed width against
        # an unindexed one would measure the index, not the embedding.
        results = [topk(conn, width, v, max(KS), exact=True) for v in vectors]
        per_width[width] = [
            len(set(r[:10]) & set(q["gold_chunk_ids"])) / len(q["gold_chunk_ids"])
            for r, q in zip(results, questions)
        ]

        print(f"emb_{width}")
        print(f"  {'slice':14} {'n':>4} " + " ".join(f"{'R@' + str(k):>7}" for k in KS))
        for label, idx in _slices(questions):
            if not idx:
                continue
            scores = []
            for k in KS:
                per_q = [
                    len(set(results[i][:k]) & set(questions[i]["gold_chunk_ids"]))
                    / len(questions[i]["gold_chunk_ids"])
                    for i in idx
                ]
                scores.append(statistics.mean(per_q))
            print(f"  {label:14} {len(idx):>4} " + " ".join(f"{v:>7.3f}" for v in scores))
        print()

    _width_significance(per_width)
    _check_floor(questions, per_width[PRODUCTION_WIDTH])


def _slices(questions: list[dict[str, Any]]) -> list[tuple[str, list[int]]]:
    """Report slices as index lists, not sublists: `list.index()` on dicts matches by
    value, so it would silently return the wrong row for any two identical questions."""
    # Aggregation questions are their own stratum, not 1-hop questions that happen to need
    # one hop. Folding them into the hop slices would silently move a published Phase 2
    # number and shift the recall floor's baseline underneath it.
    hop = [i for i, q in enumerate(questions) if not q.get("category")]
    out = [
        (f"{h}-hop", [i for i in hop if questions[i]["hops"] == h]) for h in (1, 2, 3)
    ]
    out.append(("aggregation",
                [i for i, q in enumerate(questions) if q.get("category") == "aggregation"]))
    out.append(("hand-written", [i for i, q in enumerate(questions) if q["source"] == "hand"]))
    out.append(("all", list(range(len(questions)))))
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


def _width_significance(per_width: dict[int, list[float]], trials: int = 10000) -> None:
    """Is any width actually better, or is the ranking noise?

    52 questions is a small sample and the widths land within a few points of each other,
    so eyeballing the table would support whichever conclusion was wanted. A paired
    bootstrap over per-question R@10 differences answers it properly: paired because every
    width answers the same questions, so the pairing removes question difficulty as a
    source of variance.
    """
    import random

    print("-" * 74)
    print("width comparison — paired bootstrap on per-question R@10, 95% CI")
    print("-" * 74)
    rng = random.Random(0)  # fixed seed: the published interval must be reproducible
    widths = sorted(per_width)
    n = len(next(iter(per_width.values())))
    for i, a in enumerate(widths):
        for b in widths[i + 1 :]:
            diffs = [x - y for x, y in zip(per_width[a], per_width[b])]
            boots = sorted(
                sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(trials)
            )
            lo, hi = boots[int(trials * 0.025)], boots[int(trials * 0.975)]
            verdict = "significant" if (lo > 0 or hi < 0) else "within noise"
            obs = sum(diffs) / n
            print(f"  {a:>4} vs {b:<4} {obs:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {verdict}")
    print(
        "\n  All widths indistinguishable => embed at 1024. Qwen3 is Matryoshka-trained, so\n"
        "  truncation is principled rather than lossy chopping, and this is that claim\n"
        "  measured rather than asserted. 1024 is also the only width whose HNSW index beats\n"
        "  an exact scan on latency, and it stores 4x smaller."
    )


def _check_floor(questions: list[dict[str, Any]], scores: list[float]) -> None:
    """Fail loudly if 1-hop retrieval has collapsed. See MIN_1HOP_RECALL_AT_10."""
    # Excludes aggregation, so the floor keeps measuring the same population it was sized
    # against (0.586 over 30 questions), rather than drifting as the eval set grows.
    one_hop = [s for q, s in zip(questions, scores) if q["hops"] == 1 and not q.get("category")]
    if not one_hop:
        raise SystemExit("no 1-hop questions in the eval set — nothing to gate on")
    got = statistics.mean(one_hop)

    print()
    if got < MIN_1HOP_RECALL_AT_10:
        print(
            f"FAIL  1-hop R@10 is {got:.3f} at emb_{PRODUCTION_WIDTH}, below the "
            f"{MIN_1HOP_RECALL_AT_10:.2f} floor.\n"
            "      This floor is loose enough that noise does not trip it, so a breach means\n"
            "      something structural: wrong embedding column queried, vectors written\n"
            "      against the wrong chunk_ids, or the eval set no longer matching the corpus."
        )
        raise SystemExit(1)
    print(
        f"PASS  1-hop R@10 {got:.3f} at emb_{PRODUCTION_WIDTH}, floor {MIN_1HOP_RECALL_AT_10:.2f} "
        f"(catastrophe floor, not a quality threshold — see MIN_1HOP_RECALL_AT_10)"
    )

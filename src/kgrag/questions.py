"""`kgrag mine-questions` — a retrieval eval set derived from the graph, not from judgement.

Phase 1 shipped one eval that looked clean and hid a real bug: resolution scored P=1.000
while the pipeline merged ~1,957 pairs wrongly, purely because the sampler could not
surface the failing shape. The same trap is wide open here. An eval built from single-fact
lookups will report excellent recall@k and say nothing about the multi-hop questions the
graph exists to win, so the multi-hop cases are in the set from the start -- including the
ones vector search is supposed to lose.

Labels come from structure, the way resolve.mine_pairs() derives its labels from filing
structure. Every edge in the graph already carries the chunk_ids whose text justified it
(load.py gathers them, and the evidence-span check means those chunks demonstrably contain
the supporting sentence). Those chunk ids ARE the gold set: no hand-judgement, no model
call, no circularity beyond "the graph is the answer key", which is stated as a limitation
rather than hidden.

A 2-hop question's gold set is the union of two edges' chunks, which in this corpus almost
always spans two different filings. That is precisely the shape a single embedding of a
single question cannot retrieve in one shot.
"""

from __future__ import annotations

import random
from typing import Any

from . import jsonl, load
from .config import ENTITIES, EVAL

QUESTIONS = EVAL / "questions.jsonl"

#: One template per relation, phrased as a question a person would actually ask. Kept
#: deterministic rather than model-generated: the templates use canonical entity names
#: verbatim, which *helps* vector search by handing it exact lexical matches. If the
#: multi-hop numbers still collapse under conditions that favourable, the result is
#: stronger than it would be with naturalised phrasing.
#: ponytail: templated phrasing is the known ceiling here -- one model pass to rewrite
#: these into natural language is the upgrade if 1-hop recall looks implausibly high.
ONE_HOP = {
    "SUBSIDIARY_OF": "Which company owns {subject}?",
    "ACQUIRED": "Which company did {subject} acquire?",
    "COMPETES_WITH": "Who does {subject} name as a competitor?",
    "SUPPLIES": "What does {subject} supply to {object}?",
    "PARTNERS_WITH": "Who has {subject} partnered with?",
    "AUDITED_BY": "Who is {subject}'s independent auditor?",
    "OFFICER_OF": "What executive role does {subject} hold at {object}?",
    "DIRECTOR_OF": "Which board does {subject} sit on?",
    "OFFERS": "What is {object}, and who sells it?",
    "INCORPORATED_IN": "Where is {subject} incorporated?",
    "OPERATES_IN": "Where does {subject} have operations?",
    "REGULATED_BY": "Which regulator has authority over {subject}?",
    "EXPOSED_TO": "What does {subject} identify as a material risk?",
    "PARTY_TO": "What legal proceeding is {subject} a party to?",
}

#: Chained templates. The middle entity is deliberately NOT named in the question -- that
#: is what makes it multi-hop rather than two lookups, and what a single query embedding
#: has no way to bridge.
TWO_HOP = [
    ("DIRECTOR_OF", "ACQUIRED", "Which company was acquired by a company that {subject} is a director of?"),
    ("DIRECTOR_OF", "COMPETES_WITH", "Who competes with a company that {subject} sits on the board of?"),
    ("DIRECTOR_OF", "OPERATES_IN", "Where does the company {subject} is a director of have operations?"),
    ("SUBSIDIARY_OF", "COMPETES_WITH", "Who competes with the parent company of {subject}?"),
    ("SUBSIDIARY_OF", "OPERATES_IN", "Where does the parent of {subject} operate?"),
    ("SUBSIDIARY_OF", "EXPOSED_TO", "What risks does the parent company of {subject} disclose?"),
    ("ACQUIRED", "OFFERS", "What products are sold by the company that {subject} acquired?"),
    ("SUPPLIES", "COMPETES_WITH", "Who competes with the customers that {subject} supplies?"),
    ("SUPPLIES", "EXPOSED_TO", "What risks are disclosed by the companies {subject} supplies?"),
    ("OFFICER_OF", "COMPETES_WITH", "Who competes with the company where {subject} is an officer?"),
]

THREE_HOP = [
    ("DIRECTOR_OF", "ACQUIRED", "OFFERS",
     "What products come from the company acquired by the company {subject} directs?"),
    ("DIRECTOR_OF", "COMPETES_WITH", "OPERATES_IN",
     "Where do the competitors of {subject}'s company operate?"),
    ("SUBSIDIARY_OF", "COMPETES_WITH", "OPERATES_IN",
     "Where do the competitors of {subject}'s parent company operate?"),
    ("SUBSIDIARY_OF", "ACQUIRED", "OFFERS",
     "What products are sold by companies acquired by the parent of {subject}?"),
    ("SUPPLIES", "COMPETES_WITH", "EXPOSED_TO",
     "What risks do the competitors of {subject}'s customers disclose?"),
]


def _graph() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Entities by canonical id, and a flat edge list with names attached."""
    entities = {e["canonical_id"]: e for e in jsonl.read(ENTITIES)}
    by_predicate, _ = load._edges()
    edges = []
    for predicate, rows in by_predicate.items():
        for row in rows:
            s, o = entities.get(row["subject_id"]), entities.get(row["object_id"])
            if s and o:
                edges.append(
                    {
                        "predicate": predicate,
                        "subject_id": row["subject_id"],
                        "object_id": row["object_id"],
                        "subject": s["name"],
                        "object": o["name"],
                        "chunk_ids": row["chunk_ids"],
                    }
                )
    return entities, edges


def _spans_filings(chunk_ids: list[str], chunk_filing: dict[str, str]) -> bool:
    return len({chunk_filing.get(c) for c in chunk_ids}) > 1


def run(seed: int = 0, per_hop: int = 14) -> None:
    from .config import CHUNKS

    _, edges = _graph()
    if not edges:
        raise SystemExit("no edges — run `kgrag extract` and `kgrag resolve` first.")
    chunk_filing = {c["chunk_id"]: c["accession"] for c in jsonl.read(CHUNKS)}
    rng = random.Random(seed)

    out_by: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for e in edges:
        out_by.setdefault((e["subject_id"], e["predicate"]), []).append(e)

    rows: list[dict[str, Any]] = []

    # 1-hop: stratified across predicates so the set is not all SUBSIDIARY_OF (which is
    # the single most common relation and the least interesting to retrieve).
    by_pred: dict[str, list[dict[str, Any]]] = {}
    for e in edges:
        by_pred.setdefault(e["predicate"], []).append(e)
    per_pred = max(1, per_hop // len(ONE_HOP) + 1)
    for predicate, group in sorted(by_pred.items()):
        for edge in rng.sample(group, min(per_pred, len(group))):
            rows.append(
                {
                    "question": ONE_HOP[predicate].format(subject=edge["subject"], object=edge["object"]),
                    "gold_chunk_ids": sorted(edge["chunk_ids"]),
                    "hops": 1,
                    "source": "mined",
                    "predicate": predicate,
                }
            )

    # 2-hop and 3-hop: chains through a middle entity the question never names. Only
    # chains whose gold chunks span more than one filing are kept -- a chain that happens
    # to be described entirely inside one chunk is a 1-hop question wearing a costume,
    # and keeping it would quietly inflate multi-hop recall.
    rows += _chains(TWO_HOP, out_by, rng, per_hop, chunk_filing, hops=2)
    rows += _chains(THREE_HOP, out_by, rng, per_hop, chunk_filing, hops=3)

    # Two edges can template to the same sentence -- "Which company did Teradyne acquire?"
    # when Teradyne acquired three. Emitting it twice with different gold sets makes the
    # eval unanswerable: a retrieval that finds one acquisition is marked wrong for the
    # other. Merge on question text so the gold set is every chunk that answers it.
    merged: dict[str, dict[str, Any]] = {}
    for row in rows:
        seen = merged.get(row["question"])
        if seen is None:
            merged[row["question"]] = row
        else:
            seen["gold_chunk_ids"] = sorted(set(seen["gold_chunk_ids"]) | set(row["gold_chunk_ids"]))
    rows = list(merged.values())

    for i, row in enumerate(rows):
        row["qid"] = f"m{i:03d}"

    existing = [r for r in jsonl.read(QUESTIONS) if r.get("source") == "hand"]
    jsonl.write(QUESTIONS, rows + existing)

    print(f"{QUESTIONS.name}: {len(rows)} mined + {len(existing)} hand-written")
    for hops in (1, 2, 3):
        group = [r for r in rows if r["hops"] == hops]
        if group:
            avg = sum(len(r["gold_chunk_ids"]) for r in group) / len(group)
            print(f"  {hops}-hop  {len(group):3}  avg {avg:.1f} gold chunks")
    print("\nexamples:")
    for hops in (1, 2, 3):
        for row in [r for r in rows if r["hops"] == hops][:2]:
            print(f"  [{hops}h] {row['question']}")


def _chains(
    templates: list[tuple],
    out_by: dict[tuple[str, str], list[dict[str, Any]]],
    rng: random.Random,
    want: int,
    chunk_filing: dict[str, str],
    hops: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for template in templates:
        *predicates, text = template
        # Every edge of the first predicate is a possible starting point; walk the chain
        # and keep the first few that survive the cross-filing check.
        starts = [e for (sid, p), group in out_by.items() if p == predicates[0] for e in group]
        rng.shuffle(starts)
        kept = 0
        for first in starts:
            if kept >= max(1, want // len(templates)):
                break
            chain = [first]
            for predicate in predicates[1:]:
                nxt = out_by.get((chain[-1]["object_id"], predicate))
                if not nxt:
                    break
                chain.append(rng.choice(nxt))
            if len(chain) != len(predicates):
                continue
            ids = {n for e in chain for n in e["chunk_ids"]}
            # Reject degenerate chains: self-referential loops, and chains whose evidence
            # all sits in one filing (retrievable in one shot, so not really multi-hop).
            if len({e["subject_id"] for e in chain} | {e["object_id"] for e in chain}) < len(chain) + 1:
                continue
            if not _spans_filings(sorted(ids), chunk_filing):
                continue
            rows.append(
                {
                    "question": text.format(subject=first["subject"]),
                    "gold_chunk_ids": sorted(ids),
                    "hops": hops,
                    "source": "mined",
                    "predicate": "->".join(predicates),
                }
            )
            kept += 1
    return rows

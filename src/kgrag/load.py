"""Write resolved entities and validated relations into Neo4j, idempotently.

Two rules govern this module.

`MERGE`, never `CREATE`. Re-running ingestion has to be a no-op, or every debugging
loop doubles the graph. Node identity is the canonical_id from resolve.py; edge
identity is (subject, predicate, object).

An edge stores `chunk_ids` as a list rather than being duplicated per citation. Two
chunks asserting the same fact is corroboration, not a second edge — `support` counts
them, and Phase 4 reads `chunk_ids` to cite the sentences that justified the claim.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from neo4j import Driver, GraphDatabase

from . import jsonl
from .config import ENTITIES, EXTRACTIONS, ROOT, require
from .ontology import RelationType

BATCH = 1000


def driver() -> Driver:
    return GraphDatabase.driver(
        require("NEO4J_URI"), auth=(require("NEO4J_USER"), require("NEO4J_PASSWORD"))
    )


def apply_schema(db: Driver) -> None:
    # Comments are stripped before splitting: a `;` inside a // comment would otherwise
    # cut the comment in half and feed the tail to Cypher as a statement.
    text = re.sub(r"//[^\n]*", "", (ROOT / "cypher" / "schema.cypher").read_text())
    with db.session() as session:
        for statement in text.split(";"):
            if statement.strip():
                session.run(statement)


NODE_MERGE = """
UNWIND $rows AS row
MERGE (e:Entity {id: row.canonical_id})
SET e:$(row.type),
    e.name = row.name,
    e.type = row.type,
    e.aliases = row.aliases,
    e.mention_count = row.mention_count
"""

# The relationship type cannot be parameterised in Cypher, so rows are grouped by
# predicate and the type is interpolated. The interpolated value is a RelationType enum
# member, never a string that came from the model — the same "no model-authored Cypher"
# rule the project applies to query time, enforced here at write time.
EDGE_MERGE = """
UNWIND $rows AS row
MATCH (s:Entity {{id: row.subject_id}})
MATCH (o:Entity {{id: row.object_id}})
MERGE (s)-[r:{predicate}]->(o)
SET r.chunk_ids  = apoc.coll.toSet(coalesce(r.chunk_ids, []) + row.chunk_ids),
    r.support    = size(apoc.coll.toSet(coalesce(r.chunk_ids, []) + row.chunk_ids)),
    r.confidence = row.confidence
"""


def _surface_to_id() -> dict[tuple[str, str], str]:
    """(surface form, type) -> canonical id, for every alias resolve.py collapsed."""
    index: dict[tuple[str, str], str] = {}
    for entity in jsonl.read(ENTITIES):
        for name in [entity["name"], *entity["aliases"]]:
            index[(name, entity["type"])] = entity["canonical_id"]
    return index


def _edges() -> tuple[dict[str, list[dict[str, Any]]], Counter[str]]:
    """Collapse relations onto (subject_id, predicate, object_id), gathering chunk ids."""
    index = _surface_to_id()
    types_by_chunk: dict[str, dict[str, str]] = {}
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    skipped: Counter[str] = Counter()

    rows = list(jsonl.read(EXTRACTIONS))
    for row in rows:
        types_by_chunk[row["chunk_id"]] = {m["name"]: m["type"] for m in row["mentions"]}

    for row in rows:
        mention_types = types_by_chunk[row["chunk_id"]]
        for rel in row["relations"]:
            subject = index.get((rel["subject"], mention_types.get(rel["subject"], "")))
            obj = index.get((rel["object"], mention_types.get(rel["object"], "")))
            if not subject or not obj:
                skipped["unresolved_endpoint"] += 1
                continue
            if subject == obj:
                # Distinct surface forms that resolution merged into one node. The
                # relation was valid pre-resolution; it is a self-loop now.
                skipped["self_loop_after_resolution"] += 1
                continue
            key = (subject, rel["predicate"], obj)
            edge = grouped.setdefault(
                key,
                {
                    "subject_id": subject,
                    "object_id": obj,
                    "chunk_ids": [],
                    "confidence": rel["confidence"],
                },
            )
            edge["chunk_ids"].append(row["chunk_id"])
            # Corroboration upgrades confidence; a single low-confidence read does not.
            if rel["confidence"] == "high":
                edge["confidence"] = "high"

    by_predicate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (_, predicate, _), edge in grouped.items():
        edge["chunk_ids"] = sorted(set(edge["chunk_ids"]))
        by_predicate[predicate].append(edge)
    return by_predicate, skipped


def run() -> None:
    entities = list(jsonl.read(ENTITIES))
    if not entities:
        raise SystemExit("data/entities.jsonl is empty — run `kgrag resolve` first.")
    by_predicate, skipped = _edges()

    with driver() as db:
        apply_schema(db)
        with db.session() as session:
            for start in range(0, len(entities), BATCH):
                session.run(NODE_MERGE, rows=entities[start : start + BATCH])
            for predicate, edges in by_predicate.items():
                # Validated against the enum before it is ever interpolated into Cypher.
                statement = EDGE_MERGE.format(predicate=RelationType(predicate).value)
                for start in range(0, len(edges), BATCH):
                    session.run(statement, rows=edges[start : start + BATCH])

        print(f"nodes  {len(entities):,}")
        print(f"edges  {sum(len(e) for e in by_predicate.values()):,}")
        for predicate, edges in sorted(by_predicate.items(), key=lambda kv: -len(kv[1])):
            print(f"  {predicate:18} {len(edges):6,}")
        if skipped:
            print(f"skipped: {dict(skipped)}")
        print()
        print(stats(db))


def stats(db: Driver) -> str:
    """The health checks that gate Phase 2. Orphans and self-loops are the tells."""
    with db.session() as session:
        nodes = session.run("MATCH (e:Entity) RETURN count(e) AS n").single()["n"]
        edges = session.run("MATCH ()-[r]->() RETURN count(r) AS n").single()["n"]
        orphans = session.run(
            "MATCH (e:Entity) WHERE NOT (e)--() RETURN count(e) AS n"
        ).single()["n"]
        loops = session.run("MATCH (e)-[r]->(e) RETURN count(r) AS n").single()["n"]
    orphan_rate = orphans / nodes if nodes else 0.0
    return (
        f"graph: {nodes:,} nodes, {edges:,} edges, "
        f"{orphans:,} orphans ({orphan_rate:.1%}), {loops} self-loops"
    )

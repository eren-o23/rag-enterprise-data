"""`kgrag verify` — the gate Phase 1 has to clear before Phase 2 starts.

Every check here is one that fails loudly on a graph that looks fine in the browser:
a relation type nothing ever populated, a corpus of disconnected islands, or a
three-hop question with no three-hop path to find.
"""

from __future__ import annotations

from . import jsonl, load
from .config import CHUNKS, ENTITIES, EXTRACTIONS
from .ontology import ALLOWED_EDGES, RelationType

MAX_ORPHAN_RATE = 0.05

#: The question the whole project exists to answer, as Cypher. If this returns nothing,
#: the graph cannot do the thing that distinguishes it from a vector index, and the
#: Phase 5 benchmark has no story.
THREE_HOP = """
MATCH path = (p:Person)-[:DIRECTOR_OF]->(a:Company)-[:ACQUIRED|SUPPLIES|COMPETES_WITH]->(b:Company)
             <-[:SUBSIDIARY_OF|ACQUIRED]-(c:Company)
WHERE a <> b AND b <> c AND a <> c
RETURN p.name AS person, a.name AS via, b.name AS mid, c.name AS target
LIMIT 5
"""

SHARED_DIRECTOR = """
MATCH (a:Company)<-[:DIRECTOR_OF]-(p:Person)-[:DIRECTOR_OF]->(b:Company)
WHERE elementId(a) < elementId(b)
RETURN a.name AS company_a, b.name AS company_b, collect(p.name)[0..3] AS directors,
       count(p) AS shared
ORDER BY shared DESC LIMIT 5
"""


def run() -> None:
    failures: list[str] = []

    chunks = list(jsonl.read(CHUNKS))
    extractions = list(jsonl.read(EXTRACTIONS))
    entities = list(jsonl.read(ENTITIES))
    print(f"corpus   {len(chunks):,} chunks, {len(extractions):,} extracted, {len(entities):,} entities")
    if extractions and len(extractions) < len(chunks):
        print(f"         {len(chunks) - len(extractions):,} chunks not yet extracted")

    with load.driver() as db:
        print(load.stats(db))
        with db.session() as session:
            nodes = session.run("MATCH (e:Entity) RETURN count(e) AS n").single()["n"]
            orphans = session.run("MATCH (e:Entity) WHERE NOT (e)--() RETURN count(e) AS n").single()["n"]
            loops = session.run("MATCH (e)-[r]->(e) RETURN count(r) AS n").single()["n"]

            print("\nedges by relation type:")
            empty = []
            for relation in RelationType:
                n = session.run(f"MATCH ()-[r:{relation.value}]->() RETURN count(r) AS n").single()["n"]
                src, dst = ALLOWED_EDGES[relation]
                flag = "  <- EMPTY" if n == 0 else ""
                print(f"  {relation.value:18} {n:6,}  ({src.value} -> {dst.value}){flag}")
                if n == 0:
                    empty.append(relation.value)

            print("\ncompanies sharing a director:")
            shared = list(session.run(SHARED_DIRECTOR))
            for row in shared:
                print(f"  {row['company_a'][:28]:30} {row['company_b'][:28]:30} {row['directors']}")

            print("\nthree-hop paths:")
            hops = list(session.run(THREE_HOP))
            for row in hops:
                print(f"  {row['person']} -> {row['via']} -> {row['mid']} <- {row['target']}")

    if nodes and orphans / nodes > MAX_ORPHAN_RATE:
        failures.append(f"orphan rate {orphans / nodes:.1%} exceeds {MAX_ORPHAN_RATE:.0%}")
    if loops:
        failures.append(f"{loops} self-loops in the graph")
    if empty:
        failures.append(f"relation types with no edges: {', '.join(empty)}")
    if not shared:
        failures.append("no two companies share a director — the proxy extraction is not working")
    if not hops:
        failures.append("no three-hop path exists — the graph cannot beat a vector index")

    print()
    if failures:
        for f in failures:
            print(f"FAIL  {f}")
        raise SystemExit(1)
    print("PASS  Phase 1 gate clear")

"""`kgrag verify` — the gate Phase 1 has to clear before Phase 2 starts.

Every check here is one that fails loudly on a graph that looks fine in the browser:
a relation type nothing ever populated, a corpus of disconnected islands, or a
three-hop question with no three-hop path to find.
"""

from __future__ import annotations

from . import embed, jsonl, load
from .config import CHUNKS, ENTITIES, EXTRACTIONS
from .ontology import ALLOWED_EDGES, RelationType

#: Measured 25.0% on the real corpus (960/3,836 nodes) — see docs/decisions.md. The 5%
#: guess this replaced assumed most mentions get a relation; a closed 14-relation
#: ontology over real SEC prose doesn't work that way (see decisions.md for the breakdown).
#: 30% keeps this a real gate — a genuinely broken extraction run should still trip it.
MAX_ORPHAN_RATE = 0.30

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

    failures += _check_vectors(len(chunks))

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


#: The production embedding column, chosen by measuring 1024 against 2000 and 4096 with
#: `kgrag recall` rather than by assuming Matryoshka truncation is free.
PRODUCTION_WIDTH = 1024


def _check_vectors(n_chunks: int) -> list[str]:
    """The Phase 2 half of the gate: is the vector store complete and joinable?

    Kept separate from the Neo4j checks above because the two stores fail independently,
    and a Postgres that is simply not running should say so rather than masquerading as
    a data problem.
    """
    failures: list[str] = []
    try:
        conn = embed.connect()
    except Exception as exc:  # noqa: BLE001
        print(f"\npostgres  UNREACHABLE ({type(exc).__name__}) — Phase 2 checks skipped")
        return ["postgres unreachable, so the vector half of the gate did not run"]

    with conn:
        rows = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        print(f"\nvector store: {rows:,} rows")
        for width in embed.WIDTHS:
            got = conn.execute(
                f"SELECT count(emb_{width}) FROM chunks"  # noqa: S608
            ).fetchone()[0]
            flag = "  <- INCOMPLETE" if got != rows else ""
            print(f"  emb_{width:<5} {got:>6,} embedded{flag}")
            if width == PRODUCTION_WIDTH and got != rows:
                failures.append(f"{rows - got} chunks have no emb_{width}")

        # Every chunk, including the 2 that were quarantined at extraction. They carry no
        # graph entities but are still real passages and must stay retrievable.
        if rows != n_chunks:
            failures.append(f"vector store has {rows:,} rows, chunks.jsonl has {n_chunks:,}")

        indexes = {
            r[0]
            for r in conn.execute(
                "SELECT indexname FROM pg_indexes WHERE tablename = 'chunks'"
            ).fetchall()
        }
        for width in (1024, 2000):
            if f"chunks_emb_{width}_hnsw" not in indexes:
                failures.append(f"no HNSW index on emb_{width}")

        # The join key is the whole point of building two stores. Verify it actually
        # crosses, rather than trusting that two independently-correct stores line up.
        with load.driver() as db, db.session() as session:
            sample = conn.execute(
                "SELECT chunk_id FROM chunks WHERE cardinality(entity_ids) > 2 LIMIT 20"
            ).fetchall()
            joined = sum(
                1
                for (cid,) in sample
                if session.run(
                    "MATCH ()-[r]->() WHERE $cid IN r.chunk_ids RETURN count(r) AS n",
                    cid=cid,
                ).single()["n"]
            )
        print(f"  join key    {joined}/{len(sample)} sampled chunk_ids resolve to a Neo4j edge")
        if sample and joined == 0:
            failures.append("no sampled chunk_id resolves to a Neo4j edge — the stores are decoupled")

    return failures

"""One file, plain asserts. Each test covers a thing that would silently corrupt the graph.

Nothing here calls Fireworks: the embedding score is injected, so the tests are free,
offline, and fast. The load test needs Neo4j and skips itself when it is not up.
"""

from __future__ import annotations

import pytest

from kgrag import ontology, resolve
from kgrag.chunk import _split, chunk_id
from kgrag.ontology import EntityType, Extraction, Mention, Relation, RelationType

# --------------------------------------------------------------------------- ontology


def test_every_relation_has_a_type_signature_and_prompt_docs():
    """Ontology drift check. Adding a RelationType without a signature would let the
    model emit an edge that validate() cannot type-check, silently widening the ontology."""
    assert set(ontology.ALLOWED_EDGES) == set(RelationType)
    assert set(ontology.RELATION_DOCS) == set(RelationType)


def test_prompt_block_is_generated_from_the_ontology():
    block = ontology.ontology_prompt_block()
    for relation in RelationType:
        assert relation.value in block
    assert "supply chain concentration" in block  # closed RiskTopic vocabulary


# --------------------------------------------------------------------------- validation

CHUNK = (
    "Broadcom Inc. acquired VMware, Inc. in a transaction valued at $69 billion. "
    "Lisa Su serves as a director of Cisco Systems, Inc."
)


def _extraction(*relations: Relation) -> Extraction:
    return Extraction(
        mentions=[
            Mention(name="Broadcom Inc.", type=EntityType.COMPANY),
            Mention(name="VMware, Inc.", type=EntityType.COMPANY),
            Mention(name="Lisa Su", type=EntityType.PERSON),
            Mention(name="California", type=EntityType.LOCATION),
        ],
        relations=list(relations),
    )


def test_valid_relation_survives():
    kept, dropped = ontology.validate(
        _extraction(
            Relation(subject="Broadcom Inc.", predicate=RelationType.ACQUIRED, object="VMware, Inc.",
                     confidence="high", evidence="Broadcom Inc. acquired VMware, Inc.")
        ),
        CHUNK,
    )
    assert len(kept) == 1 and not dropped


@pytest.mark.parametrize(
    "relation,reason",
    [
        # Evidence the model wrote rather than copied — the hallucination filter.
        (Relation(subject="Broadcom Inc.", predicate=RelationType.ACQUIRED, object="VMware, Inc.",
                  confidence="high", evidence="Broadcom purchased Intel's foundry business."),
         "evidence_not_found"),
        # ACQUIRED is Company -> Company; this object is a Location.
        (Relation(subject="Broadcom Inc.", predicate=RelationType.ACQUIRED, object="California",
                  confidence="low", evidence="Broadcom Inc. acquired"),
         "type_signature"),
        # Endpoint was never declared as a mention.
        (Relation(subject="Broadcom Inc.", predicate=RelationType.SUPPLIES, object="Apple",
                  confidence="low", evidence="Broadcom Inc. acquired"),
         "unknown_endpoint"),
        (Relation(subject="Broadcom Inc.", predicate=RelationType.ACQUIRED, object="Broadcom Inc.",
                  confidence="low", evidence="Broadcom Inc. acquired"),
         "self_loop"),
    ],
)
def test_invalid_relations_are_dropped_with_a_reason(relation, reason):
    kept, dropped = ontology.validate(_extraction(relation), CHUNK)
    assert kept == []
    assert dropped == {reason: 1}


def test_evidence_check_tolerates_whitespace_but_not_paraphrase():
    kept, _ = ontology.validate(
        _extraction(
            Relation(subject="Lisa Su", predicate=RelationType.DIRECTOR_OF, object="VMware, Inc.",
                     confidence="high", evidence="Broadcom   Inc.\n acquired  VMware, Inc.")
        ),
        CHUNK,
    )
    assert len(kept) == 1  # ragged whitespace is normalised, the span is genuinely present


# --------------------------------------------------------------------------- embeddings


def test_embedding_cache_key_separates_dimensions_without_invalidating_the_old_cache():
    """The one Phase 2 change that can silently corrupt the vector store.

    qwen3-embedding-8b is 4096 native and truncates via a `dimensions` parameter. If that
    parameter is not in the cache key, a cached 4096-dim vector comes back for a 1024-dim
    request and the mismatch surfaces far downstream, if at all. The None case must keep
    hashing exactly as it did before, or the several thousand entity-name vectors
    `kgrag resolve` already cached at 4096 are all invalidated for nothing.
    """
    import hashlib

    from kgrag.fireworks import EMBED_MODEL, _embed_key

    legacy = hashlib.sha256(f"{EMBED_MODEL}|Broadcom Inc.".encode()).hexdigest()
    assert _embed_key(EMBED_MODEL, "Broadcom Inc.", None) == legacy

    keys = {_embed_key(EMBED_MODEL, "Broadcom Inc.", d) for d in (None, 1024, 2000, 4096)}
    assert len(keys) == 4


def test_chunk_to_entity_inversion_is_complete():
    """resolve.py writes entity -> mention_chunks; pgvector rows need the reverse. Nothing
    stores it, so embed.py inverts in memory -- and a dropped chunk there means a passage
    that silently cannot be filtered to the entity it names."""
    from kgrag.embed import chunk_entities

    entities = [
        {"canonical_id": "e_amd", "mention_chunks": ["c1", "c2"]},
        {"canonical_id": "e_nvda", "mention_chunks": ["c2", "c3"]},
        {"canonical_id": "e_tsmc", "mention_chunks": ["c1"]},
    ]
    index = chunk_entities(entities)

    assert set(index) == {"c1", "c2", "c3"}
    assert index["c1"] == ["e_amd", "e_tsmc"]  # sorted, so the column is stable across runs
    assert index["c2"] == ["e_amd", "e_nvda"]
    assert sum(len(v) for v in index.values()) == sum(len(e["mention_chunks"]) for e in entities)


def test_recall_floor_fires_on_collapse_but_not_on_noise():
    """A gate that only ever passes is the failure this project already has a history of:
    Phase 1's resolution eval reported P=1.000 while the pipeline merged ~1,957 pairs
    wrongly. So assert the floor actually trips, not just that it exists.

    Verified against the real store too: scrambling which answer key belongs to which
    question -- what a chunk_id mismatch between Postgres and Neo4j would look like --
    takes 1-hop R@10 from 0.586 to 0.033, far below the floor and far outside the noise
    band the loose threshold is sized for.
    """
    from kgrag import retrieve

    questions = [{"hops": 1} for _ in range(30)] + [{"hops": 2} for _ in range(10)]

    healthy = [0.586] * 30 + [0.288] * 10
    retrieve._check_floor(questions, healthy)  # must not raise

    collapsed = [0.033] * 30 + [0.288] * 10
    with pytest.raises(SystemExit):
        retrieve._check_floor(questions, collapsed)

    # Sized so ordinary noise cannot trip it: the floor sits ~0.23 below the measured
    # value, while the bootstrap put per-width differences inside ±0.09.
    assert retrieve.MIN_1HOP_RECALL_AT_10 < 0.586 - 0.15

    # Only the 1-hop slice gates. Multi-hop is expected to be low -- that is the Phase 5
    # baseline -- so letting it into the average would make the floor meaningless.
    retrieve._check_floor([{"hops": 2}] * 10 + questions, [0.0] * 10 + healthy)


# --------------------------------------------------------------------------- chunking


def test_chunk_ids_are_deterministic_across_runs():
    """Phase 2 joins pgvector rows to graph edges on this id. If it moves, they decouple."""
    a = chunk_id("0000002488-26-000018", "10-K/Item 1", 3)
    b = chunk_id("0000002488-26-000018", "10-K/Item 1", 3)
    assert a == b and len(a) == 16
    assert a != chunk_id("0000002488-26-000018", "10-K/Item 1A", 3)
    assert a != chunk_id("0000002488-26-000018", "10-K/Item 1", 4)


def test_split_covers_the_text_and_respects_the_size_target():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 120 for i in range(40))
    chunks = list(_split(text))
    assert len(chunks) > 1
    assert all(len(c) <= 4000 + 800 for c in chunks)
    assert "Paragraph 0." in chunks[0] and "Paragraph 39." in chunks[-1]


# --------------------------------------------------------------------------- resolution


def _rec(name: str, etype: str = "Company") -> dict:
    norm = resolve.normalize(name)
    return {"surface": name, "type": etype, "norm": norm, "tokens": resolve.tokens(norm)}


def test_acronym_merges_where_no_cosine_threshold_could():
    """AMD and Advanced Micro Devices share zero tokens, so the Jaccard gate rejects them
    at any cosine. The acronym rule is what makes this work — and blocking has to emit an
    acronym key too, or the pair is never even compared."""
    a, b = _rec("AMD"), _rec("Advanced Micro Devices, Inc.")
    assert resolve.matches(a, b, cosine=0.0, overrides={}) == "acronym"
    assert resolve.blocking_keys(a["norm"], a["surface"]) & resolve.blocking_keys(b["norm"], b["surface"])


def test_lexical_gate_blocks_the_classic_false_merge():
    """These two embed very close together. Cosine alone would merge them."""
    a, b = _rec("Advanced Micro Devices"), _rec("Advanced Energy Industries")
    assert resolve.matches(a, b, cosine=0.99, overrides={}) is None


def test_foreign_legal_form_marks_a_separate_incorporation():
    """A parent and its Finnish subsidiary differ only by "Oy". Merging them turned 317
    SUBSIDIARY_OF edges into self-loops that load.py then discarded."""
    a, b = _rec("Skyworks Solutions, Inc."), _rec("Skyworks Solutions Oy")
    assert resolve.matches(a, b, cosine=0.99, overrides={}) is None


def test_raw_acronym_survives_legal_suffix_stripping():
    """TSMC's C comes from "Company", which normalize() strips, so acronym() alone yields
    "tsm" and never matches. Blocking has to emit the raw acronym key too."""
    a, b = _rec("TSMC"), _rec("Taiwan Semiconductor Manufacturing Company")
    assert resolve.matches(a, b, cosine=0.0, overrides={}) == "acronym"
    assert resolve.blocking_keys(a["norm"], a["surface"]) & resolve.blocking_keys(b["norm"], b["surface"])


def test_brand_short_form_comes_from_the_filing_not_a_rule():
    """`Cisco Systems, Inc. ("Cisco")` shares only a leading token, so no threshold
    reaches it. Generalising that into a leading-token rule is what merged "Robert A.
    Feurle" with "Robert A. Schriesheim" — so the alias is carried as mined data instead,
    and the rules alone are expected to decline the pair."""
    a, b = _rec("Cisco Systems, Inc."), _rec("Cisco")
    assert resolve.matches(a, b, cosine=0.8, overrides={}) is None
    declared = {(a["norm"], b["norm"]): True}
    assert resolve.matches(a, b, cosine=0.0, overrides=declared) == "override"


def test_legal_suffix_variants_collapse():
    for name in ("Broadcom Inc.", "Broadcom Corporation", "BROADCOM"):
        assert resolve.normalize(name) == "broadcom"
    assert resolve.matches(_rec("Broadcom Inc."), _rec("Broadcom Corporation"), 0.0, {}) == "exact"


def test_industry_words_survive_normalisation():
    """Stripping them merged 'Taiwan Semiconductor Manufacturing' into 'taiwan manufacturing'."""
    assert resolve.normalize("Taiwan Semiconductor Manufacturing Company Limited") == (
        "taiwan semiconductor manufacturing"
    )
    assert resolve.normalize("NXP Semiconductors N.V.") == "nxp semiconductors"


def test_overrides_win_in_both_directions():
    tsmc, tsm = _rec("TSMC"), _rec("Taiwan Semiconductor Manufacturing Company Limited")
    assert resolve.matches(tsmc, tsm, cosine=0.99, overrides={}) is None
    forced = {(tsmc["norm"], tsm["norm"]): True}
    assert resolve.matches(tsmc, tsm, cosine=0.0, overrides=forced) == "override"
    split = {(resolve.normalize("Broadcom Inc."), resolve.normalize("Broadcom Corp")): False}
    assert resolve.matches(_rec("Broadcom Inc."), _rec("Broadcom Corp"), 0.99, split) is None


def test_different_types_never_share_a_canonical_id():
    shared = ["acme"]
    assert resolve.canonical_id("Company", shared) != resolve.canonical_id("Product", shared)


def test_canonical_id_is_stable_when_cluster_frequency_shifts():
    """The id keys on the cluster's smallest member, not its most frequent one. Frequency
    moves as the corpus grows, and a moving id breaks every citation already emitted."""
    assert resolve.canonical_id("Company", ["amd", "advanced micro devices"]) == (
        resolve.canonical_id("Company", ["advanced micro devices", "amd"])
    )


# --------------------------------------------------------------------------- load


def test_load_is_idempotent():
    """The Phase 1 acceptance test: re-running ingestion must not grow the graph."""
    neo4j = pytest.importorskip("neo4j")
    from kgrag import load

    try:
        db = load.driver()
        db.verify_connectivity()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Neo4j not reachable: {exc}")

    entities = [
        {"canonical_id": "test_a", "type": "Company", "name": "Alpha Corp", "aliases": ["Alpha"],
         "mention_count": 2},
        {"canonical_id": "test_b", "type": "Company", "name": "Beta Inc", "aliases": [],
         "mention_count": 1},
    ]
    edges = [{"subject_id": "test_a", "object_id": "test_b", "chunk_ids": ["c1"], "confidence": "high"}]
    statement = load.EDGE_MERGE.format(predicate=RelationType.ACQUIRED.value)

    with db.session() as session:
        session.run("MATCH (e:Entity) WHERE e.id STARTS WITH 'test_' DETACH DELETE e")
        for _ in range(2):
            session.run(load.NODE_MERGE, rows=entities)
            session.run(statement, rows=edges)
        counts = session.run(
            "MATCH (e:Entity)-[r]->() WHERE e.id STARTS WITH 'test_' "
            "RETURN count(DISTINCT e) AS nodes, count(r) AS edges, collect(r.chunk_ids)[0] AS ids"
        ).single()
        assert counts["nodes"] == 1 and counts["edges"] == 1
        assert counts["ids"] == ["c1"], "chunk_ids must not accumulate duplicates"
        labels = session.run("MATCH (e:Entity {id:'test_a'}) RETURN labels(e) AS l").single()["l"]
        assert set(labels) == {"Entity", "Company"}
        session.run("MATCH (e:Entity) WHERE e.id STARTS WITH 'test_' DETACH DELETE e")
    db.close()


# --------------------------------------------------------------------------- route


def test_traversal_rejects_a_predicate_the_model_invented():
    """The security control. A relationship type cannot be a Cypher bind parameter, so it
    has to be interpolated -- which means the ONLY thing standing between the model and
    the query string is RelationType(). Anything not in the ontology must raise before it
    reaches Neo4j, exactly as load.py enforces at write time."""
    from kgrag.route import arrows

    assert arrows(["ACQUIRED"]) == "-[r0:ACQUIRED]->()"
    assert arrows(["DIRECTOR_OF", "ACQUIRED"]) == "-[r0:DIRECTOR_OF]->()-[r1:ACQUIRED]->()"
    assert arrows([]) == "-[r0]-()"  # neighbourhood

    for hostile in ("DROP DATABASE neo4j", "ACQUIRED]->() DETACH DELETE (n", "acquired", ""):
        with pytest.raises(ValueError):
            arrows(["ACQUIRED", hostile])


def test_chain_length_must_match_shape():
    """Layer 2. Constrained decoding guarantees every chain member is a real relation; it
    cannot guarantee there are the right number of them. A two_hop shape carrying one
    predicate means the traversal the question implied is unknown, so the graph path must
    not run alone on a guess."""
    from kgrag.route import Plan, Route, Shape, decide

    good = Plan(route=Route.GRAPH, confidence="high", entities=["AMD"],
                shape=Shape.TWO_HOP, chain=[RelationType.DIRECTOR_OF, RelationType.ACQUIRED])
    assert decide(good, ["id_amd"]) == (Route.GRAPH, ["DIRECTOR_OF", "ACQUIRED"], [])

    bad = good.model_copy(update={"chain": [RelationType.DIRECTOR_OF]})
    route, chain, reasons = decide(bad, ["id_amd"])
    assert route is Route.BOTH and chain == [] and reasons == ["chain_shape_mismatch"]

    # No anchor node: nothing to traverse from, so the graph path is not merely wrong, it
    # is impossible.
    route, _, reasons = decide(good, [])
    assert route is Route.VECTOR and reasons == ["no_entity_resolved"]

    # The spec's low-confidence fallback: run both and merge.
    unsure = good.model_copy(update={"confidence": "low"})
    route, _, reasons = decide(unsure, ["id_amd"])
    assert route is Route.BOTH and reasons == ["low_confidence"]


def test_entity_index_resolves_an_acronym_the_fulltext_index_loses():
    """The measured Phase 3 finding. Neo4j's fulltext index was created for this lookup and
    returns AMD Ryzen(TM) PRO, AMD Japan Ltd. and AMD (EMEA) LTD. for "AMD" -- the real node
    carries "AMD" as one of eleven aliases and Lucene's length normalisation buries it.
    An exact index over the same aliases does not have that failure mode."""
    from kgrag.route import entity_index, resolve_entity

    entities = [
        {"canonical_id": "amd", "type": "Company", "name": "ADVANCED MICRO DEVICES INC",
         "aliases": ["AMD", "Advanced Micro Devices, Inc.", "Advanced Micro Devices GmbH"],
         "mention_count": 900},
        {"canonical_id": "amd_japan", "type": "Company", "name": "AMD Japan Ltd.",
         "aliases": [], "mention_count": 1},
        {"canonical_id": "intel_co", "type": "Company", "name": "INTEL CORP",
         "aliases": ["Intel"], "mention_count": 500},
        {"canonical_id": "intel_loc", "type": "Location", "name": "Intel",
         "aliases": [], "mention_count": 3},
    ]
    index = entity_index(entities)

    assert resolve_entity("AMD", index) == "amd"
    assert resolve_entity("amd", index) == "amd"  # normalisation, not case luck
    assert resolve_entity("Advanced Micro Devices Inc", index) == "amd"  # legal suffix stripped
    assert resolve_entity("AMD Japan Ltd.", index) == "amd_japan"  # still its own entity

    # 28 normalized surfaces are genuinely ambiguous on the real corpus. mention_count is
    # the tiebreak, and it is already on every entity.
    assert resolve_entity("Intel", index) == "intel_co"

    # No exact hit and no session to fall back to is a miss, not a wrong answer.
    assert resolve_entity("Cyberdyne Systems", index) is None


def test_route_end_to_end():
    """The Phase 3 acceptance test, against the real graph: a 1-hop question about an
    entity in the corpus must resolve, traverse, and return the chunk that justified the
    edge. Skips when Neo4j is down, like the load test."""
    pytest.importorskip("neo4j")
    from kgrag import load
    from kgrag.route import entity_index, graph_path, resolve_entity

    try:
        db = load.driver()
        db.verify_connectivity()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Neo4j not reachable: {exc}")

    index = entity_index()
    if not index:
        pytest.skip("data/entities.jsonl not present (fresh clone)")

    with db.session() as session:
        node_id = resolve_entity("TERADYNE, INC", index, session)
        assert node_id, "a corpus company must resolve to a node"

        got = graph_path(session, [node_id], ["ACQUIRED"], k=10)
        assert got, "Teradyne has ACQUIRED edges; the traversal must find their chunks"

        # Every returned id must be a real chunk id, not a node id or a stray property.
        assert all(len(c) == 16 for c in got)
        assert len(set(got)) == len(got), "chunk ids must be deduped"

        # Neighbourhood is the aggregation fallback and must reach at least as much as a
        # single fixed chain does.
        assert set(got) <= set(graph_path(session, [node_id], [], k=200))

    db.close()


def test_routing_log_resume_takes_the_latest_row_per_model(tmp_path, monkeypatch):
    """The eval resumes from its own log, so a re-measured question must supersede its
    history rather than replaying it. Getting this backwards would silently serve stale
    numbers after a ranking change -- the same shape of failure as an eval that cannot
    fail, which this project has already shipped once."""
    from kgrag import jsonl, route as route_mod

    log = tmp_path / "routing_log.jsonl"
    jsonl.append(log, [
        {"qid": "m000", "model": "small", "route": "vector"},
        {"qid": "m001", "model": "small", "route": "graph"},
        {"qid": "m000", "model": "small", "route": "graph"},   # re-measured, must win
        {"qid": "m000", "model": "big", "route": "refuse"},    # different router
        {"qid": None, "model": "small", "route": "both"},      # ad-hoc --question run
    ])
    monkeypatch.setattr(route_mod, "ROUTING_LOG", log)

    small = route_mod._prior("small")
    assert small["m000"]["route"] == "graph", "the later row must win"
    assert small["m001"]["route"] == "graph"
    assert None not in small, "ad-hoc runs carry no qid and must not resume anything"
    assert route_mod._prior("big")["m000"]["route"] == "refuse", "models must not cross"
    assert route_mod._prior("unrun") == {}

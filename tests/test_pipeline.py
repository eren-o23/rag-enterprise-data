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
    sha = route_mod.router_sha()
    jsonl.append(log, [
        {"qid": "m000", "model": "small", "route": "vector", "router_sha": sha},
        {"qid": "m001", "model": "small", "route": "graph", "router_sha": sha},
        {"qid": "m000", "model": "small", "route": "graph", "router_sha": sha},  # re-measured
        {"qid": "m000", "model": "big", "route": "refuse", "router_sha": sha},   # other router
        {"qid": None, "model": "small", "route": "both", "router_sha": sha},     # ad-hoc run
        {"qid": "m003", "model": "small", "route": "graph", "router_sha": "stalePrompt"},
    ])
    monkeypatch.setattr(route_mod, "ROUTING_LOG", log)

    small = route_mod._prior("small")
    assert small["m000"]["route"] == "graph", "the later row must win"
    assert small["m001"]["route"] == "graph"
    assert None not in small, "ad-hoc runs carry no qid and must not resume anything"
    assert route_mod._prior("big")["m000"]["route"] == "refuse", "models must not cross"
    assert route_mod._prior("unrun") == {}
    assert "m003" not in small, "a decision from a different prompt must not be resumed"


def test_routing_log_never_resumes_over_a_router_timeout(tmp_path, monkeypatch):
    """A timed-out router call recorded a fallback, not a decision. Resuming over it would
    freeze a transient Fireworks stall into the published numbers -- and chat_json only
    caches successes, so the retry is available and cannot return a stale answer."""
    from kgrag import jsonl, route as route_mod

    log = tmp_path / "routing_log.jsonl"
    sha = route_mod.router_sha()
    jsonl.append(log, [
        {"qid": "m000", "model": "small", "route": "graph", "degrade_reason": None,
         "router_sha": sha},
        {"qid": "m001", "model": "small", "route": "both", "router_sha": sha,
         "degrade_reason": "router_unreachable:APITimeoutError+low_confidence"},
        {"qid": "m002", "model": "small", "route": "both", "degrade_reason": "low_confidence",
         "router_sha": sha},
    ])
    monkeypatch.setattr(route_mod, "ROUTING_LOG", log)

    prior = route_mod._prior("small")
    assert "m001" not in prior, "a router timeout must be retried, not resumed"
    assert prior["m000"]["route"] == "graph"
    assert "m002" in prior, "an ordinary low-confidence degrade IS a real decision"

    # A later successful decision supersedes an earlier timeout for the same question.
    jsonl.append(log, [{"qid": "m001", "model": "small", "route": "graph",
                        "degrade_reason": None, "router_sha": sha}])
    assert route_mod._prior("small")["m001"]["route"] == "graph"

    # ...and a later timeout invalidates an earlier success, so the retry happens.
    jsonl.append(log, [{"qid": "m000", "model": "small", "route": "both", "router_sha": sha,
                        "degrade_reason": "router_unreachable:APITimeoutError"}])
    assert "m000" not in route_mod._prior("small")


# --------------------------------------------------------------------------- answer


def test_a_path_walked_backwards_still_reads_forwards():
    """The verbaliser's one real trap. `arrows([])` is `-[r0]-()`, undirected, so a
    neighbourhood walk enters half its edges against their stored direction. Reading the
    endpoints off the relationship (startNode/endNode) instead of off nodes(path) is what
    keeps "TSMC supplies AMD" from being published as "AMD supplies TSMC" -- an inversion
    that produces a confident, well-cited, exactly-backwards fact with no error anywhere."""
    from kgrag.route import verbalise

    # Both steps are the SAME edge; the second is what an inbound walk hands back.
    outbound = {"type": "SUPPLIES", "subject": "TSMC", "object": "AMD",
                "chunk_ids": ["c1"], "support": 6}
    facts = verbalise([outbound])
    assert facts[0]["text"] == "TSMC supplies AMD"

    # Same edge reached from AMD's side: Cypher still reports TSMC as the start node.
    facts = verbalise([{**outbound}])
    assert facts[0]["text"] == "TSMC supplies AMD", "direction comes from the edge, not the walk"

    # Deduped on the rendered text, and the cap is honoured.
    many = [{"type": "ACQUIRED", "subject": "A", "object": f"B{i}",
             "chunk_ids": [f"c{i}"], "support": 1} for i in range(50)]
    assert len(verbalise(many + many, limit=20)) == 20


def test_every_relation_renders_without_a_keyerror():
    """`verbalise` indexes RELATION_PHRASE by RelationType. A relation added to the
    ontology without a phrase would not fail at import or in the ontology self-check
    (which only compares key sets) -- it would fail at answer time, on the one question
    that happens to traverse it."""
    from kgrag.ontology import RelationType
    from kgrag.route import verbalise

    steps = [{"type": r.value, "subject": "S", "object": "O", "chunk_ids": ["c"], "support": 1}
             for r in RelationType]
    assert len(verbalise(steps, limit=99)) == len(RelationType)


def test_citable_ids_are_exactly_what_the_context_printed():
    """The bug this cost six abandoned answers. The context renders up to CITES_PER_FACT
    ids per graph fact, which is more ids than the route's top-k chunk list contains; a
    citable set computed anywhere but here drifts from what the model can see, and the
    model gets blamed for citing its own context."""
    from kgrag.answer import build_context

    facts = [{"text": "A acquired B", "chunk_ids": ["g1", "g2", "g3", "g4"], "support": 3}]
    texts = [{"chunk_id": "v1", "company": "X", "form": "10-K",
              "section_path": "Item 1", "filing_date": "2026-01-01", "text": "body"}]
    context, citable = build_context(facts, texts)

    assert citable == {"g1", "g2", "g3", "v1"}, "the 4th id is not printed, so it is not citable"
    assert "g4" not in context
    for shown in citable:
        assert f"[{shown}]" in context, "every citable id must appear in bracket form"
    assert "GRAPH FACTS" in context and "PASSAGES" in context, "the labels are the spec's ask"


def test_a_bracketed_id_is_a_delimiter_slip_but_a_wrong_id_is_not():
    """gpt-oss-120b cites "[abc]" for an id that really was retrieved, ~600 times over a
    57-question sweep. Counting that as invention overstates the rate and makes the
    free-vs-constrained comparison measure formatting rather than grounding. The strip has
    to be narrow enough that a genuinely fabricated id still fails."""
    from kgrag.answer import Answer, Claim, invented, normalise_citations

    ans = Answer(answerable=True, refusal_reason="", claims=[
        Claim(text="a", citations=["[c1]", " c2 ", "c3"]),
        Claim(text="b", citations=["deadbeefdeadbeef"]),
    ])
    assert normalise_citations(ans) == 2, "only the two that changed are counted"
    assert ans.claims[0].citations == ["c1", "c2", "c3"]
    assert invented(ans, {"c1", "c2", "c3"}) == ["deadbeefdeadbeef"], "a real miss survives"


def test_the_repair_prompt_differs_so_the_cache_cannot_replay_the_bad_answer():
    """chat_json keys its cache on (model, prompt version, system, user, schema). A repair
    that resends a byte-identical prompt is served the same invalid answer off disk
    forever, at $0.00, looking exactly like a model that refuses to correct itself. Naming
    the rejected ids is what makes the retry a retry."""
    from kgrag.answer import _user

    first = _user("Q", "CTX")
    repair = _user("Q", "CTX", ["badid1", "badid2"])
    assert repair != first
    assert "badid1" in repair and "badid2" in repair
    assert _user("Q", "CTX", []) == first, "no rejects means no repair preamble"


def test_the_constrained_arm_closes_citations_to_the_retrieved_set():
    """The constrained arm's whole claim: an invented citation is unreachable rather than
    caught. That holds only if the enum is the retrieved set, and the two arms must hash
    differently or they resume each other's rows out of the answer log."""
    from kgrag.answer import _schema, synth_sha

    ids = ["b2", "a1"]
    free = _schema(ids, False)["$defs"]["Claim"]["properties"]["citations"]["items"]
    con = _schema(ids, True)["$defs"]["Claim"]["properties"]["citations"]["items"]

    assert free == {"type": "string"}
    assert con == {"type": "string", "enum": ["a1", "b2"]}, "sorted, so the key is stable"
    assert synth_sha(True) != synth_sha(False)


def test_a_refused_question_never_reaches_the_model(monkeypatch):
    """A refusal that still pays for a synthesis call is not a refusal. The router's
    `refuse` and an empty context both have to short-circuit before chat_json."""
    from kgrag import answer as answer_mod

    def explode(**kwargs):
        raise AssertionError("the model must not be called for a refused question")

    monkeypatch.setattr(answer_mod.fireworks, "chat_json", explode)
    monkeypatch.setattr(answer_mod.jsonl, "append", lambda *a, **k: None)
    monkeypatch.setattr(answer_mod.route_mod, "route", lambda *a, **k: {
        "route": "refuse", "graph_ids": [], "vector_ids": [], "chunk_ids": [],
        "graph_facts": [], "question": "What is the capital of France?",
    })

    row = answer_mod.answer("What is the capital of France?", None, None, {}, log=False)
    assert row["answerable"] is False
    assert row["refusal_reason"] == "router_refused"
    assert row["attempts"] == 0 and row["usd"] == 0.0


def test_one_arm_does_not_evict_the_other_arms_resumable_rows(tmp_path, monkeypatch):
    """Both citation arms append to one answer log. Treating the other arm's rows as a
    superseded prompt pops them, so running free then constrained leaves each arm with
    nothing to resume -- "0 resumed" for 57 questions that are all sitting on disk. A
    resume that silently never resumes is worse than no resume: it looks like it worked."""
    from kgrag import answer as answer_mod
    from kgrag import jsonl

    log = tmp_path / "answer_log.jsonl"
    monkeypatch.setattr(answer_mod, "ANSWER_LOG", log)
    model = answer_mod.SYNTH_MODEL

    def row(qid, constrain, sha=None):
        return {"qid": qid, "model": model, "arm": "constrained" if constrain else "free",
                "synth_sha": sha or answer_mod.synth_sha(constrain), "refusal_reason": ""}

    # free arm answered first, then the constrained arm ran the same questions.
    jsonl.append(log, [row("m000", False), row("m001", False)])
    jsonl.append(log, [row("m000", True), row("m001", True)])

    assert set(answer_mod._prior(model, constrain=False)) == {"m000", "m001"}
    assert set(answer_mod._prior(model, constrain=True)) == {"m000", "m001"}

    # A stale prompt within the SAME arm still supersedes: that row is not resumable.
    jsonl.append(log, [row("m000", False, sha="stale0000000")])
    assert set(answer_mod._prior(model, constrain=False)) == {"m001"}


# --------------------------------------------------------------------------- aggregation


def test_an_aggregate_reads_correctly_in_both_directions():
    """Direction is the whole meaning of an aggregate. 32 companies pointing SUBSIDIARY_OF
    at AMD means AMD has 32 subsidiaries; AMD pointing SUBSIDIARY_OF at 32 things would
    mean AMD is owned by all of them. Inbound needs a plural noun because the subject is a
    count -- reusing the phrase yields "32 distinct entities is a subsidiary of AMD"."""
    from kgrag.route import verbalise_aggregates

    inbound = {"anchor": "AMD", "predicate": "SUBSIDIARY_OF", "outbound": False, "n": 32,
               "examples": ["Xilinx, Inc.", "AMD Japan Ltd."], "chunk_ids": ["c1"]}
    outbound = {"anchor": "AMD", "predicate": "OPERATES_IN", "outbound": True, "n": 11,
                "examples": ["China"], "chunk_ids": ["c2"]}

    text = verbalise_aggregates([inbound])[0]["text"]
    assert text.startswith("AMD has 32 subsidiaries")
    assert "and 30 others" in text, "the sample must say how much it is not showing"
    assert "; " in text, "names contain commas, so items cannot be comma-separated"

    # Outbound pluralises the object type from the ontology: Location -> Locations.
    assert verbalise_aggregates([outbound])[0]["text"].startswith(
        "AMD has operations in 11 distinct Locations"
    )


def test_a_count_of_one_is_not_an_aggregate():
    """RELATION_NOUN is plural, so n=1 renders as "has 1 subsidiaries". A single edge is
    already carried by the ranked path facts and does not need a count sentence."""
    from kgrag.route import verbalise_aggregates

    single = {"anchor": "X", "predicate": "SUBSIDIARY_OF", "outbound": False, "n": 1,
              "examples": ["Y"], "chunk_ids": ["c1"]}
    assert verbalise_aggregates([single]) == []


def test_aggregates_are_citable_and_labelled_separately_from_facts():
    """A count is a different kind of evidence from a path fact and from a passage: it is
    computed, complete, and not stated by any single chunk. It gets its own block so the
    prompt can tell the model to quote it rather than recount, and its chunk ids have to be
    citable or every count claim fails validation."""
    from kgrag.answer import build_context

    aggs = [{"text": "AMD has 32 subsidiaries.", "chunk_ids": ["g1", "g2"], "support": 32}]
    context, citable = build_context([], [], aggs)
    assert "GRAPH COUNTS" in context and "do not recount" in context
    assert {"g1", "g2"} <= citable
    assert "[g1]" in context


def test_aggregation_is_its_own_stratum_not_a_one_hop_question():
    """Folding aggregation into the 1-hop slice would move a published Phase 2 number and
    shift the recall floor's baseline underneath it. It also has to reach the graph: no ten
    passages contain a corpus-wide total, so vector cannot be right regardless of hops."""
    from kgrag.retrieve import _slices
    from kgrag.route import expected_route

    agg = {"hops": 1, "source": "hand", "category": "aggregation", "gold_chunk_ids": ["c"]}
    plain = {"hops": 1, "source": "hand", "gold_chunk_ids": ["c"]}

    assert expected_route(agg) == {"graph", "both"}
    assert expected_route(plain) == {"vector", "both"}, "unchanged for real 1-hop questions"

    labels = dict(_slices([plain, agg]))
    assert labels["1-hop"] == [0], "the aggregation row must not land in the hop slice"
    assert labels["aggregation"] == [1]


def test_a_count_stated_in_words_scores_as_correct():
    """The aggregation slice is the project's only judgement-free accuracy metric, so its
    scorer has to be right: gpt-oss-120b answers "audits nine companies" where the graph
    says 9, and a digits-only scorer marks a correct answer wrong."""
    from kgrag.answer import stated_numbers

    assert stated_numbers("audits nine companies") == {9}
    assert stated_numbers("lists 67 subsidiaries") == {67}
    assert stated_numbers("1,234 filings and three others") == {1234, 3}
    assert 32 in stated_numbers("AMD lists 32 subsidiaries across 11 locations")


# --------------------------------------------------------------------------- phase 5


def test_the_baseline_never_touches_the_graph(monkeypatch):
    """"Plain vector RAG" has to be a different system, not this one with the graph
    subtracted. If the baseline still routes, its latency and cost carry a router call the
    system it stands in for does not have, and the published delta is measured against
    something that does not exist."""
    from kgrag import answer as answer_mod

    def explode(*a, **k):
        raise AssertionError("the baseline must not call the router")

    monkeypatch.setattr(answer_mod.route_mod, "route", explode)
    monkeypatch.setattr(answer_mod.route_mod, "vector_path", lambda *a, **k: ["c1"])
    monkeypatch.setattr(answer_mod, "passages", lambda conn, ids: [
        {"chunk_id": "c1", "company": "INTEL CORP", "form": "10-K",
         "section_path": "Item 1", "filing_date": "2025-11-01", "text": "Intel text."}
    ])
    monkeypatch.setattr(answer_mod.fireworks, "chat_json", lambda **k: {
        "answerable": True, "refusal_reason": "",
        "claims": [{"text": "Intel filed a 10-K.", "citations": ["c1"]}],
    })

    row = answer_mod.answer("Who audits Intel?", None, None, {}, graph=False, log=False)
    assert row["arm"] == "vector" and row["route"] == "vector-only"
    assert row["n_facts"] == 0 and row["n_aggregates"] == 0
    assert row["cited"] == ["c1"] and not row["invented"]


def test_the_three_arms_never_resume_each_others_rows(tmp_path, monkeypatch):
    """Phase 4 shipped this bug with two arms. The baseline makes three, and a baseline
    that resumed the graph arm's answers would publish a benchmark it never ran."""
    from kgrag import answer as answer_mod
    from kgrag import jsonl

    log = tmp_path / "answer_log.jsonl"
    monkeypatch.setattr(answer_mod, "ANSWER_LOG", log)
    model = answer_mod.SYNTH_MODEL

    def row(qid, constrain, graph):
        return {"qid": qid, "model": model, "arm": answer_mod.arm_of(constrain, graph),
                "synth_sha": answer_mod.synth_sha(constrain), "refusal_reason": ""}

    jsonl.append(log, [row("m000", False, True), row("m000", True, True), row("m000", True, False)])
    assert answer_mod.arm_of(True, False) == "vector"
    for constrain, graph in ((False, True), (True, True), (True, False)):
        assert set(answer_mod._prior(model, constrain, graph)) == {"m000"}
    # The baseline is a distinct arm even at the same enforcement setting.
    jsonl.append(log, [{"qid": "m001", "model": model, "arm": "vector",
                        "synth_sha": answer_mod.synth_sha(True), "refusal_reason": ""}])
    assert set(answer_mod._prior(model, True, graph=True)) == {"m000"}
    assert set(answer_mod._prior(model, True, graph=False)) == {"m000", "m001"}


def test_the_answer_key_accepts_an_alias_and_rejects_a_near_miss():
    """The key is judgement-free, which means it is also literal. "Ernst & Young (EY)" is
    the right answer to "who audits Intel" and does not contain "Ernst & Young LLP"; the
    alias set resolution already mined is what closes that gap. It must not close so far
    that AMD Japan Ltd. counts as AMD -- the same collision the entity index was built for."""
    from kgrag.judge import key_score, names_it

    index = {
        "ernst and young llp": [{"canonical_id": "e1", "name": "Ernst & Young LLP",
                                 "aliases": ["EY", "Ernst & Young"]}],
        "advanced micro devices": [{"canonical_id": "e2", "name": "ADVANCED MICRO DEVICES INC",
                                    "aliases": ["AMD"]}],
    }
    assert names_it("Intel's auditor is Ernst & Young (EY).", "Ernst & Young LLP", index)
    assert names_it("AMD acquired Xilinx.", "ADVANCED MICRO DEVICES INC", index)
    assert not names_it("The auditor is Deloitte & Touche LLP.", "Ernst & Young LLP", index)
    # Tokens, not substrings: a two-letter alias must not match inside another word, or
    # every answer containing "they" would name Ernst & Young.
    assert not names_it("They surveyed the filings.", "Ernst & Young LLP", index)
    # ponytail: a one-token alias over-credits -- "AMD Japan Ltd." contains "AMD", so the
    # key reads it as naming AMD. Pinned rather than fixed: the key is a floor with a known
    # bias toward the graph arm, and the judge cross-check is what catches this class.
    assert names_it("AMD Japan Ltd. is a subsidiary.", "ADVANCED MICRO DEVICES INC", index)

    q = {"hops": 1, "expected_answers": ["Ernst & Young LLP"], "gold_chunk_ids": ["c1"]}
    answered = {"answerable": True, "claims": [{"text": "Intel's auditor is EY."}]}
    refused = {"answerable": False, "claims": [], "refusal_reason": "no fact"}
    assert key_score(q, answered, index) == "correct"
    assert key_score(q, refused, index) == "incorrect"
    # No key is not the same as a wrong answer: SUPPLIES and OFFICER_OF questions are the
    # judge's alone, and scoring them 0 here would invent a failure.
    assert key_score({**q, "expected_answers": []}, answered, index) is None
    # Out-of-scope: refusing IS the correct answer, and no model is consulted for that.
    assert key_score({"hops": 0, "gold_chunk_ids": []}, refused, index) == "correct"
    assert key_score({"hops": 0, "gold_chunk_ids": []}, answered, index) == "incorrect"


def test_a_chain_renders_as_one_fact_ending_at_the_answer():
    """Phase 5 measured what a flattened walk costs: asked who competes with the customers
    Teradyne supplies, the system answered with Teradyne's own competitors. It had both
    facts and no way to tell which was the last step. A path renders whole, so the far end
    is where it looks like it is -- and the cap now counts paths, so a chain can no longer
    lose its terminal edge (the least-corroborated one) while its first edge survives."""
    from kgrag.route import verbalise

    walk = [
        {"type": "SUPPLIES", "subject": "TERADYNE, INC", "object": "Lattice",
         "chunk_ids": ["c1", "c2"], "support": 4, "path": 0, "hop": 0},
        {"type": "COMPETES_WITH", "subject": "Lattice", "object": "Xilinx",
         "chunk_ids": ["c9"], "support": 1, "path": 0, "hop": 1},
    ]
    facts = verbalise(walk)
    assert len(facts) == 1, "one walk is one fact, not two"
    assert facts[0]["text"] == (
        "TERADYNE, INC supplies Lattice → Lattice competes with Xilinx"
    )
    # One id per hop first, so the three ids build_context shows cover the whole walk
    # rather than three ids from its first edge.
    assert facts[0]["chunk_ids"][:2] == ["c1", "c9"]
    assert facts[0]["support"] == 1, "a chain is only as corroborated as its weakest step"

    # The cap counts walks. Twenty two-step chains are twenty facts, none of them halved.
    many = [
        {"type": "ACQUIRED", "subject": "A", "object": f"B{i}", "chunk_ids": [f"c{i}a"],
         "support": 2, "path": i, "hop": 0}
        for i in range(30)
    ] + [
        {"type": "OFFERS", "subject": f"B{i}", "object": f"P{i}", "chunk_ids": [f"c{i}b"],
         "support": 1, "path": i, "hop": 1}
        for i in range(30)
    ]
    facts = verbalise(sorted(many, key=lambda s: (s["path"], s["hop"])), limit=20)
    assert len(facts) == 20
    assert all(" → " in f["text"] for f in facts), "no walk was truncated mid-chain"


def test_no_cache_reaches_the_router_not_just_the_synthesiser(monkeypatch):
    """Phase 5's first latency table was measured with --no-cache and still read the router
    out of cache: 16 ms instead of a ~6.7s call, on the arm being compared against a
    baseline that has no router at all. The flag has to reach every model call the query
    makes, or the number describes the filesystem. Third occurrence in this project."""
    from kgrag import answer as answer_mod
    from kgrag import route as route_mod

    seen: list[bool] = []

    def spy(**kwargs):
        seen.append(kwargs.get("use_cache", True))
        return {"route": "vector", "confidence": "high", "entities": [],
                "shape": "neighbourhood", "chain": []}

    monkeypatch.setattr(route_mod.fireworks, "chat_json", spy)
    route_mod.make_plan("q", use_cache=False)
    route_mod.make_plan("q", use_cache=True)
    assert seen == [False, True], "the router must honour the flag it is handed"

    # ...and `answer` must hand it down rather than defaulting it back to True.
    monkeypatch.setattr(answer_mod.route_mod, "route",
                        lambda *a, **k: seen.append(k.get("use_cache", True)) or {
                            "route": "refuse", "graph_ids": [], "vector_ids": [],
                            "chunk_ids": [], "graph_facts": []})
    monkeypatch.setattr(answer_mod, "passages", lambda conn, ids: [])
    answer_mod.answer("q", None, None, {}, use_cache=False, log=False)
    assert seen[-1] is False


def test_the_claim_scanner_emits_only_finished_claims():
    """A claim may not be published before its citations are validated, and a citation
    cannot be validated from half a token. The scanner therefore hands on a claim only when
    its object closes -- and must not close early on a brace inside a string, which an
    entity name can legitimately contain."""
    from kgrag.answer import ClaimScanner

    doc = (
        '{"answerable": true, "claims": ['
        '{"text": "Intel is audited by E&Y {sic}.", "citations": ["c1"]}, '
        '{"text": "AMD acquired Xilinx.", "citations": ["c2", "c3"]}'
        '], "refusal_reason": ""}'
    )
    # Fed one character at a time: the worst case a token stream can produce.
    scanner, got = ClaimScanner(), []
    for ch in doc:
        got += scanner.feed(ch)
    assert [c["text"] for c in got] == [
        "Intel is audited by E&Y {sic}.", "AMD acquired Xilinx."
    ]
    assert got[1]["citations"] == ["c2", "c3"]

    # Nothing is emitted from a document that stops mid-claim.
    partial = ClaimScanner()
    assert partial.feed('{"answerable": true, "claims": [{"text": "half a cl') == []


def test_streaming_never_publishes_a_claim_it_could_not_validate(monkeypatch):
    """Constrained decoding makes an invented citation unreachable, and the per-claim check
    still runs: the enum makes it impossible in theory, the set membership makes it
    impossible in fact. A claim that fails is withheld rather than corrected, because a
    published claim cannot be un-published."""
    from kgrag import answer as answer_mod

    doc = (
        '{"answerable": true, "claims": ['
        '{"text": "good", "citations": ["c1"]}, '
        '{"text": "bad", "citations": ["nope"]}, '
        '{"text": "uncited", "citations": []}'
        '], "refusal_reason": ""}'
    )
    monkeypatch.setattr(answer_mod.fireworks, "chat_stream",
                        lambda **k: iter([doc[i:i + 7] for i in range(0, len(doc), 7)]))

    events = list(answer_mod.stream("q", "ctx", {"c1"}))
    claims = [c for kind, c in events if kind == "claim"]
    assert [c.text for c in claims] == ["good"], "an unvalidatable claim is never yielded"
    kind, final = events[-1]
    assert kind == "done" and final.answerable is True


def test_a_streamed_call_and_a_plain_one_share_one_cache_entry(monkeypatch, tmp_path):
    """Streaming is a delivery decision, not a different question. If the keys diverged,
    every answer would be paid for twice and the two paths could return different text for
    the same input -- which would make the streamed endpoint unmeasurable against the
    benchmark."""
    import hashlib
    import json as _json

    from kgrag import fireworks

    args = {"system": "S", "user": "U", "schema": {"type": "object"},
            "model": fireworks.EXTRACT_MODEL}
    key = hashlib.sha256(
        _json.dumps([args["model"], fireworks.PROMPT_VERSION, args["system"], args["user"],
                     args["schema"]], sort_keys=True).encode()
    ).hexdigest()

    monkeypatch.setattr(fireworks, "CACHE", tmp_path)
    monkeypatch.setattr(fireworks, "_cache_path", lambda k: tmp_path / f"{k}.json")
    (tmp_path / f"{key}.json").write_text('{"answerable": false, "claims": []}')

    streamed = "".join(fireworks.chat_stream(**args))
    assert _json.loads(streamed) == fireworks.chat_json(**args)


def test_reasoning_effort_is_in_the_cache_key_but_only_when_set():
    """A low-effort answer is a different answer. If the key ignored the setting, the
    experiment would be served the default-effort answers back off disk and would report
    that reasoning effort changes nothing -- at $0.00, instantly, and wrongly. And it must
    be absent from the key when unset, or every cache entry written before this parameter
    existed is orphaned. Same asymmetry as `dimensions` in _embed_key."""
    from kgrag.fireworks import _chat_key
    from kgrag.answer import synth_sha

    args = ("model", "sys", "usr", {"type": "object"})
    assert _chat_key(*args, None) == _chat_key(*args, None)
    assert _chat_key(*args, "low") != _chat_key(*args, None)
    assert _chat_key(*args, "low") != _chat_key(*args, "high")

    # synth_sha carries it too, so the two runs cannot resume each other's rows.
    assert synth_sha(True) == synth_sha(True, reasoning=None)
    assert synth_sha(True, reasoning="low") != synth_sha(True)


def test_reported_latency_excludes_rate_limit_sleep(monkeypatch):
    """`_pace()` holds this process to 9 requests/minute against a free account's 10 RPM
    quota -- 6,667 ms per call, more than the model call it precedes. Counted as latency,
    a sweep reports the quota: three different configurations of this system measured a p50
    within 30 ms of the pacer floor, and the graph arm read as "2x slower" purely because it
    makes two calls where the baseline makes one."""
    from kgrag import answer as answer_mod
    from kgrag import fireworks

    def slow_call(**kwargs):
        fireworks.METER.paced_ms += 6_667  # what a paced call adds
        return {"answerable": True, "refusal_reason": "",
                "claims": [{"text": "t", "citations": ["c1"]}]}

    monkeypatch.setattr(answer_mod.fireworks, "chat_json", slow_call)
    monkeypatch.setattr(answer_mod.route_mod, "vector_path", lambda *a, **k: ["c1"])
    monkeypatch.setattr(answer_mod, "passages", lambda conn, ids: [
        {"chunk_id": "c1", "company": "C", "form": "10-K", "section_path": "1",
         "filing_date": "2025-01-01", "text": "text"}])

    row = answer_mod.answer("q", None, None, {}, graph=False, log=False)
    assert row["paced_ms"] == 6_667, "the sleep is recorded, not discarded"
    assert row["latency_ms"] < 1_000, "and it is not reported as latency"

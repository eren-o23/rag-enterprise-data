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

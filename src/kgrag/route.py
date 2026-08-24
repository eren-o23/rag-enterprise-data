"""`kgrag route` — decide which store answers a question, and measure the decision.

Both stores are built and both were measured in isolation. This module is the thing that
picks between them, and the design rule is that the model never authors a query.

Three layers, in this order:

1. **Constrained decoding.** One `chat_json` call returns the route, the entities named in
   the question, and a chain of relation types — all of them enums inside the JSON schema.
   An off-vocabulary route or an invented predicate is unreachable, not merely rejected.
   `ontology.py` rests on the same guarantee at extraction time.
2. **Validation the schema cannot express.** A chain whose length contradicts its declared
   shape, or entities that resolve to no node, mean the graph path would run a query the
   question does not support. Those degrade to another route with a logged reason.
3. **Parameterised traversal.** `load.py` interpolates a relationship type into Cypher only
   after it survives `RelationType(...)`; read time uses the identical rule. The model
   supplies enum members and an entity name — never Cypher.

Both paths return the same currency: a ranked list of `chunk_id`s. That is what makes them
comparable against the same gold sets, and what lets Phase 4 cite either one.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import statistics
import time
from collections.abc import Iterable
from enum import StrEnum
from typing import Any, Literal

import psycopg
from neo4j import Session
from openai import APIConnectionError, APIStatusError
from pydantic import BaseModel, Field, ValidationError

from . import fireworks, jsonl, load, retrieve
from .config import ENTITIES, ROUTING_LOG
from .embed import PRODUCTION_WIDTH, connect
from .extract import _strict
from .ontology import ALLOWED_EDGES, RELATION_NOUN, RELATION_PHRASE, RelationType
from .questions import QUESTIONS
from .resolve import normalize

#: The spec calls for "a cheap router: a small model call", and gpt-oss-20b was the
#: obvious candidate -- half the input price of the extraction model and healthy latency
#: of 3-7s. It is not viable, and the failure is not slowness.
#:
#: Six of 57 questions stall it reproducibly: the same six, on every rerun, and raising
#: the deadline does not help. "What is home automation solutions, and who sells it?"
#: times out at 45s on 20b and is answered by 120b in **1.2s**. A failed call costs
#: nothing, so the tell was a rerun reporting $0.00000 while six questions stayed
#: unrouted -- not a bill, and not an error rate.
#:
#: So the router runs on the same model as extraction. This is Phase 1's llama3.2:3b
#: result again (5/20 hard failures, not a quality gap): a smaller model that answers
#: most inputs well and hangs on some is unusable at any price, because the failures
#: are not random and do not go away on retry. `--router-model` still swaps it.
ROUTER_MODEL = "accounts/fireworks/models/gpt-oss-120b"

TOP_K = 10

#: The router emits ~50 tokens against a fixed schema. `fireworks.DEFAULT_TIMEOUT` is 90s,
#: sized for extraction streaming a whole chunk's worth of JSON, and inheriting it here is
#: how a 57-question eval turns into hours: Fireworks' serverless endpoint intermittently
#: stalls, and 6 backoff attempts at 90s is nine minutes burned on ONE question. Measured
#: healthy latency is 3-7s for both gpt-oss-20b and gpt-oss-120b, so anything past 20s is
#: a stall, not a slow answer. Three attempts, then the question routes to `both`, which is
#: already the defined answer for "the router could not decide".
ROUTER_TIMEOUT = 20.0
ROUTER_ATTEMPTS = 3

#: How many paths one traversal returns before ranking. Generous: paths are deduped down
#: to chunk_ids afterwards, and a hub entity fans out hard.
PATH_LIMIT = 200


class Route(StrEnum):
    GRAPH = "graph"
    VECTOR = "vector"
    BOTH = "both"
    REFUSE = "refuse"


class Shape(StrEnum):
    ONE_HOP = "one_hop"
    TWO_HOP = "two_hop"
    THREE_HOP = "three_hop"
    NEIGHBOURHOOD = "neighbourhood"


#: A shape is a promise about chain length. Breaking it is layer 2's business.
CHAIN_LEN: dict[Shape, int] = {
    Shape.ONE_HOP: 1,
    Shape.TWO_HOP: 2,
    Shape.THREE_HOP: 3,
    Shape.NEIGHBOURHOOD: 0,
}


class Plan(BaseModel):
    """What the router returns. Every constrained field is an enum on purpose."""

    route: Route
    confidence: Literal["high", "low"]
    entities: list[str] = Field(
        description="Company, person, or product names as written in the question. "
        "Copy the surface form; do not expand or normalise it."
    )
    shape: Shape
    chain: list[RelationType] = Field(
        description="The relation types to traverse, in order, starting at the entity. "
        "Length must match the shape. Empty for neighbourhood."
    )


SCHEMA = _strict(Plan.model_json_schema())


def router_sha() -> str:
    """Fingerprint of everything that decides how the router behaves.

    It goes in every logged row and `_prior` refuses to resume across a change in it.
    `fireworks.chat_json` already puts the prompt in its cache key so an edit invalidates
    exactly the affected entries; without the same idea here, the resume happily replays
    decisions made by a prompt that no longer exists. That is not hypothetical -- editing
    the corpus description and rerunning reported byte-identical numbers and $0.00000
    spend, because all 57 rows resumed and not one router call was made.

    `--fresh` still exists for changes this cannot see, like traversal or ranking code.
    """
    return hashlib.sha256(f"{SYSTEM}|{json.dumps(SCHEMA, sort_keys=True)}".encode()).hexdigest()[:12]

SYSTEM = f"""You route questions about SEC filings to one of two retrieval systems.

CORPUS: 10-K, DEF 14A and 8-K filings from 24 US-listed semiconductor companies (AMD, \
Intel, NVIDIA, Broadcom, Qualcomm, Micron, TI, Applied Materials, Lam Research, KLA, \
Marvell, NXP, Analog Devices, Microchip, ON Semiconductor, Skyworks, Qorvo, Teradyne, \
Entegris, Wolfspeed, Cirrus Logic, Silicon Labs, Monolithic Power, Allegro MicroSystems), \
filed between August 2025 and August 2026.

Those 24 are the FILERS, not the scope. Their filings name thousands of other entities and
all of them are in the corpus: subsidiaries (Exhibit 21 lists them by name), directors and
officers, auditors and their staff, named customers, suppliers, competitors and partners,
products and platforms, states and countries of incorporation, regulators, and parties to
legal proceedings. A question about any of those is answerable. Refuse only when the
subject has no connection to these filings at all.

SYSTEMS:
- graph — a knowledge graph of entities and these relations: {", ".join(r.value for r in RelationType)}.
  Route here for connections between entities, chains that pass through an entity the
  question does not name, comparisons across companies, and aggregations over relationships.
- vector — semantic search over the filing text. Route here for definitions, policy and
  risk-language lookups, paraphrased single facts, and anything whose answer is a passage
  rather than a connection.
- both — the question needs a connection AND passage context, or you are genuinely unsure.
- refuse — the corpus cannot answer it: general knowledge, market or price data, forward
  guidance that filings do not disclose, advice, or an entity these filings never mention.
  Do NOT refuse merely because the subject is not one of the 24 filers -- a subsidiary,
  a director, an auditor, a product or a litigation counterparty is normal corpus content.

FIELDS:
- entities: names the question mentions, copied verbatim. Empty when it names none.
- shape + chain: the traversal, only used when route is graph or both. The chain starts at
  the FIRST entity and its length MUST equal the shape (one_hop=1, two_hop=2, three_hop=3,
  neighbourhood=0). Use neighbourhood when no fixed chain fits, e.g. counting or summarising
  everything around one entity.
- confidence: "low" whenever the choice is close. Low confidence runs both paths, which is
  cheap; a confidently wrong route is not.

EXAMPLES:
Q: Who is INTEL CORP's independent auditor?
-> graph, high, entities ["INTEL CORP"], one_hop, [AUDITED_BY]

Q: Which company was acquired by a company that Karl-Henrik Sundstrom is a director of?
-> graph, high, entities ["Karl-Henrik Sundstrom"], two_hop, [DIRECTOR_OF, ACQUIRED]

Q: Where do the competitors of NVIDIA CORP's parent company operate?
-> graph, high, entities ["NVIDIA CORP"], three_hop, [SUBSIDIARY_OF, COMPETES_WITH, OPERATES_IN]

Q: How many subsidiaries does AMD list, and in how many countries?
-> graph, high, entities ["AMD"], neighbourhood, []

Q: Does Marvell own the factories that make its chips?
-> vector, high, entities ["Marvell"], neighbourhood, []

Q: What could go wrong for Applied Materials if China limits raw material exports?
-> vector, high, entities ["Applied Materials"], neighbourhood, []

Q: Where is Picosun Japan Co., Ltd. incorporated?
-> graph, high, entities ["Picosun Japan Co., Ltd."], one_hop, [INCORPORATED_IN]

Q: What executive role does Ms. Simon hold at Deloitte & Touche LLP?
-> graph, high, entities ["Ms. Simon", "Deloitte & Touche LLP"], one_hop, [OFFICER_OF]

Q: What is the capital of France?
-> refuse, high, entities [], neighbourhood, []

Q: What is Intel's revenue forecast for 2031?
-> refuse, high, entities ["Intel"], neighbourhood, []
"""


# ---------------------------------------------------------------------------
# Entity resolution: question surface form -> canonical node id
# ---------------------------------------------------------------------------

#: Lucene syntax characters. A question entity like "Picosun Japan Co., Ltd." or a name
#: containing a bracket is a parse error, not a miss, if it reaches the fulltext index raw.
_LUCENE = re.compile(r'[+\-&|!(){}\[\]^"~*?:\\/]')


def entity_index(entities: Iterable[dict[str, Any]] | None = None) -> dict[str, list[dict[str, Any]]]:
    """normalized surface -> candidate entities, best first.

    `cypher/schema.cypher` created a fulltext index on `[e.name, e.aliases]` for exactly
    this lookup, and on the real graph it does not survive contact: "AMD" returns
    `AMD Ryzen(TM) PRO`, and filtered to companies it returns `AMD (EMEA) LTD.` and
    `AMD Japan Ltd.`. `ADVANCED MICRO DEVICES INC` does carry "AMD" as an alias, but it
    carries ten others too, and Lucene's length normalisation buries it below tiny nodes
    literally named "AMD ...". See docs/decisions.md.

    An exact index over the same aliases, keyed on `resolve.normalize`, resolves every
    case cleanly: 4,629 normalized surfaces, 28 of them ambiguous (mostly Location/Company
    collisions -- "intel", "arm", "china"). Ambiguity is broken by mention_count, which is
    already on every entity. The fulltext index stays as the fallback for surfaces the
    model paraphrases into something the alias set never saw.

    Takes an iterable so it is testable without the gitignored data file, the same way
    `embed.chunk_entities` does.
    """
    index: dict[str, list[dict[str, Any]]] = {}
    for entity in jsonl.read(ENTITIES) if entities is None else entities:
        for surface in [entity["name"], *entity["aliases"]]:
            bucket = index.setdefault(normalize(surface), [])
            if not any(c["canonical_id"] == entity["canonical_id"] for c in bucket):
                bucket.append(entity)
    for bucket in index.values():
        bucket.sort(key=lambda e: -e.get("mention_count", 0))
    return index


def resolve_entity(
    name: str, index: dict[str, list[dict[str, Any]]], session: Session | None = None
) -> str | None:
    exact = index.get(normalize(name))
    if exact:
        return exact[0]["canonical_id"]
    if session is None:
        return None
    query = _LUCENE.sub(" ", name).strip()
    if not query:
        return None
    row = session.run(
        "CALL db.index.fulltext.queryNodes('entity_search', $q) YIELD node, score "
        "RETURN node.id AS id ORDER BY score DESC LIMIT 1",
        q=query,
    ).single()
    return row["id"] if row else None


# ---------------------------------------------------------------------------
# The graph path
# ---------------------------------------------------------------------------

#: `startNode`/`endNode` rather than `nodes(path)`, and the difference is not cosmetic.
#: The neighbourhood pattern is `-[r0]-()`, undirected, so path order says which way the
#: traversal walked and NOT which end is the subject. Reading the endpoints off the
#: relationship gives the stored direction whichever way the walk went; reading them off
#: the path inverts every fact a neighbourhood walk happens to enter backwards, turning
#: "Xilinx is a subsidiary of AMD" into its opposite with no error anywhere.
TRAVERSE = """
MATCH path = (a:Entity {{id: $id}}){arrows}
RETURN [r IN relationships(path) | {{
         type: type(r), subject: startNode(r).name, object: endNode(r).name,
         chunk_ids: r.chunk_ids, support: coalesce(r.support, 1)}}] AS steps,
       reduce(s = 0, r IN relationships(path) | s + coalesce(r.support, 1)) AS support
ORDER BY support DESC LIMIT $limit
"""


def arrows(chain: Iterable[str]) -> str:
    """Render a validated predicate chain as Cypher relationship patterns.

    Cypher cannot parameterise a relationship type, so it has to be interpolated. The
    security control is that `RelationType(...)` raises on anything that is not an ontology
    member, so the interpolated value can only ever be one of fourteen literals -- exactly
    the rule `load.py` applies to writes. This is the reason the model returns enum members
    instead of a query string.
    """
    parts = [f"-[r{i}:{RelationType(p).value}]->()" for i, p in enumerate(chain)]
    return "".join(parts) if parts else "-[r0]-()"  # neighbourhood: any relation, either way


def graph_steps(session: Session, node_ids: list[str], chain: list[str]) -> list[dict[str, Any]]:
    """Traverse from each resolved entity and return the edges walked, best path first.

    A single 1-hop query fans out -- Teradyne has nine ACQUIRED edges whose chunk_ids
    overlap heavily -- so the result needs an order. `support` (how many chunks assert the
    edge) is already stored on every edge by `load.py`; summing it along a path ranks
    well-corroborated evidence first at no extra cost.

    Phase 3 only needed the chunk ids off these paths. Phase 4 has to say what the path
    *means*, which needs the predicate and both endpoint names -- so the edges come back
    whole and `graph_path` derives the ids from them. One query, two readings of it.
    """
    statement = TRAVERSE.format(arrows=arrows(chain))
    paths: list[tuple[int, list[dict[str, Any]]]] = []
    for node_id in node_ids:
        for row in session.run(statement, id=node_id, limit=PATH_LIMIT):
            paths.append((row["support"], row["steps"]))
    paths.sort(key=lambda pair: -pair[0])
    return [{**step, "path_support": support} for support, steps in paths for step in steps]


def graph_path(session: Session, node_ids: list[str], chain: list[str], k: int) -> list[str]:
    """The ranked chunk ids off those paths -- Phase 3's currency, unchanged."""
    return chunk_ids_of(graph_steps(session, node_ids, chain), k)


def chunk_ids_of(steps: Iterable[dict[str, Any]], k: int) -> list[str]:
    """Deduped chunk ids in step order, capped at k. Ranking already happened upstream."""
    out: list[str] = []
    for step in steps:
        for chunk_id in step["chunk_ids"] or []:
            if chunk_id not in out:
                out.append(chunk_id)
            if len(out) == k:
                return out
    return out


#: Facts kept per question. A hub entity's neighbourhood walk returns hundreds of edges
#: and they are already ranked, so the tail is corroborated worst and costs prompt tokens.
FACT_LIMIT = 20


def verbalise(steps: Iterable[dict[str, Any]], limit: int = FACT_LIMIT) -> list[dict[str, Any]]:
    """Graph edges -> readable statements, each carrying the chunks that justified it.

    Phase 4's spec is explicit that raw triples generate awkward text, and the fix is one
    phrase per relation in `ontology.RELATION_PHRASE` -- rendered subject-first off the
    edge's own direction, so a neighbourhood walk that entered an edge backwards still
    reads forwards.

    Deduped on the rendered text: two paths through the same hub repeat their shared edge,
    and the same fact stated twice is prompt tokens spent to say nothing. Steps arrive
    ranked, so the first sighting of a fact is its best-corroborated one.
    """
    seen: dict[str, dict[str, Any]] = {}
    for step in steps:
        phrase = RELATION_PHRASE[RelationType(step["type"])]
        text = f"{step['subject']} {phrase} {step['object']}"
        if text not in seen:
            seen[text] = {
                "text": text,
                "chunk_ids": list(step["chunk_ids"] or []),
                "support": step["support"],
            }
        if len(seen) == limit:
            break
    return list(seen.values())


#: Aggregate facts kept per question, ranked by size, and how many neighbour names each
#: one names before it says "and N others".
AGG_LIMIT = 12
AGG_EXAMPLES = 8

#: An aggregation question asks about the SHAPE of a neighbourhood -- how many, in how many
#: places -- and a ranked top-k walk structurally cannot answer it. `graph_path` returns the
#: ten best-corroborated chunk ids around AMD, which are risk and auditor edges; the 32
#: SUBSIDIARY_OF edges the question is about never make the cut. You cannot count from a
#: sample.
#:
#: So the count is computed by the database, over the whole neighbourhood, with no cap.
#: Cypher counts exactly and an LLM does not, which is the entire reason this is a query and
#: not a prompt instruction. Grouping carries direction, because inbound and outbound mean
#: opposite things: 32 companies point SUBSIDIARY_OF at AMD (AMD has 32 subsidiaries),
#: while AMD pointing SUBSIDIARY_OF at something would mean AMD is owned by it.
AGGREGATE = """
MATCH (a:Entity {id: $id})-[r]-(o)
WITH a, type(r) AS predicate, startNode(r).id = $id AS outbound,
     collect(DISTINCT o.name) AS names,
     apoc.coll.toSet(apoc.coll.flatten(collect(coalesce(r.chunk_ids, [])))) AS chunk_ids
RETURN a.name AS anchor, predicate, outbound, size(names) AS n,
       names[..$examples] AS examples, chunk_ids
ORDER BY n DESC
"""


def _plural(noun: str) -> str:
    """Entity type names, pluralised. Seven values, so the naive rule is the whole rule."""
    return noun[:-1] + "ies" if noun.endswith("y") else noun + "s"


def graph_aggregates(session: Session, node_ids: list[str]) -> list[dict[str, Any]]:
    """Exact per-predicate neighbour counts around each anchor. No cap, no ranking loss."""
    out: list[dict[str, Any]] = []
    for node_id in node_ids:
        for row in session.run(AGGREGATE, id=node_id, examples=AGG_EXAMPLES):
            out.append(dict(row))
    out.sort(key=lambda r: -r["n"])
    return out


def verbalise_aggregates(
    aggregates: Iterable[dict[str, Any]], limit: int = AGG_LIMIT
) -> list[dict[str, Any]]:
    """Counts as sentences, marked as complete so the model quotes rather than recounts.

    Outbound reuses `RELATION_PHRASE` with the object type from `ALLOWED_EDGES` -- the
    anchor is singular, so the phrase agrees. Inbound needs `RELATION_NOUN`, because the
    subject is a plural count and "32 distinct entities is a subsidiary of AMD" is what
    the phrase would produce.
    """
    facts: list[dict[str, Any]] = []
    for agg in aggregates:
        if len(facts) == limit:
            break
        # A count of one is not an aggregate, it is a fact -- and `RELATION_NOUN` is plural,
        # so it would render as "has 1 subsidiaries". The ranked path facts already carry it.
        if agg["n"] < 2:
            continue
        predicate = RelationType(agg["predicate"])
        shown = list(agg["examples"])
        tail = agg["n"] - len(shown)
        # Semicolons, because entity names contain commas ("Xilinx, Inc.") and a
        # comma-separated list of them cannot be parsed back into items.
        listed = "; ".join(shown) + (f"; and {tail} others" if tail > 0 else "")
        if agg["outbound"]:
            head = (f"{agg['anchor']} {RELATION_PHRASE[predicate]} {agg['n']} distinct "
                    f"{_plural(ALLOWED_EDGES[predicate][1].value)}")
        else:
            head = f"{agg['anchor']} has {agg['n']} {RELATION_NOUN[predicate]}"
        facts.append({
            "text": f"{head} (complete count from the knowledge graph): {listed}.",
            "chunk_ids": list(agg["chunk_ids"]),
            "support": agg["n"],
            "aggregate": True,
        })
    return facts


def vector_path(conn: psycopg.Connection, question: str, k: int) -> list[str]:
    """Phase 2's measured query, unchanged. 1024 is the production column."""
    vector = fireworks.embed([question], dimensions=PRODUCTION_WIDTH, use_cache=True)[0]
    return retrieve.topk(conn, PRODUCTION_WIDTH, vector, k, exact=True)


def merge(graph_ids: list[str], vector_ids: list[str], k: int) -> list[str]:
    """Graph first, vector filling the tail.

    Graph hits are evidence-backed by construction -- every one came off an edge whose
    evidence span was verified against the chunk text -- so they outrank a cosine
    neighbour. Real merging, labelling and citation validation are Phase 4's job.
    """
    out = list(graph_ids)
    for chunk_id in vector_ids:
        if chunk_id not in out:
            out.append(chunk_id)
    return out[:k]


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


#: What the router falls back to when it cannot produce a plan at all. `both` at low
#: confidence is the honest answer to "I do not know": it retrieves from everywhere and
#: costs a little more, which is exactly the trade the low-confidence fallback exists to make.
def _unknown() -> Plan:
    return Plan(
        route=Route.BOTH, confidence="low", entities=[], shape=Shape.NEIGHBOURHOOD, chain=[]
    )


def make_plan(question: str, model: str = ROUTER_MODEL) -> tuple[Plan, str | None]:
    """One constrained call. Returns the plan and a failure reason, never raises.

    `extract.py` learned this the expensive way: a single un-quarantined API exception
    killed a 2,743-chunk run outright, twice. The router is a 57-question loop paced at
    9 RPM, so an uncaught error 40 minutes in costs the whole eval -- and it happened,
    with `_with_backoff` exhausting six retries on 429s (~63s of sleep) before re-raising.

    A router that cannot reach the model is not a fatal condition, it is an unrouted
    question, and `both` is already the defined answer for that. `BudgetExceeded` is a
    plain RuntimeError, not an openai exception, so it still propagates and stops the run.
    """
    try:
        payload = fireworks.chat_json(
            system=SYSTEM, user=question, schema=SCHEMA, model=model,
            timeout=ROUTER_TIMEOUT, attempts=ROUTER_ATTEMPTS,
        )
    except (APIStatusError, APIConnectionError) as exc:
        return _unknown(), f"router_unreachable:{type(exc).__name__}"
    try:
        return Plan.model_validate(payload), None
    except ValidationError:
        return _unknown(), "router_invalid_plan"


def decide(plan: Plan, node_ids: list[str]) -> tuple[Route, list[str], list[str]]:
    """Layer 2: the checks constrained decoding cannot make. Returns (route, chain, reasons).

    Every degradation is named and returned rather than swallowed, because a router that
    quietly rewrites its own decision is a router whose eval numbers mean nothing.
    """
    route = plan.route
    chain = [p.value for p in plan.chain]
    reasons: list[str] = []

    if len(chain) != CHAIN_LEN[plan.shape]:
        # The shape and the chain contradict each other, so the traversal the question
        # implied is unknown. Fall back to an unconstrained neighbourhood walk, and stop
        # trusting the graph path enough to run it alone.
        reasons.append("chain_shape_mismatch")
        chain = []
        if route is Route.GRAPH:
            route = Route.BOTH

    if route in (Route.GRAPH, Route.BOTH) and not node_ids:
        # No anchor node means there is nothing to traverse from.
        reasons.append("no_entity_resolved")
        route = Route.VECTOR

    if plan.confidence == "low" and route in (Route.GRAPH, Route.VECTOR):
        # The spec's low-confidence fallback: run both and merge.
        reasons.append("low_confidence")
        route = Route.BOTH

    return route, chain, reasons


def route(
    question: str,
    session: Session,
    conn: psycopg.Connection,
    index: dict[str, list[dict[str, Any]]],
    k: int = TOP_K,
    model: str = ROUTER_MODEL,
    qid: str | None = None,
    measure_all: bool = False,
    log: bool = True,
) -> dict[str, Any]:
    """Route one question and retrieve. Returns the log row, which is also the result.

    `measure_all` runs both paths whatever the route chose, so `kgrag route`'s eval can put
    vector, graph and routed in the same table. It is off in normal use -- running the path
    the router rejected is exactly the cost routing exists to avoid.
    """
    start = time.perf_counter()
    before = fireworks.METER.usd

    plan, router_error = make_plan(question, model)
    node_ids = [
        i for i in (resolve_entity(name, index, session) for name in plan.entities) if i
    ]
    chosen, chain, reasons = decide(plan, node_ids)
    if router_error:
        reasons.insert(0, router_error)

    want_graph = measure_all or chosen in (Route.GRAPH, Route.BOTH)
    want_vector = measure_all or chosen in (Route.VECTOR, Route.BOTH)
    steps = graph_steps(session, node_ids, chain) if want_graph and node_ids else []
    graph_ids = chunk_ids_of(steps, k)
    # Only for a neighbourhood shape -- an empty chain is the router saying "no fixed
    # traversal fits", which is the same signal it emits for counting and summarising.
    # A 1-hop or 2-hop question already names the traversal it wants, and answering it
    # with corpus-wide counts would bury the fact under statistics.
    aggregates = (
        graph_aggregates(session, node_ids) if want_graph and node_ids and not chain else []
    )
    vector_ids = vector_path(conn, question, k) if want_vector else []

    if chosen is Route.GRAPH:
        chunk_ids = graph_ids
    elif chosen is Route.VECTOR:
        chunk_ids = vector_ids
    elif chosen is Route.BOTH:
        chunk_ids = merge(graph_ids, vector_ids, k)
    else:
        chunk_ids = []

    row = {
        "ts": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "qid": qid,
        "question": question,
        "model": model,
        "router_sha": router_sha(),
        "route": chosen.value,
        "confidence": plan.confidence,
        "degraded_from": plan.route.value if reasons else None,
        "degrade_reason": "+".join(reasons) or None,
        "entities": plan.entities,
        "resolved_ids": node_ids,
        "shape": plan.shape.value,
        "chain": chain,
        "chunk_ids": chunk_ids,
        "graph_ids": graph_ids,
        "vector_ids": vector_ids,
        # What the traversal actually said, not just where it pointed. Phase 4 cites these
        # and Phase 5 reads this log; without them a routing row records that the graph was
        # consulted and loses its answer, which cannot be reconstructed after the fact.
        "graph_facts": verbalise(steps),
        # Exact counts over the whole neighbourhood, for the questions a ranked top-k walk
        # cannot answer. Empty unless the router chose a neighbourhood shape.
        "graph_aggregates": verbalise_aggregates(aggregates),
        # Whether both paths were run regardless of the route. Without this the log mixes
        # two incomparable kinds of row: eval rows (both paths measured) and answer-time
        # rows (only the chosen path run, so the other is empty BY DESIGN, not by failure).
        # Reading an answer-time row as if it were an eval row shows the graph scoring 0 on
        # questions it answers perfectly. Phase 5 reads this log; it needs to tell them apart.
        "measure_all": measure_all,
        "n_graph": len(graph_ids),
        "n_vector": len(vector_ids),
        "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        "usd": round(fireworks.METER.usd - before, 6),
    }
    if log:
        jsonl.append(ROUTING_LOG, [row])
    return row


# ---------------------------------------------------------------------------
# `kgrag route`
# ---------------------------------------------------------------------------


def expected_route(question: dict[str, Any]) -> set[str]:
    """What the eval set itself says the route should be. Derived, never hand-written.

    `questions.py` derives its gold chunk sets from filing structure rather than judgement;
    the same discipline applies to the router's answer key. Three of the four slices have a
    single right answer. The fourth genuinely does not, and says so rather than inventing
    one: a mined 1-hop question is templated off a graph edge (so the graph is right) AND
    quotes the canonical entity name verbatim (so vector search has an exact lexical match
    and is also right). Only refusing is wrong.
    """
    if question["hops"] == 0:
        return {"refuse"}
    if question["hops"] >= 2:
        return {"graph", "both"}
    if question["source"] == "hand":
        return {"vector", "both"}
    return {"graph", "vector", "both"}


def _recall(got: list[str], gold: list[str], k: int = 10) -> float:
    return len(set(got[:k]) & set(gold)) / len(gold)


def run(
    question: str | None = None,
    model: str = ROUTER_MODEL,
    k: int = TOP_K,
    fresh: bool = False,
) -> None:
    index = entity_index()
    if not index:
        raise SystemExit("data/entities.jsonl is empty — run `kgrag resolve` first.")

    with load.driver() as db, db.session() as session, connect() as conn:
        if question:
            row = route(question, session, conn, index, k=k, model=model)
            _print_one(row)
            return
        _eval(session, conn, index, model, k, fresh)


def _print_one(row: dict[str, Any]) -> None:
    print(f"\nquestion  {row['question']}")
    print(f"route     {row['route']}  (confidence {row['confidence']})")
    if row["degrade_reason"]:
        print(f"degraded  {row['degraded_from']} -> {row['route']}  [{row['degrade_reason']}]")
    print(f"entities  {row['entities']} -> {row['resolved_ids']}")
    print(f"traversal {row['shape']}  {row['chain']}")
    print(f"chunks    {row['chunk_ids']}")
    print(f"           {row['n_graph']} from graph, {row['n_vector']} from vector")
    print(f"latency   {row['latency_ms']} ms   spend ${row['usd']:.5f}")


def _prior(model: str) -> dict[str, dict[str, Any]]:
    """qid -> the last logged decision for this router model, so the eval can resume.

    Every other stage in this pipeline resumes: `extract` skips chunk_ids already in
    extractions.jsonl, `embed` resumes from `WHERE emb_N IS NULL`. This one restarted from
    zero, which on an 8 GB machine running Neo4j and Postgres together means it may never
    finish -- the process gets killed for memory somewhere past question 10, having done
    40 minutes of paced API calls, and the next attempt repeats them.

    The routing log already holds everything a row needs, so replay is exact rather than
    approximate. Later rows win, so a re-measured question supersedes its own history.
    Rows whose router call never returned are NOT resumed: they record a fallback, not a
    decision, and freezing a transient stall into the numbers is exactly the kind of eval
    that cannot fail. `--fresh` ignores the log entirely, which is what to use after
    changing ranking or traversal code -- otherwise stale rows would survive into the
    numbers quietly.
    """
    prior: dict[str, dict[str, Any]] = {}
    sha = router_sha()
    for row in jsonl.read(ROUTING_LOG):
        if not row.get("qid") or row.get("model") != model:
            continue
        # A decision made by a prompt or schema that no longer exists is not a decision
        # about the current router. Rows written before this field existed carry no sha
        # and are not resumable either -- correct, if briefly wasteful.
        if row.get("router_sha") != sha:
            prior.pop(row["qid"], None)
            continue
        # A router that timed out did not decide anything -- it fell back to `both` because
        # the call never returned, and chat_json only caches successes, so retrying is both
        # possible and free of a stale answer. Resuming over one would freeze a transient
        # Fireworks stall into the published numbers. `extract.py` draws the same line by
        # quarantining failures into failures.jsonl instead of extractions.jsonl.
        if "router_unreachable" in (row.get("degrade_reason") or ""):
            prior.pop(row["qid"], None)
            continue
        prior[row["qid"]] = row
    return prior


def _eval(
    session: Session,
    conn: psycopg.Connection,
    index: dict[str, list[dict[str, Any]]],
    model: str,
    k: int,
    fresh: bool = False,
) -> None:
    questions = list(jsonl.read(QUESTIONS))
    if not questions:
        raise SystemExit(f"{QUESTIONS} is empty — run `kgrag mine-questions` first.")

    print("=" * 74)
    print(f"routing eval — {len(questions)} questions, router = {model.split('/')[-1]}")
    print("=" * 74)
    print(
        "Expected routes are derived from the eval set, not hand-assigned: hops==0 must\n"
        "refuse, hops>=2 must reach the graph, and the hand-written 1-hop paraphrases must\n"
        "reach vector. Mined 1-hop questions accept either path and are scored only on not\n"
        "refusing -- they are templated off a graph edge AND quote the canonical name\n"
        "verbatim, so both paths are legitimately right.\n"
    )

    before = fireworks.METER.usd
    prior = {} if fresh else _prior(model)
    rows: list[dict[str, Any]] = []
    for question in questions:
        done = prior.get(question["qid"])
        if done:
            rows.append(done)
            continue
        rows.append(
            route(
                question["question"], session, conn, index,
                k=k, model=model, qid=question["qid"], measure_all=True,
            )
        )
    spend = fireworks.METER.usd - before
    resumed = sum(1 for q in questions if q["qid"] in prior)
    if resumed:
        print(f"resumed {resumed}/{len(questions)} decisions from {ROUTING_LOG.name}\n")

    print(f"{'slice':16} {'n':>4} {'correct':>8} {'accuracy':>9}")
    slices: list[tuple[str, list[int]]] = [
        ("out-of-scope", [i for i, q in enumerate(questions) if q["hops"] == 0]),
        ("1-hop mined", [i for i, q in enumerate(questions) if q["hops"] == 1 and q["source"] == "mined"]),
        ("1-hop hand", [i for i, q in enumerate(questions) if q["hops"] == 1 and q["source"] == "hand"]),
        ("multi-hop", [i for i, q in enumerate(questions) if q["hops"] >= 2]),
        ("all", list(range(len(questions)))),
    ]
    for label, idx in slices:
        if not idx:
            continue
        hits = sum(1 for i in idx if rows[i]["route"] in expected_route(questions[i]))
        print(f"  {label:14} {len(idx):>4} {hits:>8} {hits / len(idx):>9.3f}")

    chosen: dict[str, int] = {}
    for row in rows:
        chosen[row["route"]] = chosen.get(row["route"], 0) + 1
    print(f"\nroutes chosen   {chosen}")

    degrades: dict[str, int] = {}
    for row in rows:
        if row["degrade_reason"]:
            degrades[row["degrade_reason"]] = degrades.get(row["degrade_reason"], 0) + 1
    print(f"degradations    {degrades or 'none'}")

    unresolved = sum(1 for row in rows if row["entities"] and not row["resolved_ids"])
    print(f"entity lookup   {unresolved} questions named entities that resolved to nothing")

    # R@10 by hop for three policies. The vector column reproduces Phase 2's measurement,
    # which makes it a built-in check on the harness before any graph claim is read.
    print("\n" + "-" * 74)
    print(f"R@{k} by hop count — the Phase 5 curve, previewed")
    print("-" * 74)
    print(f"{'slice':14} {'n':>4} {'vector':>9} {'graph':>9} {'routed':>9}")
    answerable = [i for i, q in enumerate(questions) if q["gold_chunk_ids"]]
    for hops in (1, 2, 3):
        idx = [i for i in answerable if questions[i]["hops"] == hops]
        if not idx:
            continue
        cols = [
            statistics.mean(_recall(rows[i][key], questions[i]["gold_chunk_ids"], k) for i in idx)
            for key in ("vector_ids", "graph_ids", "chunk_ids")
        ]
        print(f"  {str(hops) + '-hop':12} {len(idx):>4} " + " ".join(f"{c:>9.3f}" for c in cols))
    cols = [
        statistics.mean(
            _recall(rows[i][key], questions[i]["gold_chunk_ids"], k) for i in answerable
        )
        for key in ("vector_ids", "graph_ids", "chunk_ids")
    ]
    print(f"  {'all':12} {len(answerable):>4} " + " ".join(f"{c:>9.3f}" for c in cols))
    # The graph column is NOT a pure measurement of the graph. Entities come out of the
    # same router call that picks the route, so a router that times out or wrongly refuses
    # names no entity, the traversal has no anchor, and the graph scores 0 for a reason
    # that has nothing to do with the graph. Reporting that as a graph result would
    # understate it and invert the comparison this table exists to make.
    unanchored = [i for i in answerable if not rows[i]["resolved_ids"]]
    if unanchored:
        anchored = [i for i in answerable if i not in set(unanchored)]
        floor = statistics.mean(
            _recall(rows[i]["graph_ids"], questions[i]["gold_chunk_ids"], k) for i in anchored
        )
        print(
            f"\n  {len(unanchored)} of {len(answerable)} answerable questions gave the traversal no\n"
            f"  anchor ({', '.join(questions[i]['qid'] for i in unanchored)}) — the router named no\n"
            f"  entity, so the graph column scores them 0 by construction. That is a router\n"
            f"  failure being charged to the graph. Over the {len(anchored)} anchored questions the\n"
            f"  graph path scores {floor:.3f}, which is the graph's own number; the column above\n"
            f"  is the end-to-end number a user would actually get."
        )

    print(
        "\n  'routed' is what the router actually returned; 'graph' and 'vector' are both\n"
        "  paths run for every question regardless of the route, which is why the vector\n"
        "  column reproduces Phase 2's .586/.288/.328. If it does not, the harness is wrong\n"
        "  and nothing else in this table should be read."
    )

    appended = len(questions) - resumed
    print(f"\nrouter spend    ${spend:.5f} over {appended} routed, {resumed} resumed")
    print(f"routing log     {ROUTING_LOG} (+{appended} rows)")

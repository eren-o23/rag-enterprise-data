"""`kgrag answer` — one grounded answer from both stores, with citations that must resolve.

Phases 1-3 all trade in `chunk_id`s. This is the layer that turns them into prose, and the
whole design question is how a claim stays attached to its evidence.

**Citations are chunk ids.** Not footnote numbers, not passage indices -- the same
identifier the graph edges carry and the pgvector rows are keyed on. That makes validating
a citation set membership and nothing else, and it means a graph-derived fact and a
retrieved passage cite in the same currency, which is what lets one answer mix them.

Two enforcement arms are built, because "validated citations" is a claim and this project
measures claims rather than asserting them:

* **free** (default) — citations are strings. Invalid ones are rejected and the call is
  regenerated, which is what the spec asks for and is the only arm that can report an
  invented-citation rate.
* **constrained** (`--constrained`) — the retrieved ids become an `enum` in the JSON
  schema, so an invented citation is unreachable rather than merely caught. This is
  `ontology.py`'s guarantee applied to citations.

The gap between the two arms is the measurement. Neither is the "right" answer on its own.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import statistics
import time
from typing import Any

import psycopg
from neo4j import Session
from openai import APIConnectionError, APIStatusError
from pydantic import BaseModel, Field, ValidationError

from . import fireworks, jsonl, load
from . import route as route_mod
from .config import DATA
from .embed import connect
from .extract import _strict
from .questions import QUESTIONS

ANSWER_LOG = DATA / "answer_log.jsonl"

SYNTH_MODEL = "accounts/fireworks/models/gpt-oss-120b"

#: Synthesis reads a whole context window and writes several sentences, so it is slower
#: than routing but nowhere near extraction's streamed-JSON worst case. Measured healthy
#: latency is 4-12s. `route.py` documents why inheriting the 90s module default is how an
#: eval turns into hours.
SYNTH_TIMEOUT = 45.0
SYNTH_ATTEMPTS = 3

#: How many repair rounds a bad citation set gets before the answer is abandoned.
MAX_REPAIRS = 2

#: Passages shown in full. The router already caps retrieval at k=10; this is a second
#: guard so a future k change cannot quietly triple the prompt.
PASSAGE_LIMIT = 10


class Claim(BaseModel):
    text: str = Field(description="One factual sentence answering part of the question.")
    citations: list[str] = Field(
        description="Chunk ids from the context that state this claim, copied verbatim "
        "from inside the square brackets. Every claim needs at least one."
    )


class Answer(BaseModel):
    answerable: bool = Field(
        description="True if the context supports at least part of an answer. False only "
        "if it supports none of it -- including when the context is about the right "
        "subject but does not contain the fact asked for."
    )
    claims: list[Claim] = Field(
        description="Every part of the answer the context supports. Empty when answerable "
        "is false."
    )
    refusal_reason: str = Field(
        description="What the context does not support: the whole question when answerable "
        "is false, or the unanswered part of a compound question when it is true. Empty "
        "only when the context answers the question fully."
    )


SYSTEM = """You answer questions about SEC filings using ONLY the context provided.

The context has up to three labelled blocks:
- GRAPH COUNTS: exact totals computed over the whole knowledge graph. These are complete.
  Quote the number as given.
- GRAPH FACTS: statements derived from a knowledge graph built from these filings. Each
  was verified against the filing text that asserts it.
- PASSAGES: verbatim filing text retrieved for this question.

RULES:
1. Every claim must be supported by the context. Do not use outside knowledge, even when
   you are confident it is correct.
2. Every claim must cite at least one chunk id. Chunk ids appear in square brackets at the
   start of each fact and each passage. Copy them exactly.
3. Cite only ids that appear in the context. Never invent, abbreviate or reformat one.
4. If the context does not state the answer, set answerable to false, leave claims empty,
   and say what is missing. Context about the right company that does not contain the fact
   asked for is NOT an answer -- refuse it.
4b. A question with several parts may be partly answerable. Answer the parts the context
   supports as claims, and name the unsupported parts in refusal_reason. Discarding a
   supported fact because a different part of the question is unsupported loses a correct
   answer; asserting the unsupported part loses the reader's trust.
5. Prefer graph facts for connections between entities and passages for language, policy
   and detail. When both support a claim, cite both.
5b. A GRAPH FACT containing → is one chain, read left to right: each step begins where the
   previous step ended. A question that asks about something reached THROUGH another entity
   is asking about the end of the chain, not its start. "Who competes with the customers
   that X supplies?" is answered by the last step of "X supplies Y → Y competes with Z",
   which is Z. Answering with X's own competitors answers a different question.
6. Answer in as few claims as the question needs. One fact per claim.
7. For "how many" questions, use the number in GRAPH COUNTS exactly as written. Do not
   recount from the listed names -- the list is a sample and the count is complete.
8. Use a GRAPH COUNT only if it counts exactly what was asked. A count of a different
   relation, or of a coarser or finer thing than the question names, is not an answer:
   "distinct Locations" includes cities, states and regions and is NOT a count of
   countries. When no GRAPH COUNT matches what is asked, say the corpus does not support
   that count. Substituting the nearest available number is the worst possible answer --
   it is wrong, confident, and correctly cited."""


def arm_of(constrain: bool, graph: bool) -> str:
    """The three systems this module can be: two enforcement arms and the baseline.

    It is one string because it is one axis of the log: `_prior` refuses to resume across
    it, and an arm that resumed another arm's rows would report a comparison it never ran.
    Phase 4 already hit that with two arms; the baseline makes three.
    """
    return "vector" if not graph else ("constrained" if constrain else "free")


def _schema(valid_ids: list[str], constrain: bool) -> dict[str, Any]:
    """The answer schema, optionally with citations closed to the retrieved ids.

    In the constrained arm the citation field stops being a string and becomes an enum over
    exactly what was retrieved, so a fabricated id is not rejected after the fact -- it is
    never generated. `ontology.py` makes an off-ontology predicate unreachable the same way.

    The schema is part of `chat_json`'s cache key, so the two arms can never read each
    other's cached answers even for an identical question and context.
    """
    schema = _strict(Answer.model_json_schema())
    if constrain:
        defs = schema["$defs"]["Claim"]["properties"]["citations"]
        defs["items"] = {"type": "string", "enum": sorted(valid_ids)}
    return schema


def synth_sha(constrain: bool) -> str:
    """Fingerprint of everything that decides how an answer is produced.

    Same job as `route.router_sha`, and for the same reason: the eval resumes from its own
    log, and a resumed row written by a prompt or schema that no longer exists is not a
    measurement of the current system. The arm is in here too -- the two arms must never
    resume each other's rows.
    """
    material = f"{SYSTEM}|{json.dumps(_schema([], constrain), sort_keys=True)}|{constrain}"
    return hashlib.sha256(material.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------


def passages(conn: psycopg.Connection, chunk_ids: list[str]) -> list[dict[str, Any]]:
    """Fetch passage text for ranked chunk ids, preserving the ranking.

    One round trip with `= ANY`, then re-sorted in Python: SQL returns rows in whatever
    order it likes, and retrieval rank is the one thing the ordering has to carry.
    """
    if not chunk_ids:
        return []
    rows = conn.execute(
        "SELECT chunk_id, company, form, section_path, filing_date, text "
        "FROM chunks WHERE chunk_id = ANY(%s)",
        (chunk_ids,),
    ).fetchall()
    by_id = {
        r[0]: {"chunk_id": r[0], "company": r[1], "form": r[2],
               "section_path": r[3], "filing_date": r[4], "text": r[5]}
        for r in rows
    }
    return [by_id[i] for i in chunk_ids if i in by_id]


#: Chunk ids shown per graph fact. An edge corroborated by nine chunks does not need all
#: nine in the prompt to be citable.
CITES_PER_FACT = 3


def build_context(
    facts: list[dict[str, Any]],
    texts: list[dict[str, Any]],
    aggregates: list[dict[str, Any]] | None = None,
) -> tuple[str, set[str]]:
    """The two labelled blocks, and the ids a citation is allowed to name.

    The spec asks for graph-derived facts to be kept explicitly separate from retrieved
    passages, and the labels are load-bearing rather than decorative: a graph fact is a
    claim the pipeline already verified against filing text, a passage is raw evidence the
    model still has to read. Telling the model which is which is what lets rule 5 exist.

    **The citable set is returned from here, not computed by the caller.** It has to be
    exactly what this function printed, and deriving it anywhere else lets the two drift.
    They did: the caller used the route's top-10 `graph_ids` while this block rendered 20
    facts carrying up to three ids each, so the model cited ids that were plainly in front
    of it and the validator called them invented — six answers abandoned as
    `citation_unrecoverable` for citing their own context correctly.
    """
    aggregates = aggregates or []
    blocks: list[str] = []
    citable: set[str] = set()
    if aggregates:
        lines = ["GRAPH COUNTS (complete, computed over the whole graph — do not recount)"]
        for fact in aggregates:
            shown = fact["chunk_ids"][:CITES_PER_FACT]
            citable.update(shown)
            lines.append(" ".join(f"[{c}]" for c in shown) + f" {fact['text']}")
        blocks.append("\n".join(lines))
    if facts:
        lines = ["GRAPH FACTS (derived from the knowledge graph)"]
        for fact in facts:
            shown = fact["chunk_ids"][:CITES_PER_FACT]
            citable.update(shown)
            lines.append(" ".join(f"[{c}]" for c in shown) + f" {fact['text']}")
        blocks.append("\n".join(lines))
    if texts:
        lines = ["PASSAGES (retrieved filing text)"]
        for passage in texts[:PASSAGE_LIMIT]:
            citable.add(passage["chunk_id"])
            lines.append(
                f"\n[{passage['chunk_id']}] {passage['company']} · {passage['form']} · "
                f"{passage['section_path']} · {passage['filing_date']}\n{passage['text']}"
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks), citable


# ---------------------------------------------------------------------------
# Synthesis, and the citation contract
# ---------------------------------------------------------------------------


#: Chunk ids appear in the context wrapped in square brackets, and gpt-oss-120b copies the
#: brackets along with the id often enough to matter -- "[8b64d880da42f277]" for an id that
#: really was retrieved. That is a delimiter it failed to strip, not a source it made up,
#: and three repair rounds do not talk it out of the habit. Counting it as invention would
#: be wrong twice: it overstates the invented rate, and it makes the free-vs-constrained
#: comparison meaningless, because the constrained arm cannot emit a bracket at all and
#: would "win" on formatting rather than on grounding. Stripped narrowly -- surrounding
#: brackets and whitespace only -- so a genuinely fabricated id still fails.
def _clean(citation: str) -> str:
    return citation.strip().strip("[]").strip()


def normalise_citations(ans: Answer) -> int:
    """Strip delimiters off every citation in place. Returns how many needed it."""
    fixed = 0
    for claim in ans.claims:
        cleaned = [_clean(c) for c in claim.citations]
        fixed += sum(1 for before, after in zip(claim.citations, cleaned) if before != after)
        claim.citations = cleaned
    return fixed


def invented(ans: Answer, valid_ids: set[str]) -> list[str]:
    """Cited ids that were never retrieved. The thing the spec says to reject on."""
    return sorted({c for claim in ans.claims for c in claim.citations} - valid_ids)


def _user(question: str, context: str, rejected: list[str] | None = None) -> str:
    """The user message, and on a repair the reason the last one was thrown away.

    Naming the rejected ids is not politeness, it is a correctness requirement.
    `chat_json` keys its cache on (model, prompt version, system, user, schema), so a
    repair that resends a byte-identical prompt is served the same invalid answer out of
    `cache/` forever, at $0.00, looking exactly like a model that refuses to correct
    itself. The rejected ids change the key, which is what makes the retry a retry.
    """
    parts = [f"QUESTION: {question}", "", "CONTEXT:", context]
    if rejected:
        parts += [
            "",
            "Your previous answer cited chunk ids that are not in the context above: "
            + ", ".join(rejected)
            + ". Every citation must be copied verbatim from inside the square brackets "
            "in the context. Answer again, or set answerable to false if the context does "
            "not support the answer.",
        ]
    return "\n".join(parts)


def synthesise(
    question: str,
    context: str,
    valid_ids: set[str],
    model: str = SYNTH_MODEL,
    constrain: bool = False,
    use_cache: bool = True,
) -> tuple[Answer, int, list[str], int]:
    """Generate, validate, regenerate on a miss. Returns (answer, attempts, invented, reformatted).

    Never raises. `route.make_plan` learned this from `extract.py`: an uncaught API error
    partway through a paced sweep costs the whole sweep. An unreachable model here is an
    unanswered question, not a dead run.
    """
    schema = _schema(sorted(valid_ids), constrain)
    rejected: list[str] = []
    attempts = 0
    reformatted = 0

    for _ in range(MAX_REPAIRS + 1):
        attempts += 1
        try:
            payload = fireworks.chat_json(
                system=SYSTEM, user=_user(question, context, rejected), schema=schema,
                model=model, timeout=SYNTH_TIMEOUT, attempts=SYNTH_ATTEMPTS,
                use_cache=use_cache,
            )
        except (APIStatusError, APIConnectionError) as exc:
            return _refusal(f"synthesiser_unreachable:{type(exc).__name__}"), attempts, rejected, reformatted
        try:
            ans = Answer.model_validate(payload)
        except ValidationError:
            return _refusal("synthesiser_invalid_answer"), attempts, rejected, reformatted

        reformatted += normalise_citations(ans)
        bad = invented(ans, valid_ids)
        if not bad:
            # A claim with no citation at all is ungrounded too -- it passes the "no
            # invented id" check vacuously, which is exactly the hole a citation
            # requirement has to not have.
            if any(not claim.citations for claim in ans.claims):
                rejected = ["(a claim carried no citation)"]
                continue
            return ans, attempts, [], reformatted
        rejected = bad

    return _refusal("citation_unrecoverable"), attempts, rejected, reformatted


def _refusal(reason: str) -> Answer:
    return Answer(answerable=False, claims=[], refusal_reason=reason)


# ---------------------------------------------------------------------------
# One question, end to end
# ---------------------------------------------------------------------------


def answer(
    question: str,
    session: Session,
    conn: psycopg.Connection,
    index: dict[str, list[dict[str, Any]]],
    model: str = SYNTH_MODEL,
    router_model: str = route_mod.ROUTER_MODEL,
    constrain: bool = False,
    use_cache: bool = True,
    qid: str | None = None,
    log: bool = True,
    graph: bool = True,
) -> dict[str, Any]:
    """Route, assemble, synthesise, validate. Returns the log row.

    `graph=False` is the Phase 5 baseline: plain vector RAG, which is not this system with
    the graph switched off but a different system entirely. It makes no router call at all —
    embedding the question and taking the top k is the whole retrieval story — so its
    latency and its cost are honestly those of a vector-only stack rather than this one's
    minus a subtraction.

    Everything downstream is held identical on purpose: same `SYSTEM`, same schema, same
    citation contract, same enum constraint. Retrieval is the only variable, which is the
    only way the delta means anything. The prompt still describes GRAPH COUNTS and GRAPH
    FACTS blocks the baseline never receives — deliberately, because changing the prompt
    too would leave two variables moving and no way to say which one the delta belongs to.
    A counting question the baseline cannot answer is then a finding, not a handicap.
    """
    start = time.perf_counter()
    before = fireworks.METER.usd

    if graph:
        row = route_mod.route(question, session, conn, index, model=router_model, qid=qid,
                              log=log, use_cache=use_cache)
    else:
        vector_ids = route_mod.vector_path(conn, question, route_mod.TOP_K)
        # Not written to the routing log: no routing decision was made, and a row there
        # claiming otherwise would corrupt the one artifact Phase 3's numbers come from.
        row = {
            "route": "vector-only", "chunk_ids": vector_ids, "graph_ids": [],
            "vector_ids": vector_ids, "graph_facts": [], "graph_aggregates": [],
        }
    facts = row["graph_facts"]
    fallback = None

    # A graph route whose traversal found nothing is not an answer, it is an empty context.
    # Phase 3 measures the router's decision and correctly does not run the path it
    # rejected; at answer time the passages may still hold the fact, and one cached
    # embedding call is cheaper than a refusal. Logged, counted and reported -- a silent
    # fallback would make the routing numbers describe something other than production.
    if row["route"] == "graph" and not row["graph_ids"]:
        fallback = "graph_empty_fell_back_to_vector"
        row["vector_ids"] = route_mod.vector_path(conn, question, route_mod.TOP_K)
        row["chunk_ids"] = row["vector_ids"]

    texts = passages(conn, row["chunk_ids"])
    context, valid_ids = build_context(facts, texts, row.get("graph_aggregates"))

    if row["route"] == "refuse" or not context:
        # The router refused, or retrieval returned nothing. Either way there is nothing to
        # ground an answer in, so no model call is made and the question costs $0.00.
        ans, attempts, bad, reformatted = (
            _refusal("no_context" if row["route"] != "refuse" else "router_refused"), 0, [], 0
        )
    else:
        ans, attempts, bad, reformatted = synthesise(
            question, context, valid_ids, model, constrain, use_cache
        )

    cited = sorted({c for claim in ans.claims for c in claim.citations})
    out = {
        "ts": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "qid": qid,
        "question": question,
        "model": model,
        "arm": arm_of(constrain, graph),
        "synth_sha": synth_sha(constrain),
        "route": row["route"],
        "fallback": fallback,
        "n_facts": len(facts),
        "n_aggregates": len(row.get("graph_aggregates") or []),
        "n_passages": len(texts),
        "context_chars": len(context),
        "answerable": ans.answerable,
        "refusal_reason": ans.refusal_reason,
        "claims": [c.model_dump() for c in ans.claims],
        "cited": cited,
        "invented": bad,
        "reformatted": reformatted,
        "attempts": attempts,
        "retrieved_ids": sorted(valid_ids),
        "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        "usd": round(fireworks.METER.usd - before, 6),
    }
    if log:
        jsonl.append(ANSWER_LOG, [out])
    return out


# ---------------------------------------------------------------------------
# `kgrag answer`
# ---------------------------------------------------------------------------


def run(
    question: str | None = None,
    model: str = SYNTH_MODEL,
    constrained: bool = False,
    fresh: bool = False,
    use_cache: bool = True,
    baseline: bool = False,
) -> None:
    # The baseline runs constrained, because that is what the hybrid arm runs. Comparing a
    # constrained system against a free one would put the enforcement question and the
    # retrieval question in the same column, and Phase 4 already answered the first.
    constrained = constrained or baseline
    index = route_mod.entity_index()
    if not index:
        raise SystemExit("data/entities.jsonl is empty — run `kgrag resolve` first.")

    with load.driver() as db, db.session() as session, connect() as conn:
        if question:
            _print_one(answer(
                question, session, conn, index,
                model=model, constrain=constrained, use_cache=use_cache, graph=not baseline,
            ))
            return
        _eval(session, conn, index, model, constrained, fresh, use_cache, not baseline)


def _print_one(row: dict[str, Any]) -> None:
    print(f"\nquestion   {row['question']}")
    print(f"route      {row['route']}" + (f"  [{row['fallback']}]" if row["fallback"] else ""))
    print(f"context    {row['n_aggregates']} graph counts, {row['n_facts']} graph facts, "
          f"{row['n_passages']} passages, "
          f"{row['context_chars']:,} chars")
    print()
    if not row["answerable"]:
        print(f"REFUSED    {row['refusal_reason']}")
    for claim in row["claims"]:
        print(f"  • {claim['text']}")
        print(f"    {' '.join('[' + c + ']' for c in claim['citations'])}")
    if row["answerable"] and row["refusal_reason"]:
        print(f"\nnot covered  {row['refusal_reason']}")
    if row["invented"]:
        print(f"\ninvented   {row['invented']}")
    print(f"\nattempts   {row['attempts']}   arm {row['arm']}")
    print(f"latency    {row['latency_ms']} ms   spend ${row['usd']:.5f}")


def _prior(model: str, constrain: bool, graph: bool = True) -> dict[str, dict[str, Any]]:
    """qid -> the last logged answer for this model and arm. Same rules as `route._prior`.

    Rows written by a different prompt, schema or arm are not resumable, and neither is a
    row whose synthesiser call never returned -- that records a fallback, not an answer,
    and freezing a transient stall into the published numbers is how an eval stops being
    able to fail.
    """
    prior: dict[str, dict[str, Any]] = {}
    sha = synth_sha(constrain)
    arm = arm_of(constrain, graph)
    for row in jsonl.read(ANSWER_LOG):
        if not row.get("qid") or row.get("model") != model:
            continue
        # Both arms append to one log, so the other arm's rows are simply not evidence
        # about this one -- skipped, never popped. Popping them is a resume that silently
        # never resumes: running free then constrained left each arm's rows evicted by the
        # other's, reporting "0 resumed" for 57 questions that were all on disk.
        if row.get("arm") != arm:
            continue
        # A row written by a prompt or schema that no longer exists IS about this arm, and
        # is superseded rather than irrelevant -- so this one pops.
        if row.get("synth_sha") != sha:
            prior.pop(row["qid"], None)
            continue
        if "unreachable" in (row.get("refusal_reason") or ""):
            prior.pop(row["qid"], None)
            continue
        prior[row["qid"]] = row
    return prior


def _eval(
    session: Session,
    conn: psycopg.Connection,
    index: dict[str, list[dict[str, Any]]],
    model: str,
    constrain: bool,
    fresh: bool,
    use_cache: bool = True,
    graph: bool = True,
) -> None:
    questions = list(jsonl.read(QUESTIONS))
    if not questions:
        raise SystemExit(f"{QUESTIONS} is empty — run `kgrag mine-questions` first.")

    arm = arm_of(constrain, graph)
    label = "vector-only baseline, no graph, no router" if not graph else f"{arm} citations"
    print("=" * 74)
    print(f"answer eval — {len(questions)} questions, {label}, {model.split('/')[-1]}")
    print("=" * 74)
    print(
        "Citations are chunk ids, so validating one is set membership against what was\n"
        "actually retrieved. The free arm lets the model write any string and rejects the\n"
        "bad ones; the constrained arm puts the retrieved ids in the schema as an enum, so\n"
        "an invented id cannot be generated. The gap between the two is the measurement.\n"
    )

    before = fireworks.METER.usd
    prior = {} if fresh else _prior(model, constrain, graph)
    rows: list[dict[str, Any]] = []
    for q in questions:
        done = prior.get(q["qid"])
        rows.append(done if done else answer(
            q["question"], session, conn, index,
            model=model, constrain=constrain, use_cache=use_cache, qid=q["qid"], graph=graph,
        ))
    spend = fireworks.METER.usd - before
    resumed = sum(1 for q in questions if q["qid"] in prior)
    if resumed:
        print(f"resumed {resumed}/{len(questions)} answers from {ANSWER_LOG.name}\n")

    _report_citations(rows)
    _report_refusals(questions, rows)
    _report_grounding(questions, rows)
    _report_aggregation(questions, rows)
    _report_cost(rows)

    appended = len(questions) - resumed
    print(f"\nspend           ${spend:.5f} over {appended} answered, {resumed} resumed")
    print(f"answer log      {ANSWER_LOG} (+{appended} rows)")


def _report_citations(rows: list[dict[str, Any]]) -> None:
    answered = [r for r in rows if r["answerable"]]
    total = sum(len(r["cited"]) for r in answered)
    claims = sum(len(r["claims"]) for r in answered)
    uncited = sum(1 for r in answered for c in r["claims"] if not c["citations"])
    repaired = [r for r in rows if r["attempts"] > 1 and r["answerable"]]
    lost = [r for r in rows if r["refusal_reason"] == "citation_unrecoverable"]

    print("-" * 74)
    print("citation integrity")
    print("-" * 74)
    print(f"  answers produced            {len(answered)}/{len(rows)}")
    print(f"  claims                      {claims}")
    print(f"  citations                   {total}")
    print(f"  claims with no citation     {uncited}")
    print(f"  citations needing a strip   {sum(r['reformatted'] for r in rows)}"
          "   (id copied with its [brackets] — see _clean)")
    print(f"  answers needing a repair    {len(repaired)}")
    print(f"  abandoned after repairs     {len(lost)}"
          + (f"  {[r['qid'] for r in lost]}" if lost else ""))
    print(f"  invented ids still standing {sum(len(r['invented']) for r in answered)}"
          "   (must be 0 — every published citation resolves)")


def _report_refusals(questions: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    oos = [i for i, q in enumerate(questions) if q["hops"] == 0]
    ans = [i for i, q in enumerate(questions) if q["gold_chunk_ids"]]
    refused_oos = [i for i in oos if not rows[i]["answerable"]]
    refused_ans = [i for i in ans if not rows[i]["answerable"]]
    fell_back = [r for r in rows if r["fallback"]]

    print("\n" + "-" * 74)
    print("refusal behaviour")
    print("-" * 74)
    print(f"  out-of-scope refused        {len(refused_oos)}/{len(oos)}")
    print(f"  answerable refused          {len(refused_ans)}/{len(ans)}"
          + (f"  {[questions[i]['qid'] for i in refused_ans]}" if refused_ans else ""))
    print(f"  graph empty -> vector       {len(fell_back)}")
    reasons: dict[str, int] = {}
    for r in rows:
        if not r["answerable"]:
            reasons[r["refusal_reason"][:40]] = reasons.get(r["refusal_reason"][:40], 0) + 1
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"      {n:>3}  {reason}")


def _report_grounding(questions: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    """What share of cited chunks are gold. A FLOOR — gold sets are not exhaustive."""
    print("\n" + "-" * 74)
    print("grounding — share of cited chunks that are in the question's gold set")
    print("-" * 74)
    print(f"  {'slice':14} {'n':>4} {'precision':>10}")
    slices: list[tuple[str, list[int]]] = [
        (f"{h}-hop", [i for i, q in enumerate(questions)
                      if q["hops"] == h and not q.get("category")])
        for h in (1, 2, 3)
    ]
    slices.append(("aggregation", [i for i, q in enumerate(questions)
                                   if q.get("category") == "aggregation"]))
    for label, cand in slices:
        idx = [i for i in cand if questions[i]["gold_chunk_ids"] and rows[i]["cited"]]
        if not idx:
            continue
        scores = [
            len(set(rows[i]["cited"]) & set(questions[i]["gold_chunk_ids"])) / len(rows[i]["cited"])
            for i in idx
        ]
        print(f"  {label:12} {len(idx):>4} {statistics.mean(scores):>10.3f}")
    print(
        "\n  Gold sets are the chunks whose text justified a graph edge — a lower bound, not\n"
        "  an exhaustive answer key, so a cited chunk outside the set is not necessarily\n"
        "  wrong. Read this as a floor. Answer correctness needs a judge on every slice but\n"
        "  aggregation, which is scored exactly below, and is Phase 5's job."
    )


#: Number words the model actually uses instead of digits. gpt-oss-120b answers "audits
#: nine companies" where the graph says 9, and a digits-only scorer marks a correct answer
#: wrong -- which is a broken instrument reporting a broken system.
_WORDS = {w: i for i, w in enumerate((
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen", "twenty",
))}


def stated_numbers(text: str) -> set[int]:
    """Every integer a sentence states, in digits or in words."""
    found = {int(n.replace(",", "")) for n in re.findall(r"\b\d[\d,]*\b", text)}
    found |= {_WORDS[w] for w in re.findall(r"[a-z]+", text.lower()) if w in _WORDS}
    return found


def _report_aggregation(questions: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    """Exact correctness on the one slice that needs no judge.

    Every other stratum needs a grader to say whether an answer is right. An aggregation
    question does not: the graph computed the count, so `expected_count` in the eval row is
    an exact answer key derived from structure, the same discipline `questions.py` uses for
    gold chunk sets. This is the only judgement-free accuracy number in the project.

    It is also the slice where R@10 stops meaning anything -- the answer is a computed
    total, not a passage, so recall against gold chunks measures the wrong thing entirely.
    """
    idx = [i for i, q in enumerate(questions) if q.get("category") == "aggregation"]
    if not idx:
        return
    print("\n" + "-" * 74)
    print("aggregation — exact counts, scored against the graph's own totals")
    print("-" * 74)
    hits, misses = 0, []
    for i in idx:
        stated = stated_numbers(" ".join(c["text"] for c in rows[i]["claims"]))
        if questions[i]["expected_count"] in stated:
            hits += 1
        else:
            misses.append(questions[i]["qid"])
    print(f"  exact count correct         {hits}/{len(idx)}"
          + (f"   missed: {misses}" if misses else ""))
    print(
        "\n  No judge involved: the count comes from Cypher over the whole neighbourhood,\n"
        "  so the eval row carries the true total. R@10 is NOT meaningful on this slice --\n"
        "  the answer is a computed total, not a passage to be retrieved."
    )


def _report_cost(rows: list[dict[str, Any]]) -> None:
    """Latency and cost, measured only over questions that actually called the model.

    `chat_json` caches on content, so a rerun answers from disk in single-digit
    milliseconds at $0.00. Timing those rows reports the cache, not the system — the
    Phase 1 bakeoff shipped exactly this bug once, benchmarking the incumbent model
    against its own cached answers (docs/decisions.md). A refusal is excluded for the
    opposite reason: it is genuinely free and genuinely instant, and averaging it in
    understates what an answered question costs.
    """
    billed = [r for r in rows if r["usd"] > 0]
    refused_free = [r for r in rows if r["usd"] == 0 and not r["answerable"] and r["attempts"] == 0]
    cached = len(rows) - len(billed) - len(refused_free)

    print("\n" + "-" * 74)
    print("latency and cost per question — the other half of the Phase 5 table")
    print("-" * 74)
    print(f"  short-circuited refusals    {len(refused_free)}   (no model call, $0.00, ~2 ms)")
    print(f"  served from cache           {cached}")
    if not billed:
        print("\n  Every remaining question was served from cache, so there is no latency or\n"
              "  cost to report. Rerun with a cleared cache/ to measure them.")
        return
    lat = sorted(r["latency_ms"] for r in billed)
    print(f"  measured on                 {len(billed)} billed questions")
    print(f"  latency p50                 {statistics.median(lat):,.0f} ms")
    print(f"  latency p95                 {lat[int(len(lat) * 0.95) - 1]:,.0f} ms")
    print(f"  mean $/answered question    ${statistics.mean(r['usd'] for r in billed):.5f}")
    if cached > len(billed):
        print("\n  NOTE: most questions were served from cache. The numbers above describe the\n"
              f"  {len(billed)} that were not, which is a small and non-random sample.")

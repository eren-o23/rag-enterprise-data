"""`kgrag bench` — the Phase 5 benchmark: is the answer right, and what did it cost?

Every number this project has published so far is about *retrieval* or *provenance*. R@10
says the answering chunks came back; citation validation says every published id was really
retrieved. Neither says the answer is correct, and Phase 4's own report says so in as many
words. This module is the instrument that finally does, which makes it the most dangerous
code in the repo: the project has twice shipped an eval that could not fail (resolution
graded against its own answer key, routing resumed from a prompt that no longer existed).

So correctness is measured twice, by instruments that fail differently:

* **The key** — no model, no opinion. `expected_count` for aggregation, `expected_answers`
  (the terminal endpoint of the edge the question was templated off) for mined questions,
  and refusal for the out-of-scope five. It is a FLOOR: it rewards naming the canonical
  entity, so it structurally flatters the arm that answers out of the knowledge graph.
  That bias is the reason it is not the only instrument.
* **The judge** — one constrained call per answer, grading against the filing text that
  justified the edge. It sees the question, the evidence, and the answer, and never which
  system produced it.

The judge is validated before any of its numbers are read, three ways: against the exact
count key, against the structural key on every question that has one, and against a negative
control where each answer is graded for a DIFFERENT question. A judge that does not fail the
control is not a judge, and its output is discarded rather than caveated.
"""

from __future__ import annotations

import statistics
from typing import Any, Literal

import psycopg
from openai import APIConnectionError, APIStatusError
from pydantic import BaseModel, Field, ValidationError

from . import answer as answer_mod
from . import fireworks, jsonl
from .embed import connect
from .extract import _strict
from .questions import QUESTIONS
from .resolve import normalize
from .route import entity_index

JUDGE_MODEL = "accounts/fireworks/models/gpt-oss-120b"

#: One verdict off a short reference. Same reasoning as `route.ROUTER_TIMEOUT`: the module
#: default of 90s is sized for extraction streaming a whole chunk, and inheriting it turns a
#: stalled call into ten wasted minutes.
JUDGE_TIMEOUT = 30.0
JUDGE_ATTEMPTS = 3

#: Gold chunks shown to the judge. 3-hop questions carry up to a dozen; the evidence for the
#: claim is in the first few, and the rest is prompt cost. Truncation is reported.
REFERENCE_CHUNKS = 4
REFERENCE_CHARS = 4_000

#: The one-time cost of building the graph, which the vector baseline does not pay. Phase 1
#: extraction plus Phase 2 embedding, both printed by their own stages at the time.
INGESTION_USD = 1.41 + 0.57


# ---------------------------------------------------------------------------
# The key: correctness with no model in the loop
# ---------------------------------------------------------------------------


def surfaces(name: str, index: dict[str, list[dict[str, Any]]]) -> list[list[str]]:
    """Every normalized surface form that counts as naming this entity, as token lists.

    Resolution already mined these: `ADVANCED MICRO DEVICES INC` carries "AMD", and
    `Ernst & Young LLP` carries "EY" and "Ernst & Young". Without them the key marks
    *"Intel's auditor is Ernst & Young (EY)"* wrong for not saying "LLP", which is the key
    being pedantic about legal suffixes rather than about correctness.

    Tokens, not substrings: "EY" as a substring matches inside "they" and "survey".
    """
    forms = {name}
    for candidate in index.get(normalize(name), []):
        forms.update([candidate["name"], *candidate["aliases"]])
    return [t for t in (normalize(f).split() for f in forms) if t]


def names_it(text: str, name: str, index: dict[str, list[dict[str, Any]]]) -> bool:
    """Does the answer name this entity, under any surface form it is known by?"""
    tokens = normalize(text).split()
    for form in surfaces(name, index):
        if any(tokens[i : i + len(form)] == form for i in range(len(tokens) - len(form) + 1)):
            return True
    return False


def answer_text(row: dict[str, Any]) -> str:
    return " ".join(claim["text"] for claim in row["claims"])


def key_score(
    question: dict[str, Any], row: dict[str, Any], index: dict[str, list[dict[str, Any]]]
) -> str | None:
    """"correct" / "incorrect" / None when this question has no structural key.

    Refusals are scored here rather than by the judge: whether a refusal is right is a fact
    about the question, not a judgement about the text.
    """
    if question["hops"] == 0:
        return "correct" if not row["answerable"] else "incorrect"
    expected = question.get("expected_answers") or []
    # Whether a question HAS a key is a property of the question, not of the answer. Asking
    # about the answer first scores a refusal on a keyless question as incorrect, which
    # would give the two arms different key populations and make the columns incomparable.
    if not expected and question.get("category") != "aggregation":
        return None
    if not row["answerable"]:
        return "incorrect"
    text = answer_text(row)
    if question.get("category") == "aggregation":
        return "correct" if question["expected_count"] in answer_mod.stated_numbers(text) else "incorrect"
    # Any keyed answer counts. A merged row ("Which company did Teradyne acquire?" covers
    # three edges) is answered by naming any one of them -- the question is singular.
    return "correct" if any(names_it(text, name, index) for name in expected) else "incorrect"


# ---------------------------------------------------------------------------
# The judge
# ---------------------------------------------------------------------------


class Verdict(BaseModel):
    verdict: Literal["correct", "partial", "incorrect"] = Field(
        description="correct: the answer states what the reference supports. partial: part "
        "of it is right and nothing is wrong. incorrect: it contradicts the reference, "
        "answers a different question, or states something the reference does not support."
    )
    reason: str = Field(description="One short sentence. Name the specific mismatch.")


VERDICT_SCHEMA = _strict(Verdict.model_json_schema())

SYSTEM = """You grade answers to questions about SEC filings against reference evidence.

You are given a QUESTION, REFERENCE material taken from the filings, and an ANSWER produced
by a retrieval system. Decide whether the answer is correct.

RULES:
1. Grade only against the reference. Do not use outside knowledge about these companies.
2. The reference is evidence, not a model answer. An answer phrased differently, naming an
   entity by a different surface form, or adding true detail from the same evidence is still
   correct.
3. An answer that is about the right subject but does not answer the question asked is
   incorrect, however well written it is.
4. An answer that states a number, a name or a relationship the reference contradicts is
   incorrect.
5. If the reference does not settle it, say incorrect and name what is missing. Do not give
   the benefit of the doubt -- an unverifiable answer is not a correct one.
6. Use partial only when part of the answer is right and no part of it is wrong."""


def reference(conn: psycopg.Connection, question: dict[str, Any]) -> str:
    """The filing text that justified the edge this question was built from.

    Deliberately NOT the context the answering system retrieved: grading against that would
    measure whether the answer follows from what the system found, which is groundedness --
    already measured in Phase 4, and green even for an answer that is confidently wrong.

    Aggregation is the exception and is handed the verified count outright. The gold chunk
    for "how many subsidiaries does TI list?" is one page of an Exhibit 21, and asking a
    model to count 67 names off it would grade the judge's arithmetic instead of the answer.
    The count is a fact from Cypher; the judge's job on that slice is only to check the
    answer states it. Its agreement there is therefore a sanity check, not an independent
    measurement, and the report says so.
    """
    blocks = []
    if question.get("category") == "aggregation":
        blocks.append(
            f"VERIFIED COUNT (computed over the whole knowledge graph): "
            f"{question['expected_count']}"
        )
    texts = answer_mod.passages(conn, question["gold_chunk_ids"][:REFERENCE_CHUNKS])
    for passage in texts:
        blocks.append(
            f"[{passage['company']} · {passage['form']} · {passage['section_path']}]\n"
            f"{passage['text'][:REFERENCE_CHARS]}"
        )
    return "\n\n".join(blocks)


def judge_one(
    question: str, evidence: str, candidate: str, model: str = JUDGE_MODEL, use_cache: bool = True
) -> tuple[str, str]:
    """One verdict. Never raises -- an unreachable judge is an ungraded row, not a dead run."""
    user = f"QUESTION: {question}\n\nREFERENCE:\n{evidence}\n\nANSWER:\n{candidate}"
    try:
        payload = fireworks.chat_json(
            system=SYSTEM, user=user, schema=VERDICT_SCHEMA, model=model,
            timeout=JUDGE_TIMEOUT, attempts=JUDGE_ATTEMPTS, use_cache=use_cache,
        )
    except (APIStatusError, APIConnectionError) as exc:
        return "ungraded", f"judge_unreachable:{type(exc).__name__}"
    try:
        verdict = Verdict.model_validate(payload)
    except ValidationError:
        return "ungraded", "judge_invalid_verdict"
    return verdict.verdict, verdict.reason


# ---------------------------------------------------------------------------
# `kgrag bench`
# ---------------------------------------------------------------------------

ARMS = (("vector-only", False), ("graph + vector", True))

#: What counts as a right answer. `partial` is reported but never folded in -- half an
#: answer to "which company was acquired" is a different thing from an answer, and a
#: benchmark that blends them can move by choosing where to blend.
CORRECT = "correct"


def _rows(model: str, graph: bool) -> dict[str, dict[str, Any]]:
    """The current system's answers for one arm, straight out of `answer.py`'s own reader.

    `_prior` already enforces everything a benchmark needs: right model, right arm, written
    by the prompt and schema in the tree today, and no row whose synthesiser call never
    returned. Reimplementing that here would be a second set of rules to keep in sync, and
    the one that drifted would be the one publishing the numbers.
    """
    return answer_mod._prior(model, constrain=True, graph=graph)


def _verdict(
    question: dict[str, Any],
    row: dict[str, Any],
    conn: psycopg.Connection,
    model: str,
    use_cache: bool,
) -> tuple[str, str]:
    """The judge's verdict, with the cases that are structural rather than judgeable."""
    if not row["answerable"]:
        # Refusing an out-of-scope question is the right answer; refusing an answerable one
        # is a miss, and kept distinct from a wrong answer because they fail differently.
        return ("correct", "refused an out-of-scope question") if question["hops"] == 0 \
            else ("refused", row["refusal_reason"])
    if question["hops"] == 0:
        return "incorrect", "answered a question the corpus cannot support"
    return judge_one(question["question"], reference(conn, question), answer_text(row),
                     model=model, use_cache=use_cache)


def _rate(values: list[str], want: str = CORRECT) -> float:
    return sum(1 for v in values if v == want) / len(values) if values else 0.0


def _slices(questions: list[dict[str, Any]]) -> list[tuple[str, list[int]]]:
    """The five strata the spec names. Aggregation stays out of the hop slices."""
    out: list[tuple[str, list[int]]] = [
        (f"{h}-hop", [i for i, q in enumerate(questions)
                      if q["hops"] == h and not q.get("category")])
        for h in (1, 2, 3)
    ]
    out.append(("aggregation",
                [i for i, q in enumerate(questions) if q.get("category") == "aggregation"]))
    out.append(("out-of-scope", [i for i, q in enumerate(questions) if q["hops"] == 0]))
    return out


def run(model: str = answer_mod.SYNTH_MODEL, use_cache: bool = True) -> None:
    questions = list(jsonl.read(QUESTIONS))
    index = entity_index()
    arms = {label: _rows(model, graph) for label, graph in ARMS}
    for (label, graph), rows in zip(ARMS, arms.values()):
        missing = [q["qid"] for q in questions if q["qid"] not in rows]
        if missing:
            flag = "--constrained" if graph else "--baseline"
            raise SystemExit(
                f"{label}: {len(missing)} of {len(questions)} questions have no current "
                f"answer ({', '.join(missing[:5])}...).\n"
                f"Run `uv run kgrag answer {flag} --no-cache` first — and with --no-cache, "
                "or the latency and cost columns describe the filesystem."
            )

    print("=" * 74)
    print(f"benchmark — {len(questions)} questions, vector-only vs graph + vector")
    print("=" * 74)
    print(
        "Two instruments. The key has no opinion: exact counts, the endpoint the question\n"
        "was templated off, and refusal. The judge reads the filing text that justified the\n"
        "edge. The key is a floor and it flatters the graph arm, which answers in canonical\n"
        "entity names; the judge is validated below before any of it is read.\n"
    )

    with connect() as conn:
        keyed = {
            label: [key_score(q, rows[q["qid"]], index) for q in questions]
            for label, rows in arms.items()
        }
        judged: dict[str, list[str]] = {}
        reasons: dict[str, list[str]] = {}
        for label, rows in arms.items():
            pairs = [_verdict(q, rows[q["qid"]], conn, model, use_cache) for q in questions]
            judged[label] = [v for v, _ in pairs]
            reasons[label] = [r for _, r in pairs]

        _validate(questions, arms, keyed, judged, conn, model, use_cache)
        _accuracy(questions, keyed, judged)
        _cost(arms)
        _markdown(questions, judged, keyed)
        _misses(questions, arms, judged, reasons)


def _validate(
    questions: list[dict[str, Any]],
    arms: dict[str, dict[str, Any]],
    keyed: dict[str, list[str | None]],
    judged: dict[str, list[str]],
    conn: psycopg.Connection,
    model: str,
    use_cache: bool,
) -> None:
    """Three checks on the judge, run before its numbers are quoted."""
    print("-" * 74)
    print("instrument validation — a judge that cannot fail measures nothing")
    print("-" * 74)

    # A. Handed the exact count, does the judge reproduce it? Not an independent check --
    #    the reference carries the answer on this slice -- so a failure here means the judge
    #    cannot follow evidence at all, which would invalidate everything below it.
    agg = [i for i, q in enumerate(questions) if q.get("category") == "aggregation"]
    for label in arms:
        agree = sum(1 for i in agg if (judged[label][i] == CORRECT) == (keyed[label][i] == "correct"))
        print(f"  A. exact-count sanity   {label:16} {agree}/{len(agg)} agree with expected_count")

    # B. The real one: two instruments built from the same edge through different channels
    #    -- structured endpoint vs the model reading the filing prose.
    print()
    disagreements: list[tuple[str, str, str, str]] = []
    for label in arms:
        idx = [i for i, q in enumerate(questions)
               if keyed[label][i] is not None and q.get("category") != "aggregation"]
        agree = [i for i in idx if (judged[label][i] == CORRECT) == (keyed[label][i] == "correct")]
        print(f"  B. key agreement        {label:16} {len(agree)}/{len(idx)} "
              f"({len(agree) / len(idx):.3f})")
        disagreements += [
            (label, questions[i]["qid"], str(keyed[label][i]), judged[label][i])
            for i in idx if i not in set(agree)
        ]
    if disagreements:
        print("\n     disagreements — each is either a judge error or a key blind spot:")
        for label, qid, key, verdict in disagreements:
            print(f"       {qid:5} {label:16} key={key:9} judge={verdict}")

    # C. The negative control. Every answered row is re-graded against a DIFFERENT question,
    #    so a judge that rubber-stamps anything plausible is caught here and nowhere else.
    print()
    label, rows = "graph + vector", arms["graph + vector"]
    pool = [q for q in questions if q["gold_chunk_ids"] and rows[q["qid"]]["answerable"]]
    passed, leaked = 0, []
    for i, q in enumerate(pool):
        other = pool[(i + 1) % len(pool)]
        verdict, _ = judge_one(other["question"], reference(conn, other),
                               answer_text(rows[q["qid"]]), model=model, use_cache=use_cache)
        if verdict != CORRECT:
            passed += 1
        else:
            leaked.append(f"{q['qid']}->{other['qid']}")
    print(f"  C. negative control     {passed}/{len(pool)} answers rejected when graded "
          f"against a different question")
    if leaked:
        print(f"       accepted anyway: {', '.join(leaked)}")
    if len(pool) and passed / len(pool) < 0.9:
        raise SystemExit(
            "\nSTOP: the judge accepts answers to questions they do not answer. Its verdicts "
            "are not a measurement and nothing below this line should be published."
        )


def _accuracy(
    questions: list[dict[str, Any]],
    keyed: dict[str, list[str | None]],
    judged: dict[str, list[str]],
) -> None:
    print("\n" + "-" * 74)
    print("accuracy by stratum — both instruments, never blended")
    print("-" * 74)
    labels = [label for label, _ in ARMS]
    print(f"  {'slice':14} {'n':>3} {'key n':>6} "
          + " ".join(f"{'key ' + l.split()[0]:>12}" for l in labels)
          + " " + " ".join(f"{'judge ' + l.split()[0]:>14}" for l in labels))
    for name, idx in _slices(questions):
        if not idx:
            continue
        keyable = {l: [keyed[l][i] for i in idx if keyed[l][i] is not None] for l in labels}
        row = f"  {name:14} {len(idx):>3} {len(keyable[labels[0]]):>6} "
        row += " ".join(f"{_rate(keyable[l], 'correct'):>12.3f}" for l in labels)
        row += " " + " ".join(f"{_rate([judged[l][i] for i in idx]):>14.3f}" for l in labels)
        print(row)
    overall = {l: [judged[l][i] for i in range(len(questions))] for l in labels}
    print(f"  {'all':14} {len(questions):>3} {'':>6} "
          + " " * 26 + " " + " ".join(f"{_rate(overall[l]):>14.3f}" for l in labels))

    print("\n  verdict mix (judge):")
    for l in labels:
        mix: dict[str, int] = {}
        for v in overall[l]:
            mix[v] = mix.get(v, 0) + 1
        print(f"    {l:16} {mix}")
    print(
        "\n  'partial' is reported and never counted as correct. 'refused' is an answerable\n"
        "  question the system declined -- a miss, but a different kind from a wrong answer."
    )


def _cost(arms: dict[str, dict[str, Any]]) -> None:
    """Latency and cost per query. Billed rows only — a cached row times the filesystem."""
    print("\n" + "-" * 74)
    print("latency and cost per query")
    print("-" * 74)
    print(f"  {'arm':16} {'billed':>7} {'p50 ms':>9} {'p95 ms':>9} {'$/query':>10}")
    for label, rows in arms.items():
        billed = [r for r in rows.values() if r["usd"] > 0]
        if not billed:
            print(f"  {label:16} {'0':>7}   every row was served from cache — rerun with "
                  "--no-cache")
            continue
        lat = sorted(r["latency_ms"] for r in billed)
        print(f"  {label:16} {len(billed):>7} {statistics.median(lat):>9,.0f} "
              f"{lat[int(len(lat) * 0.95) - 1]:>9,.0f} "
              f"{statistics.mean(r['usd'] for r in billed):>10.5f}")
    print(f"\n  one-time ingestion, graph arm only: ${INGESTION_USD:.2f} "
          "($1.41 extract + $0.57 embed)")
    print("  the vector-only baseline pays none of it, which is the honest half of the claim")


def _markdown(
    questions: list[dict[str, Any]],
    judged: dict[str, list[str]],
    keyed: dict[str, list[str | None]],
) -> None:
    """The README table, emitted rather than transcribed by hand."""
    labels = [label for label, _ in ARMS]
    print("\n" + "-" * 74)
    print("README table")
    print("-" * 74)
    print("\n| slice | n | vector-only | graph + vector | delta |")
    print("|---|---|---|---|---|")
    for name, idx in _slices(questions):
        if not idx:
            continue
        v, g = (_rate([judged[l][i] for i in idx]) for l in labels)
        print(f"| {name} | {len(idx)} | {v:.3f} | {g:.3f} | {g - v:+.3f} |")
    v, g = (_rate(judged[l]) for l in labels)
    print(f"| **all** | {len(questions)} | **{v:.3f}** | **{g:.3f}** | **{g - v:+.3f}** |")


def _misses(
    questions: list[dict[str, Any]],
    arms: dict[str, dict[str, Any]],
    judged: dict[str, list[str]],
    reasons: dict[str, list[str]],
) -> None:
    """Every question the hybrid arm got wrong, with the judge's reason. Read these."""
    print("\n" + "-" * 74)
    print("what the graph arm still gets wrong")
    print("-" * 74)
    for i, q in enumerate(questions):
        verdict = judged["graph + vector"][i]
        if verdict == CORRECT:
            continue
        print(f"  {q['qid']:5} {verdict:9} {q['question'][:52]:54} {reasons['graph + vector'][i][:60]}")
    print(f"\n  vector-only comparison in {answer_mod.ANSWER_LOG}; both arms are logged in full.")

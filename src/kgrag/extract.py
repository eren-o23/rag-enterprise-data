"""Chunk -> validated (mentions, relations), via constrained decoding on an open model.

Two layers of defence, in this order:

1. The JSON schema, enforced by Fireworks' constrained decoding. Guarantees shape, and
   because the type fields are enums, guarantees the vocabulary. An off-ontology type is
   not "unlikely" here, it is unreachable.
2. `ontology.validate`, which checks the three things a schema cannot express: that
   endpoints were actually declared, that endpoint types match the relation signature,
   and that the quoted evidence really appears in the source chunk.

Layer 2's evidence check is the cheap hallucination filter. A relation the model invented
almost never comes with a span that survives a substring test against the chunk.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from openai import APIStatusError
from pydantic import ValidationError

from . import fireworks, jsonl, ontology
from .config import CHUNKS, EXTRACTIONS, FAILURES

#: Chunks probed before the full run, to turn "this feels affordable" into a number.
PROBE_CHUNKS = 15

SYSTEM = f"""You extract a knowledge graph from SEC filings. You are given one chunk of one filing.

Extract only what the chunk explicitly states. Do not infer, do not use outside knowledge, \
and do not add well-known facts that this chunk does not say.

{ontology.ontology_prompt_block()}

RULES
- `name` is the entity's surface form exactly as written in the chunk. Do not normalise or expand it.
- Every relation's `subject` and `object` must exactly match the `name` of a mention you declared.
- `evidence` must be copied verbatim from the chunk. Do not paraphrase or stitch fragments \
together; the span is checked against the source text and paraphrased evidence is discarded.
- confidence: "high" if the chunk states the relation directly, "medium" if the sentence \
strongly implies it, "low" if it is a plausible reading. Prefer omitting a relation to guessing.
- Many chunks state no relations at all. Boilerplate, forward-looking statements, and \
accounting policy are supposed to return empty lists. Returning nothing is a correct answer."""


def _strict(schema: dict[str, Any]) -> dict[str, Any]:
    """Constrained decoding needs objects closed, or the model bolts on extra keys."""
    if schema.get("type") == "object":
        schema["additionalProperties"] = False
    for value in schema.values():
        if isinstance(value, dict):
            _strict(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _strict(item)
    return schema


SCHEMA = _strict(ontology.Extraction.model_json_schema())


def _user_message(chunk: dict[str, Any]) -> str:
    return (
        f"FILING: {chunk['form']} filed {chunk['filing_date']} by {chunk['company']} "
        f"({chunk['ticker']})\n"
        f"SECTION: {chunk['section_path']}\n\n"
        f'In this chunk, "we", "us", "our", "the Company", and "the Registrant" all refer to '
        f'{chunk["company"]} — use "{chunk["company"]}" as the mention name for those references.\n\n'
        f"CHUNK:\n{chunk['text']}"
    )


def extract_one(chunk: dict[str, Any], model: str = fireworks.EXTRACT_MODEL) -> dict[str, Any]:
    """Extract one chunk. One retry with the validation error fed back, then quarantine.

    A single chunk can fail in a way no retry fixes - Fireworks has returned a flat 400
    ("safe_tokenization... not available for this model") on specific chunk content with
    nothing wrong in the request itself. Unretried, that one chunk killed a 2700-chunk
    run at chunk 196. `APIStatusError` is quarantined like a validation failure; only
    `fireworks.BudgetExceeded` (checked by name, not caught here) is meant to propagate
    and stop the whole run.
    """
    user = _user_message(chunk)
    last_error = ""

    for attempt in range(2):
        try:
            payload = fireworks.chat_json(
                system=SYSTEM,
                user=user + last_error,
                schema=SCHEMA,
                model=model,
            )
        except APIStatusError as exc:
            return {"chunk_id": chunk["chunk_id"], "error": f"{type(exc).__name__}: {exc}"}
        try:
            extraction = ontology.Extraction.model_validate(payload)
        except ValidationError as exc:
            last_error = (
                f"\n\nYour previous response failed validation. Fix it and return valid JSON:\n{exc}"
            )
            continue

        kept, dropped = ontology.validate(extraction, chunk["text"])
        return {
            "chunk_id": chunk["chunk_id"],
            "ticker": chunk["ticker"],
            "section_path": chunk["section_path"],
            "filing_date": chunk["filing_date"],
            "mentions": [m.model_dump(mode="json") for m in extraction.mentions],
            "relations": [r.model_dump(mode="json") for r in kept],
            "dropped": dict(dropped),
            "attempts": attempt + 1,
        }

    return {"chunk_id": chunk["chunk_id"], "error": last_error.strip()}


def run(limit: int | None = None, assume_yes: bool = False, model: str = fireworks.EXTRACT_MODEL) -> None:
    chunks = list(jsonl.read(CHUNKS))
    if not chunks:
        raise SystemExit("data/chunks.jsonl is empty — run `kgrag chunk` first.")
    if limit:
        chunks = chunks[:limit]

    done = {r["chunk_id"] for r in jsonl.read(EXTRACTIONS)}
    todo = [c for c in chunks if c["chunk_id"] not in done]
    print(f"{len(chunks)} chunks, {len(chunks) - len(todo)} already extracted, {len(todo)} to go")
    if not todo:
        _summarise()
        return

    # Cost projection before committing. This is the specific trap the project brief
    # warns about: it is very easy to discover the price of a corpus after paying it.
    if not assume_yes and len(todo) > PROBE_CHUNKS:
        before = fireworks.METER.usd
        for chunk in todo[:PROBE_CHUNKS]:
            extract_one(chunk, model)
        spent = fireworks.METER.usd - before
        projected = spent / PROBE_CHUNKS * len(todo)
        print(
            f"\nprobe: ${spent:.4f} over {PROBE_CHUNKS} chunks "
            f"(${spent / PROBE_CHUNKS:.5f}/chunk)\n"
            f"projected for all {len(todo)}: ${projected:.2f}\n\n"
            f"The probe results are cached, so continuing costs the remainder only.\n"
            f"Rerun with --yes to proceed (and --budget above ${projected:.2f})."
        )
        return

    failures, dropped_total = [], Counter()
    for i, chunk in enumerate(todo, 1):
        result = extract_one(chunk, model)
        if "error" in result:
            failures.append({**result, "section_path": chunk["section_path"]})
        else:
            jsonl.append(EXTRACTIONS, [result])
            dropped_total.update(result["dropped"])
        if i % 50 == 0 or i == len(todo):
            print(
                f"  {i}/{len(todo)}  ${fireworks.METER.usd:.3f}  "
                f"{len(failures)} quarantined  {sum(dropped_total.values())} relations dropped"
            )

    if failures:
        jsonl.append(FAILURES, failures)
    _summarise()


def _summarise() -> None:
    rows = list(jsonl.read(EXTRACTIONS))
    dropped: Counter[str] = Counter()
    for r in rows:
        dropped.update(r.get("dropped", {}))
    kept = sum(len(r["relations"]) for r in rows)
    total = kept + sum(dropped.values())

    print(f"\nextractions.jsonl: {len(rows)} chunks")
    print(f"  mentions          {sum(len(r['mentions']) for r in rows):,}")
    print(f"  relations kept    {kept:,}")
    if total:
        print(f"  relations dropped {sum(dropped.values()):,} ({sum(dropped.values()) / total:.1%})")
        for reason, count in dropped.most_common():
            print(f"    {reason:22} {count:,}")
    print(f"  quarantined       {len(list(jsonl.read(FAILURES)))}")

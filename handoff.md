# Handoff — Knowledge Graph RAG for Enterprise Data

_Last updated: 2026-08-19_

---

## Goal

Portfolio project 1 of an AI/ML engineering job-search portfolio: a hybrid knowledge-graph +
vector RAG system over SEC filings, built entirely on open-weight models (Fireworks-hosted,
$55 budget) instead of the Claude API the original brief specced. Currently mid-Phase 1
("Extract entities and relationships into Neo4j") of 5 phases; see
[rag-enterprise-data.md](rag-enterprise-data.md) (gitignored, local-only — the original spec)
and [docs/decisions.md](docs/decisions.md) for the full design-decision log.

---

## Current State

**Extraction is done and verified.** `data/extractions.jsonl` has 2,741 of 2,743 chunks
(2 quarantined into `data/failures.jsonl`), run to completion, output inspected:
- 14,431 mentions, 7,655 relations kept, 882 dropped (10.3%) — mostly `evidence_not_found`
  (780), the hallucination filter doing its job.
- Total spend: **$1.41** of $55 (`kgrag extract` prints this on every run).

**Gold-label set is done and validated.** `eval/extraction_gold.jsonl` — 20 chunks,
stratified across all 8 form/section types, hand-labeled with 103 mentions / 80 relations.
Every relation passed `ontology.validate()` with zero drops (verified by running it — see
scratch script referenced in the commit, not kept in-repo).

**Everything through the loader has run and passed.** `uv run pytest tests/` — 18 tests,
all passing (last run: this session, after the extract.py fixes below). `load.py` and
`verify.py` were built and smoke-tested against an **empty** graph (verify correctly fails
on it — see `git log` commit `8a5b0ea`), but have **not yet been run against the real 2,741
extracted chunks**. Docker (`neo4j` + `postgres`, both `running`) is up right now.

**Not started:** `resolve.py`'s 50-pair hand-labeled `eval/resolution_pairs.jsonl` (needed
for the τ sweep), the actual `kgrag resolve` / `kgrag load` / `kgrag verify` runs against
real data, and the 3-model bakeoff.

---

## Key Invariants

- **Chunk ids are the join key everywhere** (`sha256(accession|section_path|ordinal)[:16]`,
  in `chunk.py`). Phase 2's pgvector embeddings will join on this. Do not re-run `kgrag chunk`
  unless you intend to invalidate every downstream artifact — it would still produce the same
  ids for the same chunks (deterministic), but if the chunking *logic* changes, ids shift and
  `extractions.jsonl` / `extraction_gold.jsonl` chunk_ids go stale.
- **`.env` holds `FIREWORKS_API_KEY`, gitignored.** Not in the repo; if this is a fresh clone,
  it needs to be recreated from `.env.example` before anything that calls Fireworks will run.
- **pgvector is on host port 5433, not 5432** (a system Postgres usually owns 5432 — see
  `docker-compose.yml` comment). Neo4j is on the standard 7687/7474.
- **`fireworks.py`'s `Meter` is process-local, not persisted.** The `$1.41` spend figure
  above came from that run's printed report — there's no running total file. If you want a
  true cumulative spend figure across all runs so far, it has to be summed by hand from the
  task output logs (not saved anywhere either) or checked against the Fireworks billing
  dashboard directly.
- **`_pace()` in `fireworks.py` enforces max 9 requests/minute** (`REQUESTS_PER_MINUTE = 9`),
  one below the account's Fireworks quota (10 RPM, confirmed on the account's Quotas page).
  Do not remove this — see failure history below for why.
- **`resolve.py`'s `TAU = 0.86`** is a placeholder default, not yet tuned — the τ sweep
  (`resolve.sweep()`) needs `eval/resolution_pairs.jsonl` to exist first, which it doesn't yet.

---

## What We Tried That Failed

| Approach | Why it failed |
|----------|--------------|
| Sequential extraction loop with plain exponential backoff on 429s, no rate pacing | Backoff (up to 60s/step) massively overshoots recovering from a 10 RPM cap that resets every minute — measured throughput ~2.4 chunks/min, projected 17h for the full corpus. Fixed by adding `_pace()` to stay under the quota instead of reacting to it. |
| `OpenAI()` client with default (no explicit) timeout | A dropped connection blocks the read forever with no exception — the process stalled **11.6 hours overnight** with sockets sitting in `CLOSE_WAIT`, looking alive in `ps` the whole time, 0 progress. Fixed with `timeout=90.0, max_retries=0` on the client. |
| `extract_one` only catching `pydantic.ValidationError` | Two separate crashes: a `400 BadRequestError` ("safe_tokenization... not available for this model") on one specific chunk's content, and later an `APITimeoutError` that burned all 6 backoff retries (~10 min) before re-raising. Both killed the whole 2700-chunk run outright. Fixed by having `extract_one` catch `(APIStatusError, APIConnectionError)` and quarantine into `failures.jsonl` instead of propagating — `BudgetExceeded` (a separate RuntimeError, not an openai exception) still propagates on purpose. |
| `get_filings(form="10-K")` without `amendments=False` | Prefix-matches `10-K/A` too. AMD's and Skyworks' "latest 10-K" resolved to amendments that only carry the items they amend (AMD's exposed just Items 7 and 15) — both companies silently lost their entire Business/Risk Factors/Legal Proceedings sections with no error. Looked exactly like a parser bug; wasn't one. |
| First DEF 14A prefilter (`GOVERNANCE` regex matching any of `director`, `board`, `officer`, etc.) | Kept 2,126 of 2,126 proxy chunks — those words appear on nearly every page of a proxy. Replaced with a regex requiring an actual statement of board *service*, which drops to 971 and removes the compensation-table noise. |
| Normalizing entity names by stripping industry words (`semiconductor`, `technology`, etc.) alongside legal suffixes | Turned "Taiwan Semiconductor Manufacturing" into "taiwan manufacturing" and would have merged "ON Semiconductor" with unrelated "ON ..." names. Reverted to legal-suffix-only stripping. |

---

## Don't Touch

- `docs/decisions.md` is a living ADR log, already fairly long — add to it, don't restructure
  it, when new decisions get made in Phase 1's remaining steps.
- `eval/extraction_gold.jsonl` — the evidence spans were hand-verified against
  `ontology.validate()` with zero drops. If you touch it, re-run that validation before
  trusting it again (a scratch one-liner: build an `Extraction` from each row's
  mentions/relations and call `ontology.validate(ext, chunk_text)`, assert no drops).
- Docker containers are currently running with real data in Neo4j's volume from earlier
  smoke tests (a couple of test nodes prefixed `test_` from `tests/test_pipeline.py`, which
  clean up after themselves) — nothing production-critical, safe to `docker compose down` if
  needed, but no reason to right now.

---

## Next Step

Build `eval/resolution_pairs.jsonl` — ~50 hand-labeled `{a, b, type, same}` pairs sampled
from **real near-miss candidates** in the actual extracted mentions (not invented), the same
way the gold set was built: query `resolve._mentions()` or similar to surface the surface
forms that actually appear in `data/extractions.jsonl`, pick genuinely ambiguous pairs (like
the AMD/Advanced Micro Devices/Advanced Energy Industries triad already used as a test case),
label them, then run `resolve.sweep()` to pick a real τ before running `kgrag resolve` for
real.

After that, in order: `kgrag resolve` → `kgrag load` → `kgrag verify` (the actual Phase 1
gate — checks 3-hop paths exist, orphan rate, no empty relation types) → 3-model bakeoff on
the gold set → fill in README Phase 1 numbers.

---

## Open Questions / Blockers

_None currently blocking._ Fireworks account is confirmed working, budget has enormous
headroom ($1.41 of $55 spent), Docker stack is healthy, all code through `load.py`/`verify.py`
is written and unit-tested — the remaining Phase 1 work is data (the resolution pairs) and
then just running the pipeline stages that haven't touched real data yet.

---

## Session History

_Append-only. One line per session — never overwrite previous entries._

- 2026-08-19: Ran full extraction to completion (2,741/2,743 chunks, $1.41, 7,655 relations)
  after fixing three separate run-killing bugs found the hard way (rate-limit backoff
  overshoot, a silent 11.6-hour connection stall from no client timeout, and two exception
  types extract_one wasn't quarantining); built and validated the 20-chunk hand-labeled gold
  set for the model bakeoff.

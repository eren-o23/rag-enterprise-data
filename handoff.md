# Handoff — Knowledge Graph RAG for Enterprise Data

_Last updated: 2026-08-24_

---

## Goal

Portfolio project 1 of an AI/ML engineering job-search portfolio: a hybrid knowledge-graph +
vector RAG system over SEC filings, built entirely on open-weight models (Fireworks-hosted,
$55 budget) instead of the Claude API the original brief specced. **Phases 1, 2 and 3 of 5 are complete;
Phase 4 ("Merge both sources into one grounded answer") is the next work.** See
[rag-enterprise-data.md](rag-enterprise-data.md) (gitignored, local-only — the original spec)
and [docs/decisions.md](docs/decisions.md) for the full design-decision log.

---

## Current State

**Phase 3 is complete and measured.** `kgrag route` routes each question to the graph
path, the vector path, both, or a refusal. **Routing accuracy 54/57 (94.7%)**; routed
retrieval beats vector-only at every hop count, and the 2-hop slice goes **0.288 -> 0.758
(2.6x)**. Router costs **$0.021** per full sweep on `gpt-oss-120b` and reruns are free.
See README "Phase 3 results".

| slice | n | vector | graph | routed |
|---|---|---|---|---|
| 1-hop | 30 | 0.586 | 0.792 | 0.728 |
| 2-hop | 10 | 0.288 | 0.775 | 0.758 |
| 3-hop | 12 | 0.328 | 0.704 | 0.671 |
| all | 52 | 0.469 | 0.768 | 0.721 |

Note **graph-only (0.768) outscores routed (0.721)** — routing is not free accuracy, it
currently leaves ~5 points on the table by sending some questions to vector that the graph
answers better. What it buys is refusal and not traversing when there is no graph answer.

**Extraction is done and verified.** `data/extractions.jsonl` has 2,741 of 2,743 chunks
(2 quarantined into `data/failures.jsonl`), run to completion, output inspected:
- 14,431 mentions, 7,655 relations kept, 882 dropped (10.3%) — mostly `evidence_not_found`
  (780), the hallucination filter doing its job.
- Total spend: **$1.41** of $55 (`kgrag extract` prints this on every run).

**Gold-label set exists but is recall-only.** `eval/extraction_gold.jsonl` — 20 chunks,
stratified across all 8 form/section types, hand-labeled with 103 mentions / 80 relations.
Every relation passed `ontology.validate()` with zero drops. But the same 20 chunks yield
305 mentions from the model, so gold is a ~3x-sparse *subset* of truth, not an answer key
(see Key Invariants).

**Phase 2 is complete and the gate passes.** 2,743 chunks in pgvector at 1024/2000/4096
dims for **$0.57**; `uv run kgrag verify` → `PASS` now covers both stores plus the join
between them (20/20 sampled chunk_ids resolve to a Neo4j edge). Retrieval measured:
1-hop R@10 .586, 2-hop .288, 3-hop .328 — multi-hop at roughly half of single-hop, which is
the Phase 5 baseline. All three embedding widths are statistically indistinguishable, so
1024 ships. See README "Phase 2 results" and `docs/decisions.md`.

**Phase 1 is complete and the gate passes.** `uv run kgrag verify` → `PASS`: 4,496 nodes,
4,685 edges, 0 self-loops, all 14 relation types populated, shared-director and 3-hop paths
both non-empty. `uv run pytest tests/` — 21 tests (one skips when Neo4j is down).
README's "Phase 1 results" section has the full numbers; `docs/decisions.md` has the
reasoning.

**Docker is currently RUNNING** (both containers healthy).

**Docker was previously STOPPED.** It was shut down to free memory for the local model in the
bakeoff — this machine has 8 GB and swap was near capacity. `docker compose up -d` restores
it; the Neo4j volume is intact, so the graph is still there and does not need reloading.

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
- **`resolve.py`'s `TAU = 0.88` and `JACCARD_FLOOR = 0.67`** are tuned, and sit in the middle
  of a flat plateau rather than on a peak. Retune with `kgrag sweep` only after
  `kgrag mine-pairs` regenerates the labels.
- **`data/aliases.jsonl` is an input to `resolve`, not just an eval artifact.** It holds
  aliases the filings declare about themselves and is regenerated by `kgrag mine-pairs`.
  `data/overrides.jsonl` is hand-decided, is NOT regenerable, and is the one file under
  `data/` that is committed to git.
- **A false merge costs more than a missed one.** Merging a parent into its subsidiary
  collapses their `SUBSIDIARY_OF` edge to a self-loop, and `load.py` silently discards it —
  the failure prints a count on a *passing* run. Watch `self_loop_after_resolution` after any
  resolution change; it is 45 now and was 321.
- **The gold set is ~3x under-labeled** (103 mentions where the model finds 305), so it
  supports recall comparisons only. Do not quote precision or F1 from `kgrag bakeoff`.

### Phase 4 specifics — read before writing any synthesis code

- **`kgrag route` is the input to Phase 4.** `route.route()` returns the log row, which
  keeps `graph_ids`, `vector_ids` and the merged `chunk_ids` as separate lists — the
  labelled-context split Phase 4 needs is already there.
- **The model never authors Cypher, and that rule must survive Phase 4.** `route.arrows()`
  interpolates a relationship type only after `RelationType(...)` accepts it. Same rule as
  `load.py` at write time. `test_traversal_rejects_a_predicate_the_model_invented` asserts
  it against injection-shaped strings.
- **The router runs on `gpt-oss-120b`, not the 20b.** The 20b stalls reproducibly on 6 of
  57 questions and a timed-out call records no usage, so the failure shows up as
  `$0.00000` spend and unrouted questions rather than as an error rate. Do not "optimise"
  the router back down to the 20b without rerunning the eval and reading the degradations
  line.
- **`ROUTER_TIMEOUT = 20.0` and `ROUTER_ATTEMPTS = 3`, passed through `chat_json`.** The
  module default is still 90s/6 for extraction. Do not collapse the two.
- **`kgrag route` resumes from `data/routing_log.jsonl` and deliberately will not resume
  two things:** a row whose router call timed out (a fallback, not a decision) and a row
  whose `router_sha` does not match the current prompt+schema. Use `--fresh` after changing
  traversal or ranking code, which the sha cannot see.
- **Entity lookup does NOT use the Neo4j fulltext index.** It exists and is the *fallback*.
  The exact normalized-alias index in `route.entity_index()` is the primary, because
  fulltext loses "AMD" to `AMD Japan Ltd.` (see decisions.md).
- **`data/routing_log.jsonl` is append-only and gitignored**, but it is regenerable for
  $0.00 — router calls are content-cached. Phase 5 reads it.

### Phase 3 specifics — retained

- **1024 is the production embedding column.** All three widths were measured and are
  statistically indistinguishable (paired bootstrap, every CI crosses zero). `emb_2000` and
  `emb_4096` are kept for the README table; do not query them in production code.
- **`emb_4096` has no index and never can** — pgvector caps HNSW at 2,000 dims. A query
  against it is always a full scan.
- **Force the planner in any retrieval benchmark.** At 2,743 rows Postgres costs a seq scan
  cheaper than an HNSW probe and takes it even when the index exists, silently turning an
  "ANN" measurement into a second exact scan that reports recall 1.000 everywhere. Both arms
  need forcing: `enable_seqscan = off` for the index arm, `enable_indexscan = off` for the
  exact arm. This already bit once — the tell was a non-monotonic recall curve.
- **`fireworks.embed()` now takes `dimensions`, and it IS in the cache key** — but only as a
  suffix when non-None, so `dimensions=None` still hashes as it did in Phase 1 and resolve's
  entity-name cache stays valid. Do not "simplify" that asymmetry away.
- **Entity ids are deliberately not frozen.** `kgrag embed` upserts metadata in a statement
  separate from the vector UPDATEs and skips chunks whose column is already populated, so a
  resolution change refreshes `entity_ids` for $0.00.
- **`eval/questions.jsonl` is committed and half of it is not regenerable.** `kgrag
  mine-questions` rewrites the mined rows and preserves any row with `"source": "hand"`.
  The 10 hand-written rows are as unrecoverable as `data/overrides.jsonl`.
- **`kgrag recall` exits 1 if 1-hop R@10 drops below 0.35** (`MIN_1HOP_RECALL_AT_10`).
  That is a catastrophe floor against a measured 0.586, not a quality threshold — it is
  sized so noise cannot trip it, and it will not notice gradual drift. Deliberately not in
  `kgrag verify`: that gate needs only the two databases, and scoring recall would make it
  require an API key and network on a fresh clone.
- **Retrieval gold sets are floors, not exhaustive.** Quote recall as a lower bound. The
  bound is identical across widths, which is what makes cross-width comparison valid.

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
- **The Neo4j volume holds the finished Phase 1 graph.** Containers are stopped, not removed.
  `docker compose up -d` brings it back with the 4,496 nodes / 4,685 edges intact — do not
  re-run `kgrag load` to "restore" it, and do not `docker compose down -v`, which would drop
  the volume and cost a full reload.
- `data/overrides.jsonl` — 16 hand-decided pairs, not regenerable, the only file under
  `data/` that git tracks. `kgrag mine-pairs` does not touch it.
- **The 10 hand-written rows in `eval/questions.jsonl`** (`"source": "hand"`). Same status
  as `overrides.jsonl`: not regenerable. `kgrag mine-questions` preserves them on rerun.
- **The pgvector volume holds 2,743 chunks × 3 widths ($0.57 of embedding).** Like the
  Neo4j volume: `docker compose down -v` drops it and costs a full re-embed.
- The README's Phase 1 results are written but the "What broke" narrative section is
  **deliberately deferred to the end of the project** — the user wants documentation polish
  done once, after Phase 5, not per-phase.

---

## Next Step

**Phase 4: merge both sources into one grounded answer.** `kgrag route` already returns
ranked `chunk_id`s from whichever path(s) it chose, which is the input Phase 4 needs.

1. Convert graph paths into readable statements before they reach the prompt. Raw triples
   generate awkward text. The traversal already returns the relationship chain, so the
   verbaliser has the predicate and both endpoint names.
2. Deduplicate across the two sets, then assemble context with **explicit labels** keeping
   graph-derived facts separate from retrieved passages.
3. Require a citation per claim and validate that every citation resolves to a chunk_id
   that was actually retrieved. Reject and regenerate when it does not. This is also the
   real backstop for out-of-scope questions — the router's `refuse` is a guess made before
   looking at the corpus (see decisions.md).
4. `route.route()` returns the full log row including `graph_ids`, `vector_ids` and
   `chunk_ids` separately, so the labelling in step 2 needs no extra bookkeeping.

**One decision waiting on you:** `h007` ("Who is the chief executive of OpenAI?") is
gold-labelled out-of-scope because OpenAI is "outside the 24-filer corpus". OpenAI *is* in
the graph — a Company node, 5 mentions, `PARTNERS_WITH` NVIDIA, two `DIRECTOR_OF` edges.
The entity is present; the fact asked for is not. The router now routes it to `vector`,
which is scored as the only out-of-scope error (0.800). The row was **not** edited — the
10 hand-written rows are not regenerable, and silently relabelling an eval row that
disagrees with the system is how an eval stops being able to fail. Either relabel it
deliberately or leave the 0.800 standing.

Optional cleanup, none of it blocking:

- The two remaining genuine routing errors are multi-hop questions routed to vector
  (`h001` aggregation, `m032` 2-hop).
- The 45 remaining `self_loop_after_resolution` edges (unchanged since Phase 1). Fixing
  them means not stripping national legal forms (`GmbH`, `AG`, `Oy`) in `normalize()`.
  Entity ids are deliberately NOT frozen, so this costs a `resolve` + `embed` rerun that
  refreshes `entity_ids` in Postgres for **$0.00** and re-embeds nothing.
- Exhaustively re-label the 20 extraction gold chunks so the bakeoff can report precision.
- ~39 orphan Exhibit 21 subsidiary shells with no `SUBSIDIARY_OF` edge.

## Open Questions / Blockers

_None blocking._ Fireworks is working with enormous budget headroom ($1.41 of $55), and
Phase 1 is complete and committed. Two things to be aware of rather than solve:

- **This machine is memory-constrained**: 8 GB, and swap was near capacity during the bakeoff
  (that is what crashed it earlier in the session, running a larger local model). Neo4j and
  Postgres running together plus an index build is the heaviest thing Phase 2 will do. Embedding
  itself is network-bound and cheap. If a local model is ever needed again, stop Docker first.
- **One decision to make early:** whether entity ids are frozen before they are written into
  Postgres as metadata (see Key Invariants). Deferring is fine — it costs a metadata `UPDATE`,
  not a re-embed — but decide deliberately rather than discovering it later.

---

## Session History

_Append-only. One line per session — never overwrite previous entries._

- 2026-08-23: Closed Phase 1. Found `kgrag load` discarding 317 `SUBSIDIARY_OF` edges (47% of
  the relation) as self-loops on a *passing* run, because resolution was merging subsidiaries
  into their parents; the old eval set couldn't see it, since ranking candidates by
  |cos − τ| structurally excludes the near-identical pairs that over-merge. Replaced it with
  `kgrag mine-pairs`, deriving labels from filing structure (alias declarations for positives,
  Exhibit 21 for negatives), which showed the real discriminator is geographic tokens.
  Resolution now P=1.000/R=0.297 held out; SUBSIDIARY_OF 364→684, self-loops 321→45. Also
  caught two of my own errors: a rule that scored best on the eval set but merged unrelated
  people, and an F1 of .986 that was graded against its own answer key. Ran the bakeoff
  (gpt-oss-120b wins on recall; llama3.2:3b not viable — 5/20 failures, 60 min), after
  discovering its first run measured cache hits for the incumbent. Wrote README Phase 1.
- 2026-08-19: Ran full extraction to completion (2,741/2,743 chunks, $1.41, 7,655 relations)
  after fixing three separate run-killing bugs found the hard way (rate-limit backoff
  overshoot, a silent 11.6-hour connection stall from no client timeout, and two exception
  types extract_one wasn't quarantining); built and validated the 20-chunk hand-labeled gold
  set for the model bakeoff.
- 2026-08-24: Closed Phase 2. Embedded 2,743 chunks into pgvector at three widths ($0.57)
  and measured rather than asserted the two decisions that were open: Matryoshka truncation
  to 1024 costs nothing detectable (paired bootstrap, all pairwise CIs cross zero), and the
  retrieval eval shows multi-hop at roughly half of single-hop, which is the Phase 5
  baseline. Two bugs caught by the measurements disagreeing with themselves: the ANN sweep
  was non-monotonic because Postgres prefers a seq scan at 2,743 rows and was silently
  answering the "ANN" arm exactly (both planner arms now forced, confirmed with EXPLAIN),
  and the question miner emitted duplicate question text with different gold sets, which
  makes an eval unanswerable. Also corrected the handoff's own plan: it proposed embedding
  20 gold chunks at 4096, which cannot rank against anything.
- 2026-08-24: Closed Phase 3. Built `kgrag route`: one constrained-decoding call returns
  route + entities + a predicate chain, the chain is interpolated into Cypher only after
  `RelationType()` accepts it, and both paths return ranked chunk_ids so they are directly
  comparable. Routing accuracy 54/57; routed retrieval beats vector at every hop count and
  2-hop goes 0.288 -> 0.758. Four bugs found by measurements disagreeing with themselves:
  the fulltext index created in Phase 1 for entity lookup cannot resolve "AMD" (Lucene
  length-normalises it below `AMD Japan Ltd.`); `gpt-oss-20b` stalls forever on 6 of 57
  questions while the 120b answers them in ~1s, and because a timed-out call records no
  usage the failure reads as `$0.00000` spend rather than an error rate; the eval was
  charging those router failures to the graph column, understating 2-hop graph recall as
  0.433 when it is 0.675; and describing the corpus as "24 companies" made the router
  refuse subsidiaries, products and auditors that are in its own graph (0.895 -> 0.947 to
  fix). Also caught my own resume silently replaying decisions made by a prompt that no
  longer existed — an edit + rerun reported identical numbers at $0.00000 spend — now
  guarded by `router_sha`. Left `h007` mislabelled rather than edit a non-regenerable eval
  row; it is the only out-of-scope error and is written up for a decision.

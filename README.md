# Knowledge Graph RAG for Enterprise Data

[![CI](https://github.com/eren-o23/rag-enterprise-data/actions/workflows/ci.yml/badge.svg)](https://github.com/eren-o23/rag-enterprise-data/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![Neo4j](https://img.shields.io/badge/Neo4j-4581C3?logo=neo4j&logoColor=white)
![pgvector](https://img.shields.io/badge/pgvector-4169E1?logo=postgresql&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Fireworks](https://img.shields.io/badge/Fireworks%20AI-open--weight%20models-FF6B35)

**Retrieval that answers the questions embeddings quietly fail on.**

**Why it exists.** Almost everyone has built vector RAG. It works until a question needs two or
three hops across related entities — *who competes with the companies Teradyne supplies?* — where
no single passage contains the answer and similarity search has nothing to rank. This is the layer
that crosses those hops: a knowledge graph over the same chunks the vector index holds, a router
that picks between them, and one grounded answer whose every citation resolves to a chunk that was
actually retrieved.

**Start here.** [What the benchmark actually produced](#what-the-benchmark-actually-produced) is
one table and shows the whole arc. If you read one design section, make it
[citations are chunk ids](#citations-are-chunk-ids-and-that-is-the-spine) — it is the decision
everything else falls out of.

## The short version

- **A graph and a vector index over the same chunks**, joined on one deterministic id, so a graph
  path can be read back as text and a retrieved passage can be walked into the graph.
- **The model never writes Cypher.** It fills enum-constrained parameters; a relationship type is
  interpolated only after `RelationType()` accepts it. Security control and reproducibility in one.
- **A citation is a chunk id**, not a footnote number — so validating one is set membership.
  **Zero invented citations**: 216 published across both benchmark arms, and 319 across Phase 4's
  free-vs-constrained comparison.
- **Exact aggregation.** "How many subsidiaries does TI list?" is a Cypher count over the whole
  neighbourhood, because you cannot count from a top-k sample.
- **Measured, not asserted.** Multi-hop accuracy **0.091 → 0.591** against a vector-only baseline
  built from the same synthesiser and the same citation contract, graded by a judge that is
  validated three ways before its numbers are quoted, with a bootstrap interval on every claim.
- **The honest half:** the graph arm is **2x slower**, **12% dearer per query**, and costs **$1.98**
  to build. One-hop accuracy is a statistical tie.
- **53 tests**, no services needed, and six bugs that shipped on green runs — each now a test.

**Stack:** Python 3.12 · Neo4j · pgvector · FastAPI · Fireworks (`gpt-oss-120b`,
`qwen3-embedding-8b`) · Docker Compose. No LangChain. Total spend across five phases: **~$3.7**
of a $55 budget — $3.36 recorded in the run logs, plus the judge's calls, which are not.

## What the benchmark actually produced

Two systems, the same 65 questions, the same synthesiser, the same citation contract. The only
difference is retrieval: the baseline embeds the question, takes the top 10 passages, and makes no
router call at all. Both arms ran uncached.

| slice | n | vector-only | graph + vector | delta | 95% CI |
|---|---|---|---|---|---|
| 1-hop | 30 | 0.633 | **0.700** | +0.067 | [-0.100, +0.233] — a tie |
| 2-hop | 10 | 0.100 | **0.800** | +0.700 | [+0.400, +1.000] |
| 3-hop | 12 | 0.083 | **0.417** | +0.333 | [+0.083, +0.583] |
| aggregation | 8 | 0.125 | **0.750** | +0.625 | [+0.125, +1.000] |
| out-of-scope (refuse) | 5 | 0.800 | 1.000 | +0.200 | [+0.000, +0.600] |
| **all** | **65** | **0.400** | **0.692** | **+0.292** | **[+0.154, +0.431]** |

| | vector-only | graph + vector |
|---|---|---|
| latency p50 / p95 | **2,487 / 8,688 ms** | 5,088 / 12,301 ms |
| $ per query | **$0.00145** | $0.00162 |
| one-time ingestion | **$0.00** | $1.98 |

**Parity at one hop, and a gap that opens with depth.** The 1-hop interval crosses zero, so on 30
questions the two systems are indistinguishable there — the honest version of the result, and a
stronger claim than a small unexplained edge. Every multi-hop and aggregation interval clears zero.
That curve is the whole point: embeddings answer what a passage states and lose what a chain
implies.

**The baseline does not fail by hallucinating.** It refuses — 28 of 60 answerable questions, where
the hybrid refuses 9 — and its refusals are honest: it retrieves ten passages about the right
company and correctly reports that the fact is not in them. *"The provided passages list many AMD
subsidiaries"*, and the question asked how many there are.

Read the numbers for what they are: 65 questions, so a slice of 5-12 cannot resolve a small
difference, and re-running identical code moves a slice by about one question. That is why every
delta carries a paired-bootstrap interval and why the tables above are quoted to three decimals
nowhere in the prose. Latency excludes rate-limit sleep — see
[trade-offs](#trade-offs-and-known-ceilings).

## Run it

```bash
cp .env.example .env          # FIREWORKS_API_KEY and SEC_USER_AGENT
docker compose up -d          # neo4j + postgres/pgvector
uv sync

uv run kgrag all              # fetch → chunk → extract → resolve → load → embed
uv run kgrag verify           # the gate: both stores, and the join between them
uv run uvicorn kgrag.api:app
```

`kgrag all` is ~$2 of model calls and a few hours, mostly paced against a 10 RPM quota. Every stage
resumes: `extract` skips chunk ids already done, `embed` resumes from `WHERE emb_1024 IS NULL`, and
`route`/`answer` resume from their own logs unless the prompt fingerprint changed.

```bash
curl localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question":"Who is INTEL CORP'"'"'s independent auditor?"}'
```

```json
{"answerable": true,
 "answer": "Intel Corp's independent auditor is Ernst & Young LLP.",
 "route": "graph",
 "citations": ["c8608131724ee274", "ecec386ead8c9bcc"]}
```

Both citations are the full gold set for that question. `POST /ask/stream` returns the same answer
as server-sent events, one claim at a time.

Measuring rather than serving:

```bash
uv run kgrag route                          # route each question, score the decision
uv run kgrag answer --constrained --no-cache   # the graph arm
uv run kgrag answer --baseline --no-cache      # the vector-only arm
uv run kgrag bench                          # judge both, validate the judge, print the table
```

## How it works

```
                     ┌────────────────────────────────────────────────┐
   question ────────►│  route → traverse / search → verbalise → answer │
                     │    │           │                               │
                     │    │           ├── graph: Cypher over 4,685 edges
                     │    │           └── vector: HNSW over 2,743 chunks
                     │    └── refuse ──────────────────────────► $0.00, ~2 ms
                     └───────────────────────┬────────────────────────┘
                                             ▼
                             every claim cites a chunk id that
                             was actually retrieved, or is rejected

   ingestion, once, $1.98:
   EDGAR → chunk (sha256 id) → extract (closed ontology, evidence span verified)
                             → resolve (3 named match rules) → Neo4j
                             → embed → pgvector          both keyed on the same id
```

- **`route.py`** makes one constrained-decoding call that returns a route, the entities named in
  the question, and a chain of relation types — all enums. A chain whose length contradicts its
  declared shape, or entities that resolve to nothing, degrade to another route with a logged
  reason. Then a parameterised traversal, and both paths return the same currency: ranked chunk ids.
- **`answer.py`** verbalises graph paths subject-first off `startNode`/`endNode` — a neighbourhood
  walk is undirected, and reading direction off path order publishes "AMD supplies TSMC" backwards
  with no error anywhere. A whole walk renders as one fact, `X supplies Y → Y competes with Z`,
  because chain position is not recoverable from a pile of true facts.
- **`resolve.py`** collapses surface forms with three named rules rather than one cosine threshold.
  A false merge costs more than a missed one: merging a parent into its subsidiary turns their edge
  into a self-loop that `load.py` silently discards.
- **`ontology.py`** is closed — 7 entity types, 14 relations — and every extracted relation must
  carry a verbatim evidence span that survives a substring check against its source chunk. 780
  relations were dropped by that check, without a second model call.
- **`judge.py`** is the Phase 5 instrument: a structural answer key with no model in it, an LLM
  judge, and three validations that run before either is quoted.

### Citations are chunk ids, and that is the spine

`sha256(accession|section_path|ordinal)[:16]`, minted once in `chunk.py`. Every Neo4j edge carries
the ids of the chunks whose text justified it; pgvector is keyed on the same id; the answer cites
in the same currency. Four things fall out of that one decision:

| | because the id is the join key |
|---|---|
| **validation** | a citation resolves or it does not — set membership, no mapping to get wrong |
| **the eval set** | gold chunks are derived from graph structure, not hand-judgement |
| **mixing sources** | a graph fact and a retrieved passage cite identically, so one answer can hold both |
| **crossing over** | a passage can be walked into its graph neighbourhood, and a path read back as text |

The citable set is returned *by* the function that renders the context, never computed by the
caller. They drifted once — the caller checked against the route's top-10 while the context printed
up to 60 ids — and six correct answers were thrown away for citing what was plainly in front of
them.

## Two decisions worth defending

**Why not LangChain.** Fireworks speaks the OpenAI API, so the pipeline is `openai` + `neo4j` +
`pydantic`. The retry, cache, cost-guard and schema-validation layers are the interesting part of
this project; delegating them would leave something that configures a library. It also would not
have produced the parts that turned out to matter: a cache key that separates embedding widths
without invalidating older entries, a rate-limit pacer that stays under a quota instead of reacting
to it, and a budget guard that stops a run rather than discovering the spend afterwards.

**Why two instruments for correctness, and why the judge is validated first.** The structural key
has no opinion — exact counts, and the far endpoint of the edge each question was templated off —
but it matches on surface form, and the graph arm answers in canonical entity names while the
baseline answers in filing prose. An instrument biased toward the system the benchmark exists to
promote cannot be the only one. So an LLM judge grades the fact asked for, and `bench` validates it
against the exact-count key, against the structural key with **every departure printed by qid**, and
against a negative control that re-grades each answer for a *different* question. Below 90%
rejection on that control, `bench` exits rather than publishing.

That is not paranoia. The judge's first two versions were both wrong, both in the direction that
flattered the system under test, and the per-row printout is what caught them.

## Trade-offs and known ceilings

Deliberate shortcuts are marked `ponytail:` in the source, each naming its ceiling. The ones worth
knowing before reading the code:

- **Latency is two sequential model calls and 31 ms of retrieval.** The graph contributes 0.6%.
  Three attempts to improve it all failed, and are written up rather than hidden: streaming buys
  0.5-6% (73% of a synthesis call elapses before the first content token — the model reasons before
  it writes), halving the prompt buys 19% of cost and 1% of latency, and `reasoning_effort=low`
  costs 0.692 → 0.631 accuracy for nothing.
- **Published latency excludes rate-limit sleep.** `_pace()` holds this process to 9 requests/minute
  against a free account's 10 RPM quota — 6,667 ms per call, larger than the call. It was inside the
  first published table, where three different configurations all measured within 30 ms of the pacer
  floor.
- **Gold chunk sets are floors, not answer keys.** Recall and grounding are lower bounds. Five
  questions are `unverifiable`: the gold chunks do not contain the fact asked for, so no verdict is
  possible either way. That is the eval set's limit, and it counts against both arms identically.
- **`expected_count` is exact about the graph, not about the world.** Wolfspeed has 38
  `DIRECTOR_OF` edges and a seven-member board — a union over filings against a snapshot on a date.
- **A correctly-cited answer can still be wrong.** *"Allegro MicroSystems Argentina S.A. is
  incorporated in Argentina"* cites a real chunk; the filing says Uruguay. The graph inferred a
  jurisdiction from a company's name. Citation validation is a guarantee about provenance and says
  nothing about truth.
- **Streaming is constrained-decoding only.** The free arm may reject an answer and regenerate it,
  and a claim a user has already read cannot be un-published.
- **45 self-loop edges and ~39 orphan Exhibit 21 shells** remain, from resolution merges that are
  correct more often than not. Fixing them means not stripping national legal forms in `normalize()`.
- **`emb_4096` has no index and never can** — pgvector caps HNSW at 2,000 dims. It is kept for the
  width comparison, never queried in production.
- **HNSW is a demonstration here.** At 2,743 rows an exact scan is ~7 ms.

## Tests

```bash
uv sync --extra dev && uv run pytest      # 53 tests, no services needed
uv run ruff check .
```

Nothing calls Fireworks — embedding scores are injected — and the graph tests skip themselves when
Neo4j is down.

**Six bugs in this project shipped on runs that reported success**, and every one was caught by two
measurements disagreeing rather than by a test failing: a `verify` that passed while discarding 47%
of a relation as self-loops, an eval graded against its own answer key (F1 .986; the real recall was
0.297), a resume that replayed 57 decisions made by a prompt that no longer existed, a benchmark
that timed its own cache and reported a 7 ms p50, a `--no-cache` flag that missed the router, and a
latency table that was measuring the rate limiter. The last four each left a test behind. The full
list is in [docs/phases.md](docs/phases.md#what-broke).

The long-form record lives in [docs/phases.md](docs/phases.md) — what each phase measured and what
it cost — and [docs/decisions.md](docs/decisions.md), which is the ADR log.

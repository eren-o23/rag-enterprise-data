# Knowledge Graph RAG for Enterprise Data

Hybrid knowledge-graph + vector retrieval over SEC filings, running entirely on **open-weight
models** (Fireworks-hosted). Built to answer questions that require two or three hops across
related entities — the class of question where plain vector RAG quietly fails.

> **Benchmark vs. vector-only baseline** — _Phase 5. Accuracy by hop count, latency, and cost
> per query land here, above the architecture diagram._

## Status

| Phase | | |
|---|---|---|
| 1 | Entity + relationship extraction into Neo4j | **complete** — 4,496 entities, 4,685 edges, gate passing |
| 2 | pgvector index over the same chunks | **complete** — 2,743 chunks × 3 widths, recall measured by hop count |
| 3 | Question router (graph vs. vector) | not started |
| 4 | Grounded answer synthesis with validated citations | not started |
| 5 | Benchmark vs. vector-only baseline | not started |

## Phase 1 results

24 semiconductor filers → a queryable graph, for **$1.41** of a $55 budget.

| | |
|---|---|
| Corpus | 2,743 chunks from 4 form types; 2,741 extracted, 2 quarantined |
| Extraction | 14,431 mentions, 7,655 relations kept |
| Dropped by validation | 882 (10.3%), of which 780 failed the evidence-span check |
| Resolution | 5,174 surface forms → 4,496 entities |
| Graph | 4,496 nodes, 4,685 edges, all 14 relation types populated, 0 self-loops |
| `kgrag verify` | **PASS** |

The 780 relations dropped for `evidence_not_found` are the hallucination filter earning its
keep: every relation must carry a verbatim span that survives a substring check against its
source chunk, so a fabricated fact is discarded without a second model call.

### Entity resolution

Measured against 337 labelled pairs, held out per-pair: **precision 1.000, recall 0.297.**

The labels are derived from the filings rather than hand-judged — positives from aliases a
filing declares about itself (`NXP Semiconductors, N.V. ("NXP")`), negatives from Exhibit 21,
which exists to enumerate separate legal entities, so any two rows differ by construction.

Recall is rules-only by design. The rules are deliberately conservative because a false merge
is far more expensive than a missed one: merging a parent with its subsidiary collapses their
`SUBSIDIARY_OF` edge into a self-loop that then gets discarded. That bug cost 317 edges — 47%
of the relation — before it was caught, and the counter reporting it printed on a *passing*
run. Declared aliases supply the remaining recall in production.

### Model bakeoff

Three models over the 20 hand-labelled gold chunks, through the identical
schema-and-validation path that shipped:

| Model | Cost | Wall clock | Mention recall | Relation recall | Failed chunks |
|---|---|---|---|---|---|
| **gpt-oss-120b** (production) | $0.0365 | 249 s | **0.835** | **0.575** | 0/20 |
| gpt-oss-20b | $0.0365 | 1,113 s | 0.767 | 0.400 | 2/20 |
| llama3.2:3b (local, Ollama) | — | 3,580 s | 0.340 | 0.037 | 5/20 |

**Read recall, not precision.** The gold set labels 103 mentions where the same chunks yield
305 from the model — on an Exhibit 21 chunk the labeller stops at 11 rows and the model takes
all 68. Gold is a subset of truth, not a complete answer key, so recall is meaningful while
precision measures how thorough the labelling was, penalising the most thorough model hardest.

Two results worth stating plainly. **The cheaper hosted model is not cheaper**: gpt-oss-20b is
priced at under half per token and cost the same, generating 25% longer responses and burning
retries on schema compliance, while taking 4.5x as long. And **a 3B local model is not viable
for this task on this hardware** — llama3.2 failed 5 of 20 chunks outright and found 3 of 80
relations, in an hour. Constrained decoding guarantees schema-valid output; it guarantees
nothing about whether anything correct is inside it.

### Known limitations

- **45 self-loops still discarded** at load, down from 321. These are names like
  "Applied Materials GmbH" whose legal form `normalize()` strips, leaving them exact-matching
  their parent with no place word left to separate them. ~6% of the subsidiary relation.
- **Orphan rate is 24.8%**, against a `verify` threshold raised from 5% to 30% after measuring
  it. Of 462 orphan companies only ~39 are subsidiary shells that arguably should have an
  edge; the rest are peripheral mentions a closed 14-relation ontology has no edge for —
  former employers in director bios, competitors named in passing. The extractor declining to
  invent a relation for those is correct behaviour, not a gap.
- **The gold set cannot measure precision** (see above), which limits the bakeoff to a
  recall comparison.
- **2 chunks were never extracted**, quarantined after repeated API failures.

## Phase 2 results

2,743 chunks embedded into pgvector at three widths for **$0.57**, joined to the graph on
the chunk ids Phase 1 minted.

### Retrieval recall@k, by hop count

Exact search at 1024 dims, against 52 answerable questions:

| slice | n | R@1 | R@5 | R@10 | R@20 |
|---|---|---|---|---|---|
| 1-hop | 30 | 0.233 | 0.444 | **0.586** | 0.628 |
| 2-hop | 10 | 0.178 | 0.268 | **0.288** | 0.370 |
| 3-hop | 12 | 0.142 | 0.328 | **0.328** | 0.398 |
| all | 52 | 0.202 | 0.384 | 0.469 | 0.525 |

**Multi-hop retrieval runs at roughly half of single-hop.** That gap is the point — it is
the vector-only baseline the Phase 5 benchmark measures the graph against, and a flat curve
here would have meant the question set was too easy to be worth running.

The set is built to be able to lose. Labels come from filing structure rather than
judgement, the same principle as `kgrag mine-pairs`: every edge already carries the
`chunk_ids` whose text justified it, so those chunk ids are the answer key. Multi-hop
questions never name the middle entity, and a chain is discarded unless its evidence spans
more than one filing — a chain described inside a single chunk is a 1-hop question in
costume. 47 questions are mined this way; 10 more are hand-written for the shapes the miner
cannot produce (aggregations, paraphrases, out-of-scope refusals).

Gold sets are floors, not exhaustive — other chunks may also answer a question. The bound
applies identically to every width, which is what keeps the comparison below fair.

### Embedding width: 1024 vs 2000 vs 4096

`qwen3-embedding-8b` emits 4096 dims and pgvector will not HNSW-index above 2,000, so the
production column has to be a truncation. Qwen3 is Matryoshka-trained, which is the usual
justification for truncating — so the whole corpus was embedded at all three widths and
scored, instead of citing it.

| | mean R@10 | vs 1024 | 95% CI |
|---|---|---|---|
| **1024** (production) | **0.469** | — | — |
| 2000 | 0.443 | +0.0266 | [-0.0186, +0.0891] |
| 4096 (native, unindexable) | 0.458 | +0.0112 | [-0.0433, +0.0769] |

Every interval crosses zero on a paired bootstrap, so truncation to a quarter of the native
width costs nothing detectable at n=52. Not "provably identical" — 52 questions cannot
resolve a small real difference — which is why the interval is published, not the point
estimate.

### HNSW recall vs exact search

| ef_search | 1 | 2 | 4 | 10 | 40 | 100 | 400 |
|---|---|---|---|---|---|---|---|
| recall@10 | .079 | .174 | .381 | .900 | .968 | .993 | 1.000 |

At **2,743 chunks an exact scan is already ~7 ms**, so HNSW here is a demonstration of the
technique rather than a necessity — ef=100 gives .993 recall at ~2.6 ms, a real but small
win. Saying so is better than implying otherwise.

Getting this curve to mean anything required forcing *both* planner arms. Left alone,
Postgres costs a sequential scan cheaper than an HNSW probe on a table this small and takes
it even when the index exists, silently answering the "ANN" arm exactly and reporting
recall 1.000 at every `ef_search`. The tell was non-monotonicity — ef=4 scoring above ef=40
— with ANN latencies sitting exactly on the exact baseline.

## Stack

Python 3.12 · Neo4j · pgvector · Fireworks (`gpt-oss-120b`, `qwen3-embedding-8b`) · FastAPI

No LangChain. Fireworks speaks the OpenAI API, so the pipeline is `openai` + `neo4j` +
`pydantic`. The retry, cache, cost-guard, and schema-validation layers are the interesting
part of this project — see [docs/decisions.md](docs/decisions.md).

## Quickstart

```bash
cp .env.example .env        # add your FIREWORKS_API_KEY and SEC_USER_AGENT
docker compose up -d
uv sync

uv run kgrag fetch          # pin filings, pull sections from EDGAR
uv run kgrag chunk          # deterministic chunk ids -- the join key for Phase 2
uv run kgrag extract        # cost-probes first; rerun with --yes to commit
uv run kgrag mine-pairs     # derive resolution labels + declared aliases from the filings
uv run kgrag resolve        # collapse surface forms into canonical entities
uv run kgrag load           # MERGE into Neo4j
uv run kgrag embed          # same chunks into pgvector; --yes to commit the spend
uv run kgrag verify         # the gate: graph, vector store, and the join between them
```

### Backing up the vector store

The 2,743 × 3 embeddings cost $0.57 and live **only** in the Postgres volume — they are
deliberately not written to `cache/` (that would be ~390 MB of JSON duplicating a durable
database row). `docker compose stop` / `up` is safe; `docker compose down -v` deletes the
volume and costs a full re-embed.

```bash
docker compose exec -T postgres pg_dump -U kgrag -d kgrag -Fc > backups/chunks.dump
docker compose exec -T postgres pg_restore -U kgrag -d kgrag --clean < backups/chunks.dump
```

32 MB compressed, ~6 seconds. `backups/` is gitignored.

`kgrag all` runs fetch → embed in order. Run `mine-pairs` before `resolve` either way: it
writes `data/aliases.jsonl`, and without it resolution loses the aliases the filings declare
about themselves. Tuning and comparison live in `kgrag candidates`, `kgrag sweep`, and
`kgrag bakeoff` (see Phase 1 results). `kgrag mine-questions` derives the retrieval eval
set from the graph, and `kgrag recall` scores it.

## Corpus

~25 US semiconductor filers. Four form types, each contributing a different class of edge:

| Form | Sections | Yields |
|---|---|---|
| 10-K | Items 1, 1A, 3, 7 | competitors, suppliers, products, risks, geography |
| 10-K Exhibit 21 | whole | subsidiaries |
| DEF 14A | director bios, officer table | board and officer seats, including *other* boards |
| 8-K | Items 1.01, 2.01, 5.02 | acquisitions and executive changes, with dates |

Exhibit 21 and the proxy bios are what make three-hop questions real: *which companies share a
director with a company that acquired one of NVIDIA's named suppliers?* is a graph traversal,
not a similarity search.

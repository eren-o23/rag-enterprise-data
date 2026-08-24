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
| 3 | Question router (graph vs. vector) | **complete** — 91% routing accuracy, multi-hop recall 2.5x the vector baseline |
| 4 | Grounded answer synthesis with validated citations | **complete** — 0 invented citations, both enforcement arms measured |
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

## Phase 3 results

A router picks the retrieval path per question. **Routing accuracy 54/57 (94.7%)**, and
routed retrieval beats the vector-only baseline at every hop count.

### R@10 by hop count — routed vs. the Phase 2 baseline

| slice | n | vector only | graph only | **routed** |
|---|---|---|---|---|
| 1-hop | 30 | 0.586 | 0.792 | **0.728** |
| 2-hop | 10 | 0.288 | 0.775 | **0.758** |
| 3-hop | 12 | 0.328 | 0.704 | **0.671** |
| all | 52 | 0.469 | 0.768 | **0.721** |

The 2-hop slice is the headline: **0.288 → 0.758, a 2.6x improvement**, on exactly the
questions vector search is structurally unable to answer because the middle entity is never
named. The gap widens with hop count, which is the shape the project exists to demonstrate.

The vector column is not re-derived here — it reproduces Phase 2's 0.586 / 0.288 / 0.328
exactly, and `kgrag route` runs both paths for every question specifically so that it does.
If that column ever stops matching, the harness is broken and nothing else in the table
should be believed.

Note that **graph-only outscores routed** overall (0.768 vs 0.721). Routing is not free
accuracy — it sends some questions to vector that the graph would have answered better. The
honest reading is that the router is currently leaving ~5 points on the table, not that
routing is a win in itself. What routing buys is refusing out-of-scope questions and not
running a traversal for questions that have no graph answer.

### Routing accuracy

| slice | n | correct | accuracy |
|---|---|---|---|
| out-of-scope | 5 | 4 | 0.800 |
| 1-hop mined | 27 | 27 | **1.000** |
| 1-hop hand-written | 3 | 3 | **1.000** |
| multi-hop | 22 | 20 | 0.909 |
| **all** | **57** | **54** | **0.947** |

Expected routes are derived from the eval set rather than hand-assigned, the same discipline
`kgrag mine-questions` uses for gold chunks: `hops == 0` must refuse, `hops >= 2` must reach
the graph, and the hand-written 1-hop paraphrases must reach vector. Mined 1-hop questions
accept either path — they are templated off a graph edge *and* quote the canonical entity
name verbatim, so both paths are legitimately correct and scoring one would be inventing a
ground truth.

### The model never writes Cypher

A relationship type cannot be a Cypher bind parameter, so it has to be interpolated. The
only thing between the model and the query string is `RelationType(...)`, which raises on
anything outside the fourteen-member ontology — the same rule `load.py` already enforces at
write time, applied at read time. The model returns enum members and an entity name:

```
chain [DIRECTOR_OF, ACQUIRED]  ->  -[r0:DIRECTOR_OF]->()-[r1:ACQUIRED]->()
```

That is four traversal shapes over one template instead of ~25 hand-written query strings,
with every relation reachable and every chain up to length 3 expressible.

### What the measurements caught

**The fulltext index built for entity lookup cannot do entity lookup.** Phase 1 created a
fulltext index on `[e.name, e.aliases]` with a comment saying it existed so a question
saying "AMD" could find the canonical node. It returns `AMD Ryzen™ PRO`, and filtered to
companies it returns `AMD (EMEA) LTD.` and `AMD Japan Ltd.` — the right node carries "AMD"
among eleven aliases, and Lucene normalises by field length. An exact index over the same
aliases keyed on `resolve.normalize` resolves every case: 4,629 surfaces, 28 ambiguous,
broken by `mention_count`.

**`gpt-oss-20b` is not viable as the router, and the failure is not slowness.** It stalls
reproducibly on 6 of 57 questions — the same six every rerun, unaffected by a longer
deadline. "What is home automation solutions, and who sells it?" times out at 45s on 20b
and is answered by `gpt-oss-120b` in 1.2s. The tell was not a bill or an error rate: a
failed call records no usage, so a rerun reported **$0.00000** while six questions silently
stayed unrouted. This is Phase 1's `llama3.2:3b` result again — a smaller model that
answers most inputs and hangs on the rest is unusable at any price.

**Describing the corpus as "24 companies" made the router refuse its own contents.** Four
of six routing errors were entities that are *in* the graph but are not filers: a subsidiary
(Picosun Japan), a product, an auditor's officer, a litigation counterparty. The filings
name thousands of such entities. Correcting that one paragraph took 1-hop accuracy from
23/27 to **27/27** and overall from 0.895 to 0.947.

### Known limitations

- **One out-of-scope question is arguably mislabelled, not misrouted.** `h007` ("Who is the
  chief executive of OpenAI?") is gold-labelled out-of-scope because OpenAI is "outside the
  24-filer corpus" — but OpenAI *is* in the graph, a Company node with 5 mentions,
  `PARTNERS_WITH NVIDIA` and two `DIRECTOR_OF` edges. The fact asked for is absent; the
  entity is not. The row is left as-is pending a decision, since the hand-written eval rows
  are not regenerable.
- The two remaining genuine errors are multi-hop questions routed to vector (`h001`, `m032`).
- Refusal happens before retrieval, so it is a judgement about corpus scope made without
  looking at the corpus. Phase 4's citation validation is the real backstop.
- Router cost is **$0.021 per full 57-question sweep** on `gpt-oss-120b`, content-cached, so
  reruns are free.

## Phase 4 results

Both retrieval paths merge into one answer, and **every published citation resolves to a
chunk that was actually retrieved** — 288 citations across two full sweeps, zero invented.

A citation is a chunk id: the same 16-character key minted in `chunk.py`, carried on every
Neo4j edge and used as the pgvector primary key. Validating one is a set-membership test
with no mapping to get wrong, and it is what lets a graph-derived fact and a retrieved
passage cite in the same currency.

### Does constrained decoding beat reject-and-regenerate?

The spec asks to reject invalid citations and regenerate. `ontology.py`'s design says to put
the retrieved ids in the schema as an `enum` so an invalid one is unreachable. Both were
built and run over the same 57 questions.

| | free (validate + regenerate) | constrained (enum) |
|---|---|---|
| answers produced | 45/57 | 45/57 |
| claims / citations | 56 / 133 | 84 / 155 |
| **invented ids published** | **0** | **0** |
| citations needing a delimiter strip | 598 | 0 |
| answers needing a repair round | 2 | 0 |
| out-of-scope refused | 5/5 | 5/5 |
| answerable wrongly refused | 7/52 | 7/52 |
| latency p50 / p95 | 6,545 / 11,213 ms | 6,544 / 10,073 ms |
| $ per answered question | $0.00124 | $0.00116 |

**The constraint is free and deletes the repair loop.** Same answer rate, same refusals,
latency inside noise, marginally cheaper because it never pays for a regeneration. The free
arm is kept because it is the only one that can *measure* invention; the constrained arm is
the API default.

### Graph paths become sentences before they reach the prompt

Raw triples generate awkward text, so each edge is rendered through one phrase per relation
in `ontology.RELATION_PHRASE`, subject-first, with its chunk ids attached:

```
GRAPH FACTS (derived from the knowledge graph)
[2f5b29f93fecbae1] ADVANCED MICRO DEVICES INC is exposed to the risk of export controls...
[234156833fa87f30] TSMC supplies ADVANCED MICRO DEVICES INC
```

That second line is the one that matters. A neighbourhood walk is undirected, so half its
edges arrive against their stored direction — direction is read off the relationship
(`startNode`/`endNode`), never off path order. Read it off the path and the system publishes
"AMD supplies TSMC": confident, well-cited, and exactly backwards, with no error anywhere.

### Refusal moved from a guess to a fact

Phase 3's `refuse` is a judgement about corpus scope made *before* looking at the corpus.
Grounding is the real backstop, and it costs nothing: a refused question short-circuits
before any model call at **$0.00000 and ~2 ms**. All 5 out-of-scope questions are refused,
and 7 of 52 answerable ones are refused for stated reasons ("the provided context does not
contain...").

`h007` — "Who is the chief executive of OpenAI?" — is exactly this case, and the reason its
Phase 3 label was left alone. OpenAI is in the graph; the fact asked for is in no filing.
The system now reaches retrieval and refuses for the true reason instead of guessing.

### Grounding, and what it is not

Share of cited chunks that appear in the question's gold set — 1-hop **0.662**, 2-hop
**0.567**, 3-hop **0.213** (free arm). These are **floors**. Gold sets are the chunks whose
text justified a graph edge, a lower bound rather than an exhaustive answer key, so a cited
chunk outside the set is not necessarily wrong.

Whether the answers are *correct* is deliberately not measured here: that needs a judge, and
a judge is an instrument that has to be validated before it is trusted. Phase 5.

### What the measurements caught

- **Six answers were abandoned for citing their own context correctly.** The context renders
  20 graph facts carrying up to 3 chunk ids each; the validator was checking against the
  route's top-10 `graph_ids`. Answers produced 39 → 45, abandonments 6 → 0, wrong refusals
  13 → 7. The citable set now comes back from the function that prints it.
- **The cost report was timing the cache** — p50 of 7 ms, because 126 of 127 calls came off
  disk. Same bug as the Phase 1 bakeoff. Latency is now measured only over billed questions.
- **~600 "invented" citations were brackets.** The model copies `[c8608131…]` verbatim,
  delimiter included, for ids that really were retrieved. Counting those as invention would
  have made the arm comparison measure formatting rather than grounding.
- **The router's own example question has no path in the graph.** Sundström has eight
  `DIRECTOR_OF` edges and none of those companies has an `ACQUIRED` edge — but he is
  `OFFICER_OF` NXP, which acquired Freescale, and the passages say so. A graph route that
  traverses to nothing now falls back to vector, logged and counted.

### The endpoint

```bash
uv run uvicorn kgrag.api:app
curl -s localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question":"Who is INTEL CORP'"'"'s independent auditor?"}'
```

```json
{"answerable": true,
 "answer": "Intel Corp's independent auditor is Ernst & Young LLP.",
 "route": "graph",
 "citations": ["c8608131724ee274", "ecec386ead8c9bcc"]}
```

Both citations are m001's full gold set. The Neo4j driver and the 4,629-surface entity index
are built once at startup; the psycopg connection is per-request.

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

uv run kgrag route          # route each question, measure the decision
uv run kgrag answer         # grounded answers; --constrained, --no-cache, --fresh
uv run uvicorn kgrag.api:app
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

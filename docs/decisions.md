# Design decisions

Short ADRs. Each one exists because the alternative was tried, considered, or would be
the obvious question in an interview. Ordered by when they were made, not by importance.

---

## Open-weight models via Fireworks, not the Claude API

The project brief specs Claude. Every project in this portfolio runs on open-weight
models instead — the point is to show the same engineering rigor works without a
frontier proprietary API, on a fixed $55 budget. `gpt-oss-120b` for extraction,
`qwen3-embedding-8b` for entity resolution and (Phase 2) chunk embeddings.

**Cost of the decision:** open models are worse at instruction-following edge cases,
so the schema-and-validator layer below carries more weight than it would with Claude.
That's a feature for this portfolio, not a bug — it's the part worth showing.

## No LangChain

Fireworks is OpenAI-API-compatible, so the entire stack is `openai` + `neo4j` +
`pydantic`. The pipeline's interesting engineering — retry, content-addressed caching,
budget enforcement, schema validation — is exactly the layer LangChain would abstract
away. Burying it saves ~150 lines and costs the one part of the codebase worth an
interviewer reading.

## Ontology frozen before any pipeline code

`ontology.py` was the second file written, before fetch/chunk/extract existed. Types
are `StrEnum`s embedded in the JSON schema handed to Fireworks, so constrained decoding
makes an off-ontology entity or relation type structurally unreachable, not merely
discouraged by a prompt. `ALLOWED_EDGES` — a `(subject_type, object_type)` signature per
relation — is the single source of truth: it generates the prompt's ontology block,
validates extraction output, and gates the Cypher write.

**Alternative considered:** open-ended extraction ("extract all entities and
relationships you find"), rejected per the project brief — it produces a graph nobody
can write a query against.

## RiskTopic is a closed vocabulary, not free text

Fifteen fixed risk themes (`supply chain concentration`, `export controls`, ...) rather
than letting the model name risks in its own words. Left open, every filing invents a
new phrasing and every `RiskTopic` node ends up with degree 1 — a graph that looks full
and answers nothing about which companies share which exposures.

## Evidence-span verification as the hallucination filter

Every extracted relation carries a verbatim `evidence` field, checked as a
whitespace-normalized substring of the source chunk (`ontology.validate`, rule 3). A
fabricated relation essentially never comes with a real quotable span, so this catches
hallucination for free — no second model call, no separate fact-checking pass. Dropped
relations are counted by reason (`unknown_endpoint`, `type_signature`,
`evidence_not_found`, `self_loop`, `unknown_risk_topic`) and reported after every run.

## Deterministic chunk ids: `sha256(accession|section_path|ordinal)[:16]`

Not a UUID, not a row number. Phase 2 embeds these same chunks into pgvector and joins
on this key; Phase 4 cites it in answers. A hash of the chunk's identity means the same
document chunked next year — or after an unrelated code change — produces byte-identical
ids, which is what makes `MERGE`-based ingestion actually idempotent instead of merely
re-runnable. Verified by `tests/test_pipeline.py::test_chunk_ids_are_deterministic_across_runs`.

## Cost probe before committing to a full extraction run

`extract.run()` samples 15 chunks, prints `$/chunk` and a projected total for the full
corpus, and requires `--yes` to proceed. This is the specific trap the project brief
warns about — it's easy to learn what a corpus costs only after paying for it. The probe
itself is cached, so continuing after reviewing the number costs only the remainder.

## Entity resolution: three named match rules, not one embedding threshold

`resolve.matches()` returns `"exact"`, `"embedding"`, `"acronym"`, or `"override"` — never
a bare boolean — because each rule fails differently and a portfolio project should be
able to say *why* two names merged, not just *that* they did.

- **Embedding cosine alone is not enough.** "Advanced Micro Devices" and "Advanced
  Energy Industries" embed close together; cosine ≥ τ merges them. The lexical gate
  (token Jaccard ≥ 0.5, required in addition to cosine) blocks it —
  `test_lexical_gate_blocks_the_classic_false_merge`.
- **The lexical gate alone can't catch acronyms.** "AMD" and "Advanced Micro Devices"
  share zero tokens, so Jaccard is 0 regardless of threshold. A dedicated acronym rule
  exists for exactly this case, and blocking (candidate generation) emits an acronym key
  too — without it the pair is never even compared, since prefix blocking alone puts
  "amd" and "advanced..." in different buckets — `test_acronym_merges_where_no_cosine_threshold_could`.
- **A few pairs will never be right by rule.** `overrides.jsonl` holds hand-decided
  merges and splits in both directions. "TSMC" expands to Taiwan Semiconductor
  Manufacturing *Company*, and its C comes from a word `normalize()` deliberately keeps
  as legal boilerplate... except "Company" isn't stripped as a legal suffix (see below),
  so it's actually the *N.V./Inc./Ltd.*-style single-letter-token stripping that misses
  it. Rather than special-case the normalizer further, the pair is a one-line override.
  The override count is itself a quality signal — if it grows past a couple dozen, the
  rules are wrong, not the corpus.

## Normalization strips legal form only, never industry words

The first pass also stripped "technology", "solutions", "semiconductor", etc., because
they read like filler. They aren't, in this corpus: it turned "Taiwan Semiconductor
Manufacturing" into "taiwan manufacturing" and would have merged "ON Semiconductor"
with any other company starting "ON ...". Reverted to legal-suffix-only stripping
(`inc`, `corp`, `llc`, `ltd`, `plc`, `nv`, `sa`, `ag`, `gmbh`, `holdings`, `the`), plus a
trailing-single-letter-token drop to catch `"N.V."` → `"n v"` after punctuation removal.

## Canonical id keys on the cluster's smallest normalized member, not its most frequent

`resolve.canonical_id()` hashes `min(normalized names in the cluster)`. Frequency shifts
every time the corpus grows or the prompt changes what gets extracted; the alphabetic
minimum of a cluster's members does not. An id that moves invalidates every citation
already written against it — a stability property, not an optimization.

## `MERGE`-only writes, relationship type interpolated from a validated enum

Cypher can't parameterize a relationship type — `MERGE (a)-[:$type]->(b)` isn't legal
syntax — so `load.py` groups edges by predicate and interpolates the type into the query
string per group. This is safe *only* because the interpolated value is always a
`RelationType` enum member, checked via `RelationType(predicate).value` immediately
before use, never a raw string from model output or elsewhere. It's the same "never let
the model write Cypher" rule the brief sets for the Phase 3 router, applied one phase
earlier at write time.

## One edge per fact, `chunk_ids` as a list, not one edge per citation

Two chunks stating the same relationship is corroboration, not two separate facts.
`load.py` accumulates `chunk_ids` on the edge via `apoc.coll.toSet`, tracks `support`
(distinct chunk count) as a corroboration-strength signal, and upgrades `confidence` to
`"high"` if any contributing chunk stated it directly. Phase 4 reads `chunk_ids`
straight off the edge to cite the sentences that justified a claim.

---

## Bugs that would have silently corrupted the corpus

Kept here rather than buried in commit history because both looked like something else
at first, and "what looked like a parser bug and wasn't" is a more useful thing to be
able to say out loud than the fix itself.

### `10-K` filing search also matches `10-K/A` amendments

`Company.get_filings(form="10-K")` prefix-matches, so it returned AMD's and Skyworks'
most recent **amendments** as their "latest 10-K." An amendment contains only the items
it amends — AMD's exposed Items 7 and 15 only; Items 1, 1A, and 3 (Business, Risk
Factors, Legal Proceedings) were silently absent, and the code correctly skipped them as
"not present in this filing." Both companies would have entered the graph with no
`EXPOSED_TO` or `SUPPLIES` edges at all, and nothing would have errored. Fixed with
`get_filings(form="10-K", amendments=False)`.

### The first DEF 14A prefilter matched everything

The initial regex looked for any of `director`, `board`, `officer`, `chairman`, etc. —
words that appear on nearly every page of a proxy statement — and kept 2,126 of 2,126
proxy chunks. It compiled, ran, and produced a plausible-looking chunk count, so nothing
about the failure was visible without actually inspecting the ratio of governance
content to compensation-table noise. Replaced with a requirement for an actual statement
of board *service* (`"serves on the board"`, `"director of"`, `"election of directors"`,
...), which drops to 971 chunks and removes the compensation and equity-plan pages that
were diluting the ones that actually name directors' seats at other companies — the
signal Phase 5's three-hop questions depend on.

### `_mentions()` never set `tokens`, so `kgrag resolve` crashed on first real data

`matches()` has always needed `a["tokens"]`/`b["tokens"]` for the Jaccard gate, and
`resolve.sweep()`'s inline test dicts supply it — but `_mentions()`, the function that
builds records for the real `cluster()` path, never did. Every unit test builds its own
`_rec()` fixture with `tokens` already set, and `sweep()` builds its pairs the same way,
so nothing caught it until `kgrag resolve` ran against the actual 2,741-chunk corpus and
raised `KeyError: 'tokens'` on the first candidate pair. Fixed by computing `tokens()`
alongside `norm` in `_mentions()`, the same way `sweep()` already did inline.

### The acronym rule has no cosine floor, and short acronyms collide by coincidence

Running entity resolution on real data merged "AMD" with "Applied Materials Deutschland
Holding GmbH" — after legal-suffix stripping, "Applied Materials Deutschland" reduces to
three tokens whose initials happen to spell "amd", and the acronym rule (deliberately
cosine-independent, see `test_acronym_merges_where_no_cosine_threshold_could`) doesn't
check whether the two names are otherwise related at all. The same pattern hit six
Company-type pairs total (`GF`/"Georg Fischer AG" vs. `GF`/"Global Foundries",
`QTI`/"Qualcomm Technologies International, Ltd." vs. the real QTI, `MCP`/"Meridian
Compensation Partners" vs. the real MCP, `HP`/"Holdback Parties") and several Product-type
2-3 letter acronyms (`MS`, `PC`, `AI`) colliding with unrelated multi-word phrases.

**A cosine floor cannot fix this generally** — plotting cosine against confirmed
true/false labels for every acronym-rule match in the corpus shows true and false pairs
thoroughly interleaved from ~0.48 to ~0.70 (e.g. `MCP`/"Media Content Protection" at 0.508
is true, `AMD`/"Applied Materials Deutschland Holding GmbH" at 0.669 is false — a floor
anywhere between them keeps one wrong). Fixed the confirmed cases via `data/overrides.jsonl`
instead, per the override philosophy above — cheap, pair-specific, and the override count
(16, after this) is still well inside the "couple dozen" tolerance.

**Separately**, the Jaccard gate over-merged some Product-type entities — e.g. "AMD Ryzen™
processors" and "AMD EPYC™ processors" share the tokens "amd" and "processors" out of 3
total, hitting `JACCARD_FLOOR` (0.5) despite naming different product lines; "ryzen" vs.
"epyc" is the one token that actually distinguishes them. Stripping generic descriptor
words ("processors", "series", "solutions", ...) the way `LEGAL_SUFFIXES` strips legal
form would fix this — but those same words are load-bearing in real company names in this
corpus (`Meta Platforms, Inc.`, `Broadridge Financial Solutions, Inc.`), so doing it
globally in `normalize()` risks repeating the exact "industry words survive normalisation"
mistake above, one level up. Patched the two confirmed bridging pairs via overrides;
Product-type resolution quality is weaker than Company-type as a result, and unlike the
Company case this doesn't affect `verify.py`'s Phase 1 gate (its checks are all about the
director/company/subsidiary network, not products) — left as a known limitation rather
than a blocker.

## Resolution merged subsidiaries into parents, and the eval set could not see it

`kgrag load` printed `self_loop_after_resolution: 321` and passed. Those were real edges:
317 `SUBSIDIARY_OF` and 4 `ACQUIRED`, discarded because resolution had merged both
endpoints into one node. Against 364 surviving subsidiary edges that is ~47% of the
relation destroyed — and `SUBSIDIARY_OF` is one of the two relations `THREE_HOP`
traverses, so the capability the whole project exists to demonstrate was running on half
its data. Nothing errored. The counter was the only symptom, and it prints on a passing
run.

**Why the eval set was blind to it.** `kgrag candidates` ranks by `|cosine - TAU|`, which
is the right sampler for locating a decision boundary and the wrong one for auditing
over-merging: a parent and its subsidiary are near-identical strings, so they embed *far
above* TAU and the sampler structurally cannot surface them. The set reported P=1.0 while
the pipeline was merging 1,957 Company pairs wrongly. A clean number from a sampler that
cannot see the failure mode is worse than no number.

**Ground truth from the documents, not from judgement.** `kgrag mine-pairs` derives labels
from structure the filings already encode: positives from the filings' own alias
declarations (`NXP Semiconductors, N.V. ("NXP")` states the alias outright), negatives
from Exhibit 21, which exists to enumerate separate legal entities — so any two rows
differ by construction, as does each row from the filer above it. 337 pairs, 37 positive,
regenerable and idempotent rather than hand-curated.

**What actually discriminates.** Measured over those labels, the tokens separating
`same=false` pairs are overwhelmingly geographic (shanghai, india, ireland, korea) while
those separating `same=true` pairs are industry descriptors (technologies, manufacturing,
solutions). A parent and its local subsidiary differ by a *place*; two spellings of one
company differ by a descriptor. `entity_markers()` encodes exactly that, with the
geography half derived from the corpus's own `Location` mentions so it tracks the corpus,
over a static `GEO_CORE` floor so a fresh clone that has not run `extract` yet still
blocks the merge instead of silently reverting to the old behaviour.

Three changes, each measured separately: the marker block took precision .645 → .905; a
raw acronym taken *before* legal-suffix stripping took recall .514 → .676 (this is what
recovers TSMC, whose C comes from the "Company" the normaliser strips — the case this
document previously settled with a hand override, now handled by rule, along with ESMC,
SMIC, UMC and ADI); `JACCARD_FLOOR` .5 → .67. Result on the real corpus: `SUBSIDIARY_OF`
364 → 684 edges, self-loops 321 → 45.

**The rule that looked good and was wrong.** A fourth rule — merge when one name is the
other's leading token — scored best of all on the eval set, taking recall to .919, because
the eval positives are brand aliases and that is exactly their shape. Run against the real
corpus it merged "Robert A. Feurle" with "Robert A. Schriesheim", and "Power Isolators"
with "Power Products". The eval set was Company-only, so it never scored the `Person` and
`Product` damage. Dropped the rule; the alias declarations it was trying to generalise are
carried as mined data in `data/aliases.jsonl` instead. Documents beat heuristics, and a
metric that only covers one entity type will happily recommend a rule that wrecks the
others.

**Scoring honestly.** Since the eval positives and `aliases.jsonl` come from the same
mining pass, supplying the latter while scoring the former grades the resolver against its
own answer key — it reported F1 .986 that way. `sweep()` now holds each pair's own alias
out while leaving every other pair's available. That gives **P=1.000, R=0.297**: zero false
merges across 300 document-derived negatives, with the rules deliberately conservative and
declared aliases supplying recall in production. The reported recall is rules-only on
purpose, and the residual 45 self-loops are the known remainder — names like "Applied
Materials GmbH" and "Texas Instruments Limited" whose legal form `normalize()` strips,
leaving them exact-matching their parent with no place word left to block on.

## `verify.py`'s 5% orphan-rate gate was a guess; the real corpus runs 25%

`MAX_ORPHAN_RATE = 0.05` was written before `kgrag load`/`verify` ever touched real data,
on the implicit assumption that most extracted mentions end up in some relation. Running
against the real 3,836-entity graph: 960 orphans (25.0%). Sampling 20 orphan `Company`
nodes to check whether this was corruption or expected behavior: only ~8% (39/462 orphan
companies) are Exhibit-21-style subsidiary shells that plausibly should have gotten a
`SUBSIDIARY_OF` edge and didn't — the extraction prompt evidently doesn't always infer a
relation from a bare table row with no per-row sentence. The other ~92% are genuinely
peripheral entities a 7-type/14-relation closed ontology has no edge for at all: former
employers named in director bios ("LSI Logic", "JDS Uniphase"), competitors named in
passing in risk-factor prose ("Oracle", "Google", "LG Electronics"), consultants,
universities, activist shareholders. The extractor correctly declined to invent a
relation for these rather than fabricating one — this is the hallucination filter
working as designed, not extraction failing.

Raised the threshold to 30% (see `verify.py`), grounded in this measurement rather than
picking a number that merely clears the observed rate — high enough that a real
extraction regression (e.g. a broken prompt that stops finding relations at all) still
trips it, not so high that it stops being a gate. The Exhibit-21 subsidiary-shell gap
(~39 companies) is real but small and would require touching the extraction prompt and
re-running against real chunks to fix — left as a known limitation rather than pursued
for Phase 1.

## Phase 2: three embedding widths, because "truncation is principled" is a claim, not a result

`qwen3-embedding-8b` returns 4096 dimensions and pgvector will not build an HNSW index
above 2,000 (`vector`) or 4,000 (`halfvec`). So the production column has to be a
truncation — there is no version of this where the native width ships. Qwen3 is
Matryoshka-trained, which is the standard justification for truncating, and it would have
been easy to cite that and embed at 1024 without checking.

Embedded the whole corpus at 1024, 2000, and 4096 instead ($0.57, 129 API calls) and scored
all three on the same question set. Mean R@10: **1024 = .469, 2000 = .443, 4096 = .458.**
A paired bootstrap over per-question differences (10,000 resamples, fixed seed, paired
because every width answers the same questions) puts every pairwise interval across zero:

| | difference | 95% CI | |
|---|---|---|---|
| 1024 vs 2000 | +0.0266 | [-0.0186, +0.0891] | within noise |
| 1024 vs 4096 | +0.0112 | [-0.0433, +0.0769] | within noise |
| 2000 vs 4096 | -0.0154 | [-0.0481, +0.0077] | within noise |

So truncation to a quarter of the native width costs nothing detectable at n=52, and 1024
ships. Worth noting what this does *not* say: 52 questions cannot resolve a small real
difference, so this is "no measurable difference", not "provably identical". The interval
is published rather than the point estimate for that reason.

The handoff's version of this experiment proposed embedding "the 20 gold chunks at 4096".
That cannot work — ranking needs the entire corpus present at each width or there is
nothing for the gold chunks to be ranked against.

## The embedding cache key had to change in the same edit as the `dimensions` parameter

`fireworks.embed()` keyed its cache on `sha256(f"{model}|{text}")`. Adding a `dimensions`
parameter without touching that key would have made the cache return 4096-dim vectors for
1024-dim requests — several thousand entity-name vectors were already cached at native
width from `kgrag resolve`, so the collision was not hypothetical. A dimension mismatch
surfaces far downstream, if at all.

The dimension is suffixed only when one is requested, so `dimensions=None` hashes exactly
as before and resolve's existing cache stays valid rather than being invalidated for
nothing. There is a test asserting both halves of that.

## At 2,743 rows the planner silently answers "ANN" queries exactly

The first ANN-vs-exact sweep produced a non-monotonic curve: `ef_search=4` scored recall
1.000 while `ef_search=40` scored 0.968. HNSW search is deterministic, so higher `ef` can
never retrieve less — the curve was measuring something other than what it claimed.

The latency column gave it away. Every 1.000 row ran at ~7 ms, exactly the full-scan
baseline, while the sub-1.0 rows ran at ~2 ms. On a 2,743-row table Postgres costs a
sequential scan cheaper than an HNSW probe and takes it *even when the index exists*, so
roughly half the "ANN" measurements were second exact scans agreeing perfectly with the
first. Disabling index scans for the exact arm was not enough; the ANN arm has to have
sequential scans disabled too. Confirmed with `EXPLAIN` that both plans now use the
intended access path.

Forced, the curve is monotonic and the sweep says something:

| ef_search | 1 | 2 | 4 | 10 | 40 | 100 | 400 |
|---|---|---|---|---|---|---|---|
| recall@10 (1024) | .079 | .174 | .381 | .900 | .968 | .993 | 1.000 |

`ef_search` is swept down to 1 because the conventional range does not bend at this scale.
Two honest caveats belong with this table. An exact scan over 2,743 chunks is ~7 ms, so
HNSW here is a demonstration of the technique rather than a necessity — at ef=100 it is
~2.6 ms for .993 recall, a real but small win. And the emb_2000 index reaches full recall
almost immediately while showing no latency benefit at all over its own exact baseline
(10.3 ms vs 10.9 ms), which is a second, independent reason 1024 is the right production
column.

## The retrieval eval set is derived from the graph, and built to be able to lose

Phase 1's resolution eval reported P=1.000 while the pipeline was merging ~1,957 pairs
wrongly, purely because the sampler could not surface the failing shape. An eval of
single-fact lookups would repeat that mistake exactly: it would report good recall and say
nothing about the multi-hop questions the graph exists to win.

Labels come from filing structure, the same principle as `kgrag mine-pairs`. Every edge
already carries the `chunk_ids` whose text justified it, and the evidence-span check means
those chunks demonstrably contain the supporting sentence — so the chunk ids *are* the gold
set, with no hand-judgement and no model call. 47 mined questions across all 14 relation
types and 1/2/3 hops, plus 10 hand-written cases the miner structurally cannot produce
(aggregations answered by no single chunk, paraphrases sharing almost no surface tokens
with their target, and out-of-scope questions that should retrieve nothing).

Three decisions that keep the numbers meaningful:

- **Multi-hop questions never name the middle entity.** Naming it makes the question two
  lookups; withholding it is what a single query embedding has no way to bridge.
- **A chain is discarded unless its gold chunks span more than one filing.** A chain that
  happens to be described inside one chunk is a 1-hop question in costume, and keeping it
  would have quietly inflated multi-hop recall.
- **Questions are deduplicated by text with their gold sets merged.** Two edges can
  template to the same sentence — "Which company did Teradyne acquire?" when Teradyne
  acquired three — and emitting it twice with different answer keys makes it unanswerable:
  retrieval finding one acquisition gets marked wrong for the other.

Templated phrasing is the known ceiling. It is deliberate: the templates hand vector search
canonical entity names verbatim, which *helps* it. Multi-hop recall collapsing under
conditions that favourable is a stronger result than it would be with natural phrasing.

Result at 1024, exact search: **1-hop .586 @10, 2-hop .288, 3-hop .328.** Multi-hop at
roughly half of single-hop, which is the Phase 5 baseline being established rather than a
Phase 2 failure — a flat curve here would have meant the eval set was too easy to bother
running. The gold sets are floors, not exhaustive: other chunks may also answer a question,
and that bound applies identically to all three widths, which is what keeps the comparison
between them fair.

## The recall regression check is a catastrophe floor, and it does not live in `verify`

The Phase 2 plan called for `kgrag verify` to fail below a measured 1-hop recall floor,
the same discipline `MAX_ORPHAN_RATE` got. Two things were wrong with that as specified.

**It would have broken the gate.** `kgrag verify` needs only the two databases — no API
key, no network, no spend. Scoring recall means embedding 57 questions, and `cache/` is
gitignored, so on a fresh clone the gate would make 57 paid Fireworks calls. `kgrag recall`
already pays that cost, so the check is free there and expensive in `verify`.

**A quality threshold is not supportable at this sample size.** The 1-hop slice is 30
questions, and the paired bootstrap over the larger 52-question set already showed
differences of ~0.03 failing to survive resampling. Any threshold tight enough to detect
genuine quality drift would flap on noise.

So the floor is deliberately loose: **0.35 against a measured 0.586**. It does not claim to
detect drift. It detects the failures that collapse rather than drift — querying the wrong
embedding column, vectors written against the wrong `chunk_ids`, or the eval set decoupling
from the corpus after a re-chunk. Confirmed by simulating exactly that: shuffling which
answer key belongs to which question takes 1-hop R@10 from **.586 to .033**, three
resampling-noise-widths clear of the floor in the direction that matters.

That check is itself tested (`test_recall_floor_fires_on_collapse_but_not_on_noise`),
because a gate that only ever passes is precisely the failure this project already made
once: Phase 1's resolution eval reported P=1.000 while the pipeline merged ~1,957 pairs
wrongly. Asserting a gate can fail is cheaper than discovering it never could.


---

## Phase 3: the fulltext index built for entity lookup cannot do entity lookup

`cypher/schema.cypher` created a fulltext index on `[e.name, e.aliases]` in Phase 1, with a
comment saying it exists so Phase 3 can map a name in a question onto a node — "a question
says 'AMD' and the canonical node is 'Advanced Micro Devices, Inc.'". That was written
before there was a graph to try it against. Tried against the real one, it fails on the
exact example it was written for:

```
'AMD'                     -> AMD Ryzen(TM) PRO (Product) 3.44, AMD Radeon(TM) graphics 3.41
'AMD' + node.type=Company -> AMD (EMEA) LTD. 1.87, AMD Japan Ltd. 1.87, AMD Design, LLC 1.87
```

`ADVANCED MICRO DEVICES INC` **does** carry `"AMD"` in its alias list. It also carries ten
other aliases, and Lucene normalises by field length: a match inside a long `aliases` array
scores below a match against a node whose entire name is "AMD Japan Ltd.". The right answer
is not merely ranked low, it is absent from the top five either way. Filtering to
`type = 'Company'` makes it worse, because it removes the product noise and leaves nothing
but the small subsidiaries.

The replacement is a dict, not a better query: normalize every name and alias with
`resolve.normalize` — already written, already tuned, already stripping legal suffixes —
and look up the question's surface form directly. On the real corpus that is 4,629
normalized surfaces over 4,496 entities, of which **28 are ambiguous**, and they are almost
all Location/Company collisions (`intel`, `arm`, `china`, `nasdaq`, `european union`).
`mention_count`, already stored on every entity, breaks those.

The fulltext index stays, demoted to the fallback for surfaces the model paraphrases into
something no alias ever said. It also needs its input scrubbed of Lucene syntax characters
first: `Picosun Japan Co., Ltd.` and any name containing a bracket are a query parse error,
not a miss, which is a silent difference.

The general shape of this is the same mistake as Phase 1's resolution eval: an artifact
built ahead of the data it would run on, believed because it looked right. The only reason
it was caught before the router was written is that the graph was queried first.

## One traversal family with validated predicates, not a template library per query type

The spec says to "keep a template library keyed by query type and let the model fill
parameters only". Read literally that is roughly 25 hand-written Cypher strings —
`acquisitions_by`, `auditor_of`, `shared_director` — and every question whose shape is
missing becomes unanswerable.

The security property the spec is actually protecting is *the model never authors a query*.
That is satisfied by something much smaller. `load.py` already established the pattern at
write time: a relationship type cannot be a Cypher bind parameter, so it is interpolated —
but only after passing through `RelationType(...)`, which raises on anything outside the
fourteen-member enum. Read time uses the identical rule.

So the library is four shapes (`one_hop`, `two_hop`, `three_hop`, `neighbourhood`) over one
template, and the model returns a *chain of enum members*:

```
chain [DIRECTOR_OF, ACQUIRED]  ->  -[r0:DIRECTOR_OF]->()-[r1:ACQUIRED]->()
```

Thirty lines, every one of the 14 relations reachable, every chain up to length 3
expressible, and the interpolated substring can only ever be one of fourteen literals.
`test_traversal_rejects_a_predicate_the_model_invented` asserts that directly, including on
strings shaped like injection attempts. This also mirrors how `questions.py` mines the eval
set in the first place — as predicate chains — so the router is answering questions in the
same vocabulary they were generated in.

The shape field is redundant with the chain length, deliberately: the model declaring both
means a disagreement between them is detectable. That is layer 2's `chain_shape_mismatch`,
and it degrades to running both paths rather than guessing which of the two fields to
believe.

## The router refuses before retrieving, which is a guess, and it is measured as one

Routing to `refuse` is a judgement about what the corpus contains, made *before* looking at
the corpus. That is strictly weaker than refusing because retrieval came back empty or
because no citation validated — which is Phase 4's job and the real backstop.

It is in Phase 3 anyway for one reason: `eval/questions.jsonl` already ships 5 hand-written
out-of-scope questions with empty gold sets, so a `refuse` enum member costs one line and
turns them from unscorable into a measurement. `h009` — "What is Intel's revenue forecast
for 2031?" — is the case that decides whether it was worth it. It names a corpus company,
asks for a figure of a kind filings routinely contain, and is specifically not disclosed.
A router that refuses `h005` ("capital of France") and accepts `h009` has learned nothing
except keyword matching.

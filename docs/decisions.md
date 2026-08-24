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

## The router runs on gpt-oss-120b, because gpt-oss-20b hangs rather than errs

The spec asks for "a cheap router: a small model call", and `gpt-oss-20b` is the obvious
pick — half the input price of the extraction model, healthy measured latency of 3-7s,
already priced in `fireworks.PRICES`. It is not usable, and the reason is worth recording
because it is invisible in every metric you would normally look at.

Six of the 57 eval questions stall it. Not "sometimes" — the same six, on every rerun,
unaffected by raising the deadline:

```
'What is home automation solutions, and who sells it?'
  gpt-oss-20b   45.4s  APITimeoutError
  gpt-oss-120b   1.2s  OK
```

The tell was neither a bill nor an error rate. A call that times out records no usage, so
the meter reads `$0.00000` and the run "succeeds" — six questions simply carry a fallback
route instead of a decision, and (before this was fixed) their zero retrieval scores were
charged to the graph column. A cheap model that answers most inputs well and hangs on the
rest is unusable at any price, because the failures are deterministic and survive retry.

This is Phase 1's `llama3.2:3b` result in a different costume: that model was rejected for
5 hard failures out of 20, not for a quality gap. Same conclusion, found the same way — by
looking at what did not come back rather than at the average of what did.

At $0.021 per 57-question sweep the 120b router is not expensive enough for the trade to be
interesting.

## The router refused its own corpus, because the prompt described the wrong thing

The system prompt opened by naming the corpus: "10-K, DEF 14A and 8-K filings from 24
US-listed semiconductor companies (AMD, Intel, NVIDIA, ...)". That sentence is true and it
was the single largest source of routing error.

Four of six errors were `refuse` on questions the corpus answers perfectly well:

| question | why it was refused | what it actually is |
|---|---|---|
| Where is Picosun Japan Co., Ltd. incorporated? | not one of the 24 | a subsidiary, listed in Exhibit 21 |
| What is home automation solutions, and who sells it? | not one of the 24 | a product |
| What executive role does Ms. Simon hold at Deloitte & Touche LLP? | not one of the 24 | an auditor's officer |
| What legal proceeding is EireOg Innovations Ltd. a party to? | not one of the 24 | a litigation counterparty |

The 24 are the *filers*. The graph holds **4,496 entities**, because that is the entire
point of extracting one. Naming only the filers taught the router that 99.5% of the corpus
was out of scope.

Correcting that paragraph — the 24 are filers, their filings name subsidiaries, directors,
officers, auditors, customers, suppliers, competitors, products, regulators and litigation
counterparties, and all of those are in scope — moved 1-hop routing accuracy from 23/27 to
**27/27** and overall from 0.895 to **0.947**.

This is prompt correction, not eval-tuning: the original sentence made a false claim about
the data. The guard against tuning is the out-of-scope slice, which is scored in the same
table and would have caught an over-correction. It caught one, and it is documented below.

## `h007` is probably a mislabelled gold row, and it is being left alone

Widening the corpus description cost one out-of-scope refusal: `h007`, "Who is the chief
executive of OpenAI?", now routes to `vector` instead of refusing. Its hand-written note
gives the reason for the label as "Company outside the 24-filer corpus".

That reason is false. OpenAI is in the graph — a `Company` node with 5 mentions, a
`PARTNERS_WITH` edge to NVIDIA, and two `DIRECTOR_OF` edges from a proxy bio. What is
absent is the *fact* asked for: no filing states who runs OpenAI. So the question is a
"fact not in the corpus" case wearing an "entity not in the corpus" label, and routing it
to retrieval so that grounding can fail honestly is defensible — arguably more defensible
than refusing on a premise that is not true.

The row is **not** being edited. The ten hand-written rows in `eval/questions.jsonl` are
not regenerable and sit alongside `data/overrides.jsonl` in the don't-touch list, and
quietly relabelling an eval row that disagrees with the system is how an eval stops being
able to fail. It is recorded here for a decision instead, and the published 0.800 keeps
counting it as an error in the meantime.

## The routing eval resumes, and refuses to resume the two things it must not

Every other stage of this pipeline resumes: `extract` skips chunk ids already in
`extractions.jsonl`, `embed` resumes from `WHERE emb_N IS NULL`. The routing eval restarted
from zero, which on a 9 RPM budget meant a stall past question 10 threw away forty minutes
of paced calls — and it did, repeatedly, before an uncaught `APITimeoutError` was quarantined
the way `extract_one` already quarantines them.

Two things are deliberately *not* resumable, both learned by being bitten:

**A router timeout is not a decision.** It records a fallback to `both` because the call
never returned. Resuming over one freezes a transient stall into the published numbers, and
`chat_json` only caches successes, so retrying is free of stale answers. Retrying the ten
frozen rows moved 2-hop graph recall from **0.433 to 0.675** — the fallbacks were not noise,
they were suppressing the result.

**A decision made by a different prompt is not a decision about this router.** Editing the
corpus description and rerunning reported byte-identical numbers and `$0.00000` spend,
because all 57 rows resumed and not one router call was made. `chat_json` already puts the
prompt in its cache key so an edit invalidates exactly the right entries; the resume now
carries the same idea as `router_sha`, a fingerprint of the prompt and schema that every
logged row stores and `_prior` requires to match. `--fresh` remains for changes it cannot
see, such as traversal or ranking code.

## Phase 4: citations are chunk ids, because that makes validation set membership

The spec asks for "a citation per claim" and for every citation to "resolve to a chunk id
that was actually retrieved". The tempting design is footnote numbers, or indices into the
passage list, with a mapping back to chunks. Every one of those adds a translation step
between what the model writes and what gets checked, and a citation system is only worth
having if the check cannot be fooled.

So the citation token IS the chunk id — the same 16-character key minted in `chunk.py`,
carried in every Neo4j edge's `chunk_ids`, and used as the primary key of the pgvector
`chunks` table since Phase 2. Validating a citation is then one set-membership test against
what retrieval returned, with no mapping to get wrong, and a graph-derived fact and a
retrieved passage cite in the same currency — which is what lets one answer mix them at
all.

## Two citation-enforcement arms, because "validated citations" is a claim

`ontology.py` rests on the idea that constrained decoding makes an off-vocabulary value
unreachable rather than merely rejected. Applied to citations, that says: put the retrieved
chunk ids in the JSON schema as an `enum` and an invented citation cannot be generated.

But the spec explicitly asks to "reject and regenerate", and a system where invention is
impossible has no invented-citation rate to report — it cannot answer the question of what
the constraint is worth. So both were built and measured over the same 57 questions:

| | free (validate + regenerate) | constrained (enum) |
|---|---|---|
| answers produced | 45/57 | 45/57 |
| claims / citations | 56 / 133 | 84 / 155 |
| citations needing a delimiter strip | 598 | 0 |
| answers needing a repair round | 2 | 0 |
| abandoned after repairs | 0 | 0 |
| invented ids in published answers | 0 | 0 |
| out-of-scope refused | 5/5 | 5/5 |
| answerable wrongly refused | 7/52 | 7/52 |
| latency p50 / p95 | 6,545 / 11,213 ms | 6,544 / 10,073 ms |
| $ per answered question | $0.00124 | $0.00116 |

The constraint costs nothing measurable — same answer rate, same refusals, same latency
inside noise, marginally cheaper because it never pays for a repair round — and it deletes
the entire repair mechanism. That is the result. The free arm is kept because it is the
only arm that can *measure* invention; the constrained arm is the default for the API.

Both arms publish zero invented citations, which is the claim the spec actually makes. They
arrive there differently: one by construction, one by catching two answers and regenerating
them.

## A citation copied with its brackets is a delimiter slip, not an invented source

The context prints ids as `[c8608131724ee274]`, and gpt-oss-120b cites `[c8608131724ee274]`
— brackets included — about 600 times over a 57-question sweep, for ids that really were
retrieved. Three repair rounds do not talk it out of the habit.

Treating that as invention is wrong twice. It overstates the invented rate by roughly two
orders of magnitude, and it makes the free-vs-constrained comparison measure formatting
instead of grounding: the constrained arm cannot emit a bracket at all, so it would "win"
on a typographic technicality rather than on whether the model made up a source. The strip
is deliberately narrow — surrounding brackets and whitespace only — so a fabricated or
truncated id still fails, and the count is reported rather than hidden.

## The citable set has to be exactly what the context printed

Six answers were abandoned as `citation_unrecoverable` for citing their own context
correctly.

`build_context` renders up to 20 graph facts, each showing up to three of the chunk ids
that justified its edge — as many as 60 ids. The caller was computing the citable set from
the route's `graph_ids`, which `chunk_ids_of` caps at k=10. So the model read an id off the
page, cited it, and the validator called it invented. The measurement was wrong, not the
model: fixing it moved answers produced from 39 to 45, abandonments from 6 to 0, and
answerable questions wrongly refused from 13 to 7.

The set now comes back from `build_context` itself, alongside the text. What is printed and
what is citable are produced by one function and cannot drift.

## A repair prompt must differ from the prompt it repairs

`chat_json` caches on `(model, PROMPT_VERSION, system, user, schema)`. A regeneration that
resends a byte-identical prompt is therefore served the *same invalid answer* off disk,
forever, at $0.00 — indistinguishable in the logs from a model that refuses to correct
itself. The repair message names the rejected ids, which changes the key and makes the
retry an actual retry.

This is the third appearance of one bug in this project: the Phase 1 bakeoff measured cache
hits for the incumbent model, and the Phase 3 resume replayed decisions made by a prompt
that no longer existed. Content-addressed caching is the right design and it fails the same
way every time — when identity of *inputs* is mistaken for identity of *intent*.

## The cost report was timing the cache, again

A rerun of the answer sweep reported a p50 latency of 7 ms and $0.00003 per question,
because 126 of 127 calls came off disk. Those numbers describe the filesystem.

Latency and cost are now measured only over questions that actually billed, short-circuited
refusals are excluded and counted separately (genuinely free and genuinely instant, but
averaging them in understates what an answer costs), and the report says out loud when most
of the sample was cached. `--no-cache` forces a real measurement, exactly as `kgrag bakeoff`
already does and for the same reason.

## A graph route whose traversal finds nothing falls back to vector, and says so

The router's own prompt uses "Which company was acquired by a company that Karl-Henrik
Sundstrom is a director of?" as its two-hop example. There is no such path in this graph:
Sundström has eight `DIRECTOR_OF` edges and none of those companies carries an `ACQUIRED`
edge. He is, however, `OFFICER_OF` NXP, and the filings state that NXP acquired Freescale —
so the passages answer a question the graph cannot.

Phase 3 correctly does not run the path the router rejected; that is the cost routing
exists to avoid. But at answer time an empty graph result is not a decision, it is an empty
context, and one cached embedding call is cheaper than a refusal. The fallback is logged as
`graph_empty_fell_back_to_vector` and counted in the report — a silent version would make
the Phase 3 routing numbers describe something other than what production does. It fired on
0 of the 57 eval questions and turned that example question from an abandoned answer into a
correct one.

## Answer correctness is not graded in Phase 4

Phase 4 reports citation integrity, refusal behaviour, groundedness against the gold chunk
sets, latency and cost. It does not report whether the answers are *right*, because that
needs a judge, and a judge is a measurement instrument that itself has to be validated.

The groundedness numbers (1-hop 0.662, 2-hop 0.567, 3-hop 0.213 in the free arm) are the
share of cited chunks that appear in a question's gold set, and they are **floors**: gold
sets are the chunks whose text justified a graph edge, a lower bound rather than an
exhaustive answer key, so a cited chunk outside the set is not necessarily wrong. The 3-hop
figure in particular says more about gold-set sparsity at three hops than about the answers.
The accuracy-by-hop-count table the README opens with is Phase 5's job.

## Aggregation needs a different retrieval shape, because you cannot count from a sample

`h000` — "How many subsidiaries does AMD list, and in how many countries?" — failed in
Phase 4 while the graph held the answer exactly: 32 `SUBSIDIARY_OF` edges, sitting right
there. The reason is structural, not a tuning problem. `graph_path` returns the ten
best-corroborated chunk ids around AMD, ranked by `support`; those are risk-topic and
auditor edges, and the 32 subsidiary edges never make the cut. No value of k fixes this,
because a count is a property of the *whole* neighbourhood and any top-k is a sample.

So counts are computed by Cypher over the entire neighbourhood, with no cap, grouped by
predicate **and direction**, and handed to the model as stated facts. Two reasons for the
split:

- **Cypher counts exactly and a language model does not.** Handing over 32 names and asking
  for a total is a worse instrument than a `count(DISTINCT o)`.
- **Direction is the meaning.** 32 companies pointing `SUBSIDIARY_OF` at AMD means AMD has
  32 subsidiaries; AMD pointing at 32 companies would mean AMD is owned by all of them.

It runs only for a neighbourhood shape — an empty chain is the router saying "no fixed
traversal fits", which is the signal it already emits for counting and summarising. A
1-hop or 2-hop question named the traversal it wants, and burying that fact under
corpus-wide statistics would make it harder to answer, not easier.

Result: **8/8 exact counts correct**, grounding precision **1.000** on the slice.

## The near-miss count: correctly cited, and wrong

The first version of this answered "AMD operates in 11 countries."

AMD's own `OPERATES_IN` has 11 endpoints. AMD's subsidiaries' `INCORPORATED_IN` has 11
distinct locations. Two unrelated sets that happen to share a size — and *neither* is a
count of countries, because `Location` in this ontology spans streets, cities, states and
regions: AMD's eleven include "United States", "U.S." (both), "Santa Clara, California",
"2485 Augustine Drive, Santa Clara, California 95054", "Europe" and "Asia".

**Citation validation cannot catch this.** The citation resolved perfectly. The claim was
grounded in a real retrieved chunk. It was simply an answer to a different question, and it
would have shipped as a success — the most dangerous failure shape in the project so far,
because every mechanism built to catch invented sources reports green.

Two changes. The synthesis prompt now forbids substituting a near-miss count: a count of a
different relation, or of a coarser or finer thing than the question names, is not an
answer, and "distinct Locations" is explicitly not a count of countries. And the answer
schema supports **partial answers** — a compound question can have one half supported and
the other not, and discarding a correct fact because a different part is unsupported throws
away a real answer. `h000` now returns the exact 32 and names the country count as
unsupported, which is the truthful shape of that question.

Counting countries remains unsupported, and honestly so: it needs both a chained aggregate
and a `Location` type that distinguishes a country from a street address. Neither is worth
inventing to make one eval row go green.

## The aggregation stratum is where R@10 stops meaning anything

| slice | n | vector | graph | routed |
|---|---|---|---|---|
| 1-hop | 30 | 0.586 | 0.792 | 0.728 |
| 2-hop | 10 | 0.288 | 0.775 | 0.758 |
| 3-hop | 12 | 0.328 | 0.704 | 0.671 |
| **aggregation** | **8** | **0.274** | **0.088** | **0.088** |

The graph scores 0.088 on the slice it answers 8/8 correctly, and loses to vector on it.
Both facts are true and neither is a contradiction: the answer to "how many subsidiaries"
is a computed total, not a passage, so recall against gold chunks is measuring the wrong
object. A gold set of 45 chunks cannot be recalled at k=10 no matter how good retrieval is.

This matters for Phase 5, which is built on recall-as-proxy: **that assumption does not
extend to aggregation**, and the headline table must score this slice on answer
correctness instead. Fortunately that needs no judge. The count came from Cypher, so
`expected_count` in each eval row is an exact answer key derived from structure — the same
discipline `questions.py` uses for gold chunk sets, and the only judgement-free accuracy
number in the project.

## The eval set gains an aggregation stratum, hand-phrased and structurally labelled

The spec asks for a set "stratified by difficulty: single hop, two hop, three hop,
aggregation, and out-of-scope". Four of the five existed. The set is now 65 questions with
all five, and routing accuracy is **62/65 (95.4%)** with all 8 aggregation questions routed
to the graph.

The questions are phrased by hand and marked `"source": "hand"` so `mine-questions`
preserves them, but their labels are derived from the graph exactly like the mined rows:
`gold_chunk_ids` is the union of chunk ids on the counted edges, and `expected_count` is
the count itself. No hand-judgement enters the answer key.

They carry `"category": "aggregation"` and are deliberately **excluded from the hop
slices**. An aggregation question needs one hop, but it is not a 1-hop question — folding
it in would move the published Phase 2 recall figures and shift `MIN_1HOP_RECALL_AT_10`'s
baseline underneath a floor that was sized against a specific population.

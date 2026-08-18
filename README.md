# Knowledge Graph RAG for Enterprise Data

Hybrid knowledge-graph + vector retrieval over SEC filings, running entirely on **open-weight
models** (Fireworks-hosted). Built to answer questions that require two or three hops across
related entities — the class of question where plain vector RAG quietly fails.

> **Benchmark vs. vector-only baseline** — _Phase 5. Accuracy by hop count, latency, and cost
> per query land here, above the architecture diagram._

## Status

| Phase | | |
|---|---|---|
| 1 | Entity + relationship extraction into Neo4j | in progress |
| 2 | pgvector index over the same chunks | not started |
| 3 | Question router (graph vs. vector) | not started |
| 4 | Grounded answer synthesis with validated citations | not started |
| 5 | Benchmark vs. vector-only baseline | not started |

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
uv run kgrag all --budget 5.00
```

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

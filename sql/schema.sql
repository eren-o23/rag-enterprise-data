-- Run before the first embed. Idempotent, applied by embed.apply_sql().
--
-- The Postgres half of the store. Neo4j holds the entities and edges; this holds the
-- passages those edges were extracted from. `chunk_id` is the join key between them --
-- the same value that appears in every Neo4j edge's `chunk_ids` list.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    ticker       TEXT NOT NULL,
    company      TEXT NOT NULL,
    form         TEXT NOT NULL,
    accession    TEXT NOT NULL,
    section_path TEXT NOT NULL,
    filing_date  DATE NOT NULL,
    text         TEXT NOT NULL,

    -- Canonical entity ids this chunk mentions, from resolve.py. Denormalised on purpose:
    -- it lets a vector search be filtered to an entity's neighbourhood without a round
    -- trip to Neo4j. Written by a statement separate from the embedding UPDATEs, so a
    -- resolution change refreshes these for $0.00 rather than forcing a re-embed.
    entity_ids   TEXT[] NOT NULL DEFAULT '{}',

    -- Three widths, measured against each other before one is chosen (see `kgrag recall`).
    -- pgvector caps HNSW index dimensions at 2000 for `vector` (4000 for `halfvec`), so
    -- emb_4096 -- the model's native output -- is storable but exact-scan only. It is the
    -- reference ceiling the truncated widths are scored against, never the shipped column.
    emb_1024     vector(1024),
    emb_2000     vector(2000),
    emb_4096     vector(4096)
);

-- Phase 3 filters a vector search by entity ("what does NVIDIA say about supply risk")
-- and by filer. GIN is the right index for the array containment operator.
CREATE INDEX IF NOT EXISTS chunks_entity_ids ON chunks USING GIN (entity_ids);
CREATE INDEX IF NOT EXISTS chunks_ticker ON chunks (ticker);

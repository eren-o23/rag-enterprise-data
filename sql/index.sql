-- Applied AFTER the bulk load, not with schema.sql: pgvector builds an HNSW graph
-- substantially faster over a populated table than it maintains one across 2,743 inserts.
--
-- Cosine, because qwen3 embeddings are not length-normalised and only direction is
-- meaningful. Only the two indexable widths appear here -- emb_4096 exceeds pgvector's
-- 2000-dimension index cap and can only ever be scanned exactly.

CREATE INDEX IF NOT EXISTS chunks_emb_1024_hnsw ON chunks USING hnsw (emb_1024 vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_emb_2000_hnsw ON chunks USING hnsw (emb_2000 vector_cosine_ops);

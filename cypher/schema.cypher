// Run once, before the first load. Idempotent.

// Every node carries :Entity plus its type label (:Entity:Company). :Entity anchors the
// uniqueness constraint and the fulltext index; the type label keeps queries readable
// and lets Neo4j use a label scan instead of a property filter.
CREATE CONSTRAINT entity_id IF NOT EXISTS
FOR (e:Entity) REQUIRE e.id IS UNIQUE;

// Phase 3 maps entities named in a question onto node ids. Aliases are indexed too,
// because a question says "AMD" and the canonical node is "Advanced Micro Devices, Inc."
CREATE FULLTEXT INDEX entity_search IF NOT EXISTS
FOR (e:Entity) ON EACH [e.name, e.aliases];

CREATE INDEX entity_type IF NOT EXISTS
FOR (e:Entity) ON (e.type);

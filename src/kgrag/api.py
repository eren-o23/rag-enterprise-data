"""`uvicorn kgrag.api:app` — the endpoint the spec's "Done when" asks for.

A thin shell over `answer.answer`. All the substance -- routing, traversal, retrieval,
citation validation -- already happened by the time a response is built, and none of it
belongs here.

The lifecycle is the only real decision. The Neo4j driver and the entity index are built
once at startup: the index inverts 4,496 entities into 4,629 normalized surfaces, and
rebuilding that per request would dominate the response time it is meant to serve. The
Postgres connection is opened per request instead, because a psycopg connection is not
safe to share across concurrent requests and connecting to a local socket costs
microseconds. Neo4j's driver is thread-safe by design; its sessions are not, so each
request takes its own.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import answer as answer_mod
from . import load, route
from . import route as route_mod
from .embed import connect

STATE: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["driver"] = load.driver()
    STATE["index"] = route.entity_index()
    if not STATE["index"]:
        raise SystemExit("data/entities.jsonl is empty — run `kgrag resolve` first.")
    try:
        yield
    finally:
        STATE["driver"].close()


app = FastAPI(
    title="kgrag",
    summary="Hybrid knowledge-graph + vector RAG over SEC filings, with validated citations.",
    lifespan=lifespan,
)


class Ask(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    constrained: bool = Field(
        default=True,
        description="Put the retrieved chunk ids in the schema as an enum, so an invented "
        "citation cannot be generated. Off falls back to validate-and-regenerate.",
    )


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness plus the two things that actually break: the graph and the vector store."""
    try:
        with STATE["driver"].session() as session:
            nodes = session.run("MATCH (e:Entity) RETURN count(e) AS n").single()["n"]
        with connect() as conn:
            chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    except Exception as exc:  # noqa: BLE001 — the endpoint's whole job is to report this
        raise HTTPException(503, f"{type(exc).__name__}: {exc}") from exc
    return {"status": "ok", "entities": nodes, "chunks": chunks,
            "aliases": len(STATE["index"])}


@app.post("/ask")
def ask(body: Ask) -> dict[str, Any]:
    """Answer one question. Every citation in the response resolves to a retrieved chunk.

    `answer()` never raises on a model failure -- it returns a refusal row -- so the only
    5xx here is a database that has gone away, which is what /health is for.
    """
    try:
        with STATE["driver"].session() as session, connect() as conn:
            row = answer_mod.answer(
                body.question, session, conn, STATE["index"], constrain=body.constrained
            )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"{type(exc).__name__}: {exc}") from exc

    return {
        "question": row["question"],
        "answerable": row["answerable"],
        "answer": " ".join(c["text"] for c in row["claims"]),
        "claims": row["claims"],
        "refusal_reason": row["refusal_reason"] or None,
        "route": row["route"],
        "citations": row["cited"],
        "retrieved": row["retrieved_ids"],
        "latency_ms": row["latency_ms"],
        "usd": row["usd"],
    }


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


@app.post("/ask/stream")
def ask_stream(body: Ask) -> StreamingResponse:
    """The same answer, published claim by claim as it is written.

    Latency here is two sequential model calls -- the router and the synthesiser -- and
    almost nothing else: retrieval, graph traversal included, is ~31 ms at p50. Waiting for
    the last token before showing the first is therefore the single largest avoidable part
    of what a user experiences, and it buys nothing: a claim is complete and validated long
    before the answer is.

    Constrained decoding only, and that is not a limitation, it is the reason this endpoint
    can exist. See `answer.stream`.

    Retrieval still happens up front, so the first event carries the route and the retrieved
    ids -- the client knows what the answer may cite before it cites anything.
    """
    def events():
        try:
            with STATE["driver"].session() as session, connect() as conn:
                row = route_mod.route(body.question, session, conn, STATE["index"])
                if row["route"] == "graph" and not row["graph_ids"]:
                    row["vector_ids"] = route_mod.vector_path(conn, body.question, route_mod.TOP_K)
                    row["chunk_ids"] = row["vector_ids"]
                texts = answer_mod.passages(conn, row["chunk_ids"])
                context, valid_ids = answer_mod.build_context(
                    row["graph_facts"], texts, row.get("graph_aggregates")
                )
                yield _sse("retrieved", {"route": row["route"], "citable": sorted(valid_ids)})
                if row["route"] == "refuse" or not context:
                    yield _sse("done", {"answerable": False,
                                        "refusal_reason": "router_refused" if row["route"] == "refuse"
                                        else "no_context"})
                    return
                for kind, item in answer_mod.stream(body.question, context, valid_ids):
                    if kind == "claim":
                        yield _sse("claim", item.model_dump())
                    else:
                        yield _sse("done", {"answerable": item.answerable,
                                            "refusal_reason": item.refusal_reason or None})
        except Exception as exc:  # noqa: BLE001 — a stream cannot raise a 503 once it has begun
            yield _sse("error", {"error": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(events(), media_type="text/event-stream")

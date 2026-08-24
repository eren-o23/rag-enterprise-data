"""Fireworks client: constrained JSON generation, embeddings, caching, and a cost guard.

Fireworks speaks the OpenAI API, so this is the `openai` SDK with a different base_url.
No framework — the caching, retry, and budget logic below is small enough to read in one
sitting and is the part of the pipeline most worth being able to reason about.

Cost control is the point of this module. The md's warning about accidentally spending
two hundred dollars in an afternoon is a real failure mode of LLM extraction pipelines,
so spend is metered per call, projected before the run continues, and hard-capped.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from .config import CACHE, require

BASE_URL = "https://api.fireworks.ai/inference/v1"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

#: Bump when the prompt changes. It is part of the cache key, so an edited prompt
#: invalidates exactly the entries it should and nothing else.
PROMPT_VERSION = "v1"

EXTRACT_MODEL = "accounts/fireworks/models/gpt-oss-120b"
EMBED_MODEL = "accounts/fireworks/models/qwen3-embedding-8b"

#: USD per million tokens (input, output). Verify against `kgrag models` — Fireworks
#: changes its catalogue, and a stale price here makes the budget guard lie.
PRICES: dict[str, tuple[float, float]] = {
    "accounts/fireworks/models/gpt-oss-120b": (0.15, 0.60),
    "accounts/fireworks/models/gpt-oss-20b": (0.07, 0.30),
    "accounts/fireworks/models/qwen3-embedding-8b": (0.10, 0.0),
    "llama3.2:latest": (0.0, 0.0),  # run locally via Ollama, no per-token cost
}
UNKNOWN_PRICE = (2.0, 6.0)  # deliberately pessimistic: an unpriced model trips the guard early


class BudgetExceeded(RuntimeError):
    pass


#: The account's Fireworks quota page shows Serverless Inference Rpm: 10 — a hard cap,
#: not a burst allowance. A sequential loop with no pacing bursts past it in the first
#: few calls, gets 429'd, and exponential backoff (up to 60s a step) then wildly
#: overshoots recovering from a limit that resets every minute. Pacing calls to stay
#: under the cap avoids triggering the penalty at all, instead of reactively paying for
#: it. Raise this if the account's Rpm quota changes.
REQUESTS_PER_MINUTE = 9  # one below the quota, as margin
_last_call = 0.0


T = TypeVar("T")

#: Exponential backoff for transient failures. A 2743-chunk sequential run WILL hit rate
#: limits; without this a single 429 kills the whole run instead of pausing for it.
#: 5xx is included because Fireworks' serverless endpoints occasionally 503 under load.
def _pace() -> None:
    global _last_call
    interval = 60.0 / REQUESTS_PER_MINUTE
    wait = _last_call + interval - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()


def _with_backoff(call: Callable[[], T], attempts: int = 6) -> T:
    for attempt in range(attempts):
        _pace()
        try:
            return call()
        except RateLimitError:
            if attempt == attempts - 1:
                raise
            time.sleep(min(2**attempt, 60))
        except APIStatusError as exc:
            if exc.status_code < 500 or attempt == attempts - 1:
                raise
            time.sleep(min(2**attempt, 60))
        except APIConnectionError:
            if attempt == attempts - 1:
                raise
            time.sleep(min(2**attempt, 60))
    raise AssertionError("unreachable")


@dataclass
class Meter:
    """Running spend, and the hard stop."""

    limit_usd: float = 5.0
    calls: int = 0
    cached: int = 0
    tokens: dict[str, list[int]] = field(default_factory=dict)

    def record(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        row = self.tokens.setdefault(model, [0, 0])
        row[0] += prompt_tokens
        row[1] += completion_tokens
        self.calls += 1
        if self.usd > self.limit_usd:
            raise BudgetExceeded(
                f"spent ${self.usd:.2f} of ${self.limit_usd:.2f} budget after {self.calls} calls. "
                "Nothing is lost — completed calls are cached, so rerunning with a higher "
                "--budget resumes rather than restarts."
            )

    @property
    def usd(self) -> float:
        total = 0.0
        for model, (pin, pout) in self.tokens.items():
            rate_in, rate_out = PRICES.get(model, UNKNOWN_PRICE)
            total += pin / 1e6 * rate_in + pout / 1e6 * rate_out
        return total

    def report(self) -> str:
        lines = [f"spend ${self.usd:.4f} over {self.calls} calls ({self.cached} served from cache)"]
        for model, (pin, pout) in sorted(self.tokens.items()):
            lines.append(f"  {model.split('/')[-1]:24} in={pin:>10,}  out={pout:>9,}")
        return "\n".join(lines)


METER = Meter()


def client(base_url: str = BASE_URL) -> OpenAI:
    # Without an explicit timeout, a connection that drops mid-response hangs the
    # underlying socket read forever - no exception, so _with_backoff never gets a
    # chance to retry. A 2743-call overnight run stalled silently for 11+ hours on
    # exactly this: sockets sat in CLOSE_WAIT while the process waited on a read that
    # was never coming. max_retries=0 because _with_backoff already owns retry policy;
    # the SDK's own retry-on-timeout would silently double it.
    api_key = require("FIREWORKS_API_KEY") if base_url == BASE_URL else "ollama"
    return OpenAI(api_key=api_key, base_url=base_url, timeout=90.0, max_retries=0)


def _cache_path(key: str) -> Any:
    path = CACHE / key[:2] / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def chat_json(
    *,
    system: str,
    user: str,
    schema: dict[str, Any],
    model: str = EXTRACT_MODEL,
    temperature: float = 0.0,
    base_url: str = BASE_URL,
    use_cache: bool = True,
) -> dict[str, Any]:
    """One constrained-decoding call, cached by content.

    `response_format=json_schema` makes Fireworks (or Ollama, same OpenAI-compatible
    shape) constrain generation to the schema, so enum fields cannot take an
    off-ontology value — the guarantee the whole ontology design rests on. Returns
    parsed JSON; the caller validates semantics.
    """
    key = hashlib.sha256(
        json.dumps([model, PROMPT_VERSION, system, user, schema], sort_keys=True).encode()
    ).hexdigest()
    path = _cache_path(key)
    # `kgrag bakeoff` sets use_cache=False. The production corpus was extracted with
    # EXTRACT_MODEL, so every gold chunk is already cached under that model's key — a
    # benchmark reading them back would report $0.0000 and 0.0s for the incumbent and
    # full price for its challengers. Writes still happen; only the read is skipped.
    if use_cache and path.exists():
        METER.cached += 1
        return json.loads(path.read_text())

    response = _with_backoff(
        lambda: client(base_url).chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_schema", "json_schema": {"name": "extraction", "schema": schema}},
        )
    )
    usage = response.usage
    METER.record(model, usage.prompt_tokens, usage.completion_tokens)

    payload = json.loads(response.choices[0].message.content)
    path.write_text(json.dumps(payload))
    return payload


def _embed_key(model: str, text: str, dimensions: int | None) -> str:
    """Cache key for one embedding. The dimension MUST be in here.

    `qwen3-embedding-8b` returns 4096 dims natively and accepts an OpenAI-style
    `dimensions` parameter to truncate (it is Matryoshka-trained, so truncation is
    principled rather than lossy chopping). Without the dimension in the key, a cached
    4096-dim vector is handed back for a 1024-dim request and the mismatch surfaces far
    downstream, if at all.

    The suffix is omitted when no dimension is requested, so the several thousand
    entity-name vectors `kgrag resolve` already cached at native 4096 stay valid.
    """
    suffix = f"|{dimensions}" if dimensions else ""
    return hashlib.sha256(f"{model}|{text}{suffix}".encode()).hexdigest()


def embed(
    texts: list[str],
    model: str = EMBED_MODEL,
    batch: int = 64,
    dimensions: int | None = None,
    use_cache: bool = True,
) -> list[list[float]]:
    """Embed strings. Cached per string, so resolution reruns are free.

    `kgrag embed` passes use_cache=False: 2,743 chunks at three widths is ~390 MB of JSON
    sitting beside a Postgres row that is already durable, and that stage resumes from a
    `WHERE emb_N IS NULL` query instead.
    """
    out: list[list[float] | None] = [None] * len(texts)
    todo: list[int] = []

    for i, text in enumerate(texts):
        path = _cache_path(_embed_key(model, text, dimensions))
        if use_cache and path.exists():
            out[i] = json.loads(path.read_text())
            METER.cached += 1
        else:
            todo.append(i)

    # `dimensions=None` must be omitted entirely rather than sent as null: the parameter is
    # an OpenAI extension and Fireworks rejects an explicit null for it.
    extra = {"dimensions": dimensions} if dimensions else {}

    api = client() if todo else None
    for start in range(0, len(todo), batch):
        idxs = todo[start : start + batch]
        response = _with_backoff(
            lambda: api.embeddings.create(model=model, input=[texts[i] for i in idxs], **extra)
        )
        METER.record(model, response.usage.total_tokens, 0)
        for i, item in zip(idxs, response.data):
            out[i] = item.embedding
            if use_cache:
                _cache_path(_embed_key(model, texts[i], dimensions)).write_text(
                    json.dumps(item.embedding)
                )

    return out  # type: ignore[return-value]


def list_models() -> None:
    """`kgrag models` — the API is the only authoritative catalogue; the docs pages 404."""
    for m in client().models.list().data:
        marker = "*" if m.id in PRICES else " "
        print(f"{marker} {m.id}")
    print("\n* = priced in fireworks.PRICES")

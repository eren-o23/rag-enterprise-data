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
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from .config import CACHE, require

BASE_URL = "https://api.fireworks.ai/inference/v1"

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
}
UNKNOWN_PRICE = (2.0, 6.0)  # deliberately pessimistic: an unpriced model trips the guard early


class BudgetExceeded(RuntimeError):
    pass


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


def client() -> OpenAI:
    return OpenAI(api_key=require("FIREWORKS_API_KEY"), base_url=BASE_URL)


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
) -> dict[str, Any]:
    """One constrained-decoding call, cached by content.

    `response_format=json_schema` makes Fireworks constrain generation to the schema, so
    enum fields cannot take an off-ontology value — the guarantee the whole ontology
    design rests on. Returns parsed JSON; the caller validates semantics.
    """
    key = hashlib.sha256(
        json.dumps([model, PROMPT_VERSION, system, user, schema], sort_keys=True).encode()
    ).hexdigest()
    path = _cache_path(key)
    if path.exists():
        METER.cached += 1
        return json.loads(path.read_text())

    response = client().chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_schema", "json_schema": {"name": "extraction", "schema": schema}},
    )
    usage = response.usage
    METER.record(model, usage.prompt_tokens, usage.completion_tokens)

    payload = json.loads(response.choices[0].message.content)
    path.write_text(json.dumps(payload))
    return payload


def embed(texts: list[str], model: str = EMBED_MODEL, batch: int = 64) -> list[list[float]]:
    """Embed short strings (entity names). Cached per string, so resolution reruns are free."""
    out: list[list[float] | None] = [None] * len(texts)
    todo: list[int] = []

    for i, text in enumerate(texts):
        key = hashlib.sha256(f"{model}|{text}".encode()).hexdigest()
        path = _cache_path(key)
        if path.exists():
            out[i] = json.loads(path.read_text())
            METER.cached += 1
        else:
            todo.append(i)

    api = client() if todo else None
    for start in range(0, len(todo), batch):
        idxs = todo[start : start + batch]
        response = api.embeddings.create(model=model, input=[texts[i] for i in idxs])
        METER.record(model, response.usage.total_tokens, 0)
        for i, item in zip(idxs, response.data):
            out[i] = item.embedding
            key = hashlib.sha256(f"{model}|{texts[i]}".encode()).hexdigest()
            _cache_path(key).write_text(json.dumps(item.embedding))

    return out  # type: ignore[return-value]


def list_models() -> None:
    """`kgrag models` — the API is the only authoritative catalogue; the docs pages 404."""
    for m in client().models.list().data:
        marker = "*" if m.id in PRICES else " "
        print(f"{marker} {m.id}")
    print("\n* = priced in fireworks.PRICES")

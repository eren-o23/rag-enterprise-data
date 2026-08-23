"""Collapse surface forms into canonical entities.

"Acme Corp", "Acme Corporation", and "ACME" must become one node. Three match rules,
each cheap and each defensible:

1. Identical after normalisation.
2. Embedding cosine >= TAU *and* token Jaccard >= JACCARD_FLOOR. Two independent
   signals, both required. Cosine alone happily merges "Advanced Micro Devices" with
   "Advanced Energy Industries"; the lexical gate stops it.
3. One name is the acronym of the other. This is the case rule 2 structurally cannot
   catch — "AMD" and "Advanced Micro Devices" share no tokens, so their Jaccard is 0 —
   and it is also the single most common alias pattern in SEC filings.

Plus a hand-maintained override file, in both directions. Every corpus has a handful of
pairs no threshold will ever get right - "TSMC" expands to Taiwan Semiconductor
Manufacturing *Company*, so its C comes from a word the normaliser strips - and spending
a week tuning for them is worse engineering than writing them down. The override count
is itself a quality metric: if it grows past a couple of dozen, the rules are wrong.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Iterable

import numpy as np

from . import fireworks, jsonl
from .config import ENTITIES, EVAL, EXTRACTIONS, OVERRIDES

TAU = 0.88           # cosine threshold, from `kgrag sweep` on eval/resolution_pairs.jsonl (F1=0.769, plateau 0.88-0.96)
JACCARD_FLOOR = 0.5
MIN_ACRONYM = 2

#: Legal form only. It is tempting to also strip industry words - "technology",
#: "solutions", "semiconductor" - but in this corpus those are part of the name:
#: stripping them turns "Taiwan Semiconductor Manufacturing" into "taiwan manufacturing"
#: and merges "ON Semiconductor" with anything else starting in ON.
LEGAL_SUFFIXES = re.compile(
    r"\b(inc|incorporated|corp|corporation|co|company|llc|l\.l\.c|ltd|limited|plc"
    r"|nv|n\.v|sa|s\.a|ag|gmbh|holdings?|the)\b",
    re.IGNORECASE,
)


def normalize(name: str) -> str:
    # EDGAR appends the state of incorporation to some filer names ("APPLIED MATERIALS
    # INC /DE"). Left in, "applied materials de" acronyms to "amd" and false-merges with
    # the real AMD via the acronym rule, which has no cosine floor by design (see module
    # docstring) — this has to be stripped before that rule ever sees the name.
    name = re.sub(r"\s*/[A-Za-z]{2}$", "", name)
    name = name.replace("&", " and ").casefold()
    name = re.sub(r"[^\w\s]", " ", name)
    name = LEGAL_SUFFIXES.sub(" ", name)
    # Punctuation is stripped before the suffix pass, so "N.V." has already become
    # "n v" and no longer matches. Dropping trailing single-letter tokens catches it,
    # along with "S.A." and "L.L.C." — real names never end in a bare letter.
    parts = re.sub(r"\s+", " ", name).strip().split()
    while len(parts) > 1 and len(parts[-1]) == 1:
        parts.pop()
    return " ".join(parts)


def tokens(normalized: str) -> frozenset[str]:
    return frozenset(normalized.split())


def acronym(normalized: str) -> str:
    parts = normalized.split()
    return "".join(p[0] for p in parts) if len(parts) > 1 else ""


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def blocking_keys(normalized: str, surface: str) -> set[str]:
    """Cheap candidate generation. Every key a name emits is a bucket it may match in.

    Prefix blocking alone is what breaks naive resolvers: "amd" and "advanced micro
    devices" share no prefix, so they are never even compared. The acronym keys exist
    to put them in the same bucket.
    """
    keys = {f"exact:{normalized}", f"pre:{normalized[:4]}"}
    if initials := acronym(normalized):
        keys.add(f"acr:{initials}")
    # A short all-caps surface form is probably itself an acronym.
    if " " not in normalized and 2 <= len(normalized) <= 6 and surface.isupper():
        keys.add(f"acr:{normalized}")
    return keys


class Union:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def join(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[max(ri, rj)] = min(ri, rj)


def matches(
    a: dict[str, Any], b: dict[str, Any], cosine: float, overrides: dict[tuple[str, str], bool]
) -> str | None:
    """Return the rule that matched, or None. Named rules make the sweep interpretable."""
    for key in ((a["norm"], b["norm"]), (b["norm"], a["norm"])):
        if key in overrides:
            return "override" if overrides[key] else None
    if a["norm"] == b["norm"]:
        return "exact"
    if cosine >= TAU and jaccard(a["tokens"], b["tokens"]) >= JACCARD_FLOOR:
        return "embedding"
    for x, y in ((a, b), (b, a)):
        if len(x["norm"]) >= MIN_ACRONYM and " " not in x["norm"] and acronym(y["norm"]) == x["norm"]:
            return "acronym"
    return None


def _candidates(records: list[dict[str, Any]]) -> set[tuple[int, int]]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        for key in blocking_keys(rec["norm"], rec["surface"]):
            buckets[f"{rec['type']}|{key}"].append(i)

    pairs: set[tuple[int, int]] = set()
    for members in buckets.values():
        # A pathological bucket (every one-word Location, say) would be quadratic.
        # ponytail: 400 is far above any real bucket here; raise it if that stops holding.
        if len(members) > 400:
            continue
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                pairs.add((members[x], members[y]))
    return pairs


def cluster(records: list[dict[str, Any]], overrides: dict[tuple[str, str], bool]) -> list[list[int]]:
    vectors = np.array(fireworks.embed([r["surface"] for r in records]), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9

    union = Union(len(records))
    rules: Counter[str] = Counter()
    for i, j in _candidates(records):
        cos = float(vectors[i] @ vectors[j])
        if rule := matches(records[i], records[j], cos, overrides):
            union.join(i, j)
            rules[rule] += 1

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(records)):
        groups[union.find(i)].append(i)
    print(f"  merges by rule: {dict(rules)}")
    return list(groups.values())


def _mentions() -> list[dict[str, Any]]:
    """Unique (surface form, type) pairs, with the chunks each was seen in."""
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for row in jsonl.read(EXTRACTIONS):
        for mention in row["mentions"]:
            key = (mention["name"], mention["type"])
            norm = normalize(mention["name"])
            rec = seen.setdefault(
                key,
                {
                    "surface": mention["name"],
                    "type": mention["type"],
                    "norm": norm,
                    "tokens": tokens(norm),
                    "chunks": [],
                    "count": 0,
                },
            )
            rec["chunks"].append(row["chunk_id"])
            rec["count"] += 1
    # A mention that normalises to nothing ("The Company", "Inc.") is not an entity.
    return [r for r in seen.values() if r["norm"]]


def canonical_id(entity_type: str, cluster_norms: Iterable[str]) -> str:
    """Stable across runs: keyed on the cluster's smallest member, not its most frequent.

    Frequency shifts when the corpus grows, and an id that moves is an id that breaks
    every citation Phase 4 ever emitted.
    """
    import hashlib

    return hashlib.sha256(f"{entity_type}|{min(cluster_norms)}".encode()).hexdigest()[:16]


def run() -> None:
    records = _mentions()
    if not records:
        raise SystemExit("data/extractions.jsonl is empty — run `kgrag extract` first.")
    overrides = {(normalize(r["a"]), normalize(r["b"])): r["same"] for r in jsonl.read(OVERRIDES)}
    print(f"{len(records)} unique (surface form, type) mentions, {len(overrides)} hand overrides")

    entities = []
    for group in cluster(records, overrides):
        members = [records[i] for i in group]
        # The display name is the most frequent surface form; the rest become aliases,
        # which Phase 3 needs to map entities named in a question onto node ids.
        best = max(members, key=lambda r: (r["count"], -len(r["surface"])))
        entities.append(
            {
                "canonical_id": canonical_id(best["type"], (m["norm"] for m in members)),
                "type": best["type"],
                "name": best["surface"],
                "aliases": sorted({m["surface"] for m in members} - {best["surface"]}),
                "mention_chunks": sorted({c for m in members for c in m["chunks"]}),
                "mention_count": sum(m["count"] for m in members),
            }
        )

    assert len({e["canonical_id"] for e in entities}) == len(entities), "canonical id collision"
    jsonl.write(ENTITIES, entities)

    merged = len(records) - len(entities)
    print(f"entities.jsonl: {len(entities)} entities ({merged} surface forms merged away)")
    by_type = Counter(e["type"] for e in entities)
    for etype, count in by_type.most_common():
        print(f"  {etype:18} {count:5}")
    print("\nlargest clusters:")
    for e in sorted(entities, key=lambda e: -len(e["aliases"]))[:8]:
        print(f"  {e['name'][:38]:40} <- {e['aliases'][:5]}")


def write_candidates(limit: int = 80) -> None:
    """`kgrag candidates` — surface real near-miss pairs from the corpus to hand-label.

    Ranked by closeness to the current TAU: the boundary cases are what a tau sweep
    actually needs to discriminate, not the obvious exact/acronym matches.
    """
    records = _mentions()
    if not records:
        raise SystemExit("data/extractions.jsonl is empty — run `kgrag extract` first.")

    vectors = np.array(fireworks.embed([r["surface"] for r in records]), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9

    scored = []
    for i, j in _candidates(records):
        cos = float(vectors[i] @ vectors[j])
        scored.append((abs(cos - TAU), records[i], records[j], cos))
    scored.sort(key=lambda t: t[0])

    seen: set[tuple[str, str]] = set()
    rows = []
    for _, a, b, cos in scored:
        key = tuple(sorted((a["norm"], b["norm"])))
        if key in seen:
            continue
        seen.add(key)
        rows.append({"a": a["surface"], "b": b["surface"], "type": a["type"], "same": None})
        if len(rows) >= limit:
            break

    path = EVAL / "resolution_pairs.jsonl"
    jsonl.write(path, rows)
    print(f"{path}: {len(rows)} candidate pairs written — hand-label 'same' (true/false), trim to ~50")


def sweep() -> None:
    """Score the match rules against hand-labelled pairs and sweep TAU.

    Reads eval/resolution_pairs.jsonl: {"a", "b", "type", "same": true|false}. The pairs
    are sampled from real near-miss candidates, not invented, or the numbers flatter.
    """
    path = EVAL / "resolution_pairs.jsonl"
    labelled = list(jsonl.read(path))
    if not labelled:
        raise SystemExit(f"{path} is empty — run `kgrag candidates` to generate pairs to label.")

    names = sorted({p["a"] for p in labelled} | {p["b"] for p in labelled})
    vectors = np.array(fireworks.embed(names), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    index = {name: i for i, name in enumerate(names)}

    global TAU
    original, best = TAU, (0.0, TAU)
    print(f"{'tau':>6} {'P':>7} {'R':>7} {'F1':>7}   (n={len(labelled)})")
    for tau in [round(0.70 + 0.02 * i, 2) for i in range(14)]:
        TAU = tau
        tp = fp = fn = 0
        for pair in labelled:
            a = {"norm": normalize(pair["a"]), "tokens": tokens(normalize(pair["a"])), "surface": pair["a"]}
            b = {"norm": normalize(pair["b"]), "tokens": tokens(normalize(pair["b"])), "surface": pair["b"]}
            cos = float(vectors[index[pair["a"]]] @ vectors[index[pair["b"]]])
            predicted = matches(a, b, cos, {}) is not None
            tp += predicted and pair["same"]
            fp += predicted and not pair["same"]
            fn += (not predicted) and pair["same"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        marker = " <-" if f1 > best[0] else ""
        best = max(best, (f1, tau))
        print(f"{tau:>6} {precision:>7.3f} {recall:>7.3f} {f1:>7.3f}{marker}")
    TAU = original
    print(f"\nbest F1 {best[0]:.3f} at tau={best[1]} (module default TAU={original})")

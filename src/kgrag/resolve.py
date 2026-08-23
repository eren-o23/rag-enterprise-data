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
from .config import CHUNKS, ENTITIES, EVAL, EXTRACTIONS, OVERRIDES

#: Both from `kgrag sweep` over eval/resolution_pairs.jsonl. F1 sits on a flat 0.930
#: plateau across tau 0.86-0.90 at floor 0.67-0.75, so these are the middle of a stable
#: region rather than a single tuned peak.
TAU = 0.88
JACCARD_FLOOR = 0.67
#: The prefix rule leans on the lexical anchor (a shared leading token), so it needs far
#: less from the embedding than the fuzzy rule does.
PREFIX_COSINE = 0.55
MIN_ACRONYM = 2

#: Foreign legal forms `LEGAL_SUFFIXES` deliberately does not strip. When one of these is
#: what separates two otherwise-identical names, it separates two *incorporations*:
#: "Skyworks Solutions, Inc." and "Skyworks Solutions Oy" are a US parent and its Finnish
#: subsidiary, and Exhibit 21 lists them precisely because they are different entities.
FOREIGN_LEGAL_FORMS = frozenset(
    "oy oyj ab as aps sas spa srl sarl pte sdn bhd kk gk kft doo bv ug kg cv lda ltda "
    "pty pt zoo sro sl unipersonal".split()
)

#: Place words too generic to imply a separate entity on their own.
GEO_STOPWORDS = frozenset("the of and new north south east west central".split())

#: The geography half of `entity_markers()` is derived from the corpus, which means it is
#: empty on a fresh clone before `kgrag extract` has run. This floor keeps the rule
#: working anyway: without it the parent/subsidiary block would silently degrade to
#: nothing and quietly reintroduce the self-loop bug it exists to prevent.
GEO_CORE = frozenset(
    "america american argentina asia australia austria belgium bermuda brasil brazil "
    "britain canada cayman chengdu chile china colombia czech denmark deutschland dublin "
    "egypt emea england europe european finland france french german germany greece "
    "holland hongkong hungary iberia india indonesia ireland israel italia italy japan "
    "kong korea luxembourg malaysia malta mexico nederland netherlands nordic norway "
    "pacific philippines poland portugal prc romania russia scotland shanghai shenzhen "
    "singapore slovakia slovenia spain suisse sweden swiss switzerland taiwan thailand "
    "turkey uk ukraine vietnam wales".split()
)

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


def raw_acronym(surface: str) -> str:
    """Initials taken *before* legal-form stripping.

    `normalize()` removes "Company", so `acronym()` of Taiwan Semiconductor Manufacturing
    Company is "tsm" and can never match TSMC — the case decisions.md settles with a hand
    override. Taking initials from the raw surface form recovers TSMC, and with it ESMC,
    SMIC, UMC and ADI, which all borrow their last letter from a stripped legal word.
    """
    words = re.sub(r"[^\w\s]", " ", surface.casefold()).split()
    return "".join(w[0] for w in words if w) if len(words) > 1 else ""


_MARKERS: frozenset[str] | None = None


def entity_markers() -> frozenset[str]:
    """Tokens that mark a *separate* legal entity when they are what differs.

    Two company names that differ only by a place are a parent and its local subsidiary,
    not two spellings of one company — "Qorvo, Inc." vs "Qorvo Germany GmbH". That is the
    single strongest signal separating the two: measured over the labelled pairs, the
    tokens distinguishing `same=false` pairs are overwhelmingly geographic (shanghai,
    india, ireland, korea) while those distinguishing `same=true` pairs are industry
    descriptors (technologies, manufacturing, solutions).

    The geography half is derived from the corpus's own `Location` mentions rather than a
    hand-written country list, so it tracks the corpus instead of rotting beside it.
    """
    global _MARKERS
    if _MARKERS is None:
        geo: set[str] = set()
        for row in jsonl.read(EXTRACTIONS):  # yields nothing if the file is absent
            for mention in row["mentions"]:
                if mention["type"] == "Location":
                    geo |= tokens(normalize(mention["name"]))
        _MARKERS = frozenset((geo - GEO_STOPWORDS) | FOREIGN_LEGAL_FORMS | GEO_CORE)
    return _MARKERS


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
    # Without this the raw-acronym rule in matches() could never fire: "tsmc" blocks on
    # acr:tsmc while "taiwan semiconductor manufacturing" blocks on acr:tsm, so the pair
    # is never generated as a candidate in the first place.
    if raw := raw_acronym(surface):
        keys.add(f"acr:{raw}")
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

    # Acronyms are checked before the marker block on purpose: TSMC expands to *Taiwan*
    # Semiconductor Manufacturing, so a place word is exactly what separates the pair,
    # and blocking on it would reject the one rule that gets this right.
    for x, y in ((a, b), (b, a)):
        if len(x["norm"]) >= MIN_ACRONYM and " " not in x["norm"]:
            if acronym(y["norm"]) == x["norm"] or raw_acronym(y["surface"]) == x["norm"]:
                return "acronym"

    if (a["tokens"] ^ b["tokens"]) & entity_markers():
        return None

    if cosine >= TAU and jaccard(a["tokens"], b["tokens"]) >= JACCARD_FLOOR:
        return "embedding"

    # A filing that says `Cisco Systems, Inc. ("Cisco")` is naming the same company by its
    # brand. Those share only a leading token, so Jaccard is far below the floor — but the
    # shared *first* token is a strong anchor once a place word has been ruled out above.
    if cosine >= PREFIX_COSINE:
        for x, y in ((a, b), (b, a)):
            short = x["norm"].split()
            if len(short) == 1 and len(short[0]) >= 3 and y["norm"].split()[:1] == short:
                return "prefix"
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


#: A filing that writes `NXP Semiconductors, N.V. ("NXP")` is declaring an alias itself,
#: which is ground truth for same=true that involves no judgement from me. An Exhibit 21
#: table is the mirror image: it exists to enumerate *separate legal entities*, so any two
#: rows are same=false by construction, as is any row against the filer that owns it.
ALIAS_DEF = re.compile(
    r"([A-Z][\w&.,'\-]*(?:\s+[A-Z][\w&.,'\-]*){1,6})\s*\(\s*[“\"]?([A-Z][A-Za-z]{1,14})[”\"]?\s*\)"
)


def _declares_alias(full: str, abbr: str) -> bool:
    """True only if the parenthetical is really an alias, not a role or an aside.

    `Dawn Hudson (Chairperson)` and `Intel, Intel (China)` both match the regex; neither
    declares an alias. Requiring the short form to be a token of the long one, or its
    initials, drops those without hand-curating a stoplist.
    """
    short = normalize(abbr)
    if not short:
        return False
    if short in tokens(normalize(full)):
        return True
    words = re.sub(r"[^\w\s]", " ", full.casefold()).split()
    return short in {"".join(w[0] for w in words if w), acronym(normalize(full))}


def mine_pairs(negatives: int = 220) -> None:
    """`kgrag mine-pairs` — derive labelled pairs from document structure, not judgement.

    `kgrag candidates` ranks by |cosine - TAU|, which is the right sampler for finding the
    decision boundary but the wrong one for auditing over-merging: the pairs that actually
    over-merge (a parent and its own subsidiary) sit far *above* TAU, so that sampler never
    surfaces them. This mines the two shapes the corpus can label for itself, then merges
    the result into whatever is already labelled by hand.
    """
    chunks = list(jsonl.read(CHUNKS))
    if not chunks:
        raise SystemExit("data/chunks.jsonl is empty — run `kgrag chunk` first.")
    mentions_by_chunk = {r["chunk_id"]: r["mentions"] for r in jsonl.read(EXTRACTIONS)}
    companies = {
        m["name"] for ms in mentions_by_chunk.values() for m in ms if m["type"] == "Company"
    }

    seen: set[tuple[str, str]] = set()

    def make(a: str, b: str, same: bool, source: str) -> dict[str, Any] | None:
        key = tuple(sorted((normalize(a), normalize(b))))
        # Equal normalised forms are the same row seen twice through extraction noise
        # ("... (China) Co., Ltd." vs "... (China) Co., China Ltd."), not a real pair.
        if key in seen or not key[0] or key[0] == key[1]:
            return None
        seen.add(key)
        return {"a": a, "b": b, "type": "Company", "same": same, "source": source}

    # Preserve existing hand labels — they stay authoritative where they exist.
    rows: list[dict[str, Any]] = []
    for pair in jsonl.read(EVAL / "resolution_pairs.jsonl"):
        if pair.get("same") is None:
            continue
        key = tuple(sorted((normalize(pair["a"]), normalize(pair["b"]))))
        if key in seen or not key[0] or key[0] == key[1]:
            continue
        seen.add(key)
        rows.append({**pair, "source": pair.get("source", "hand")})
    kept_hand = len(rows)

    positives = 0
    for chunk in chunks:
        for full, abbr in ALIAS_DEF.findall(chunk["text"]):
            full = full.strip()
            if full not in companies or abbr not in companies or not _declares_alias(full, abbr):
                continue
            if row := make(full, abbr, True, "alias-declaration"):
                rows.append(row)
                positives += 1

    pool: list[dict[str, Any]] = []
    for chunk in chunks:
        if "EX-21" not in chunk["section_path"]:
            continue
        subs = [m["name"] for m in mentions_by_chunk.get(chunk["chunk_id"], []) if m["type"] == "Company"]
        for sub in subs:
            if row := make(chunk["company"], sub, False, "ex21-parent"):
                pool.append(row)
        for x in range(len(subs)):
            for y in range(x + 1, len(subs)):
                if row := make(subs[x], subs[y], False, "ex21-sibling"):
                    pool.append(row)

    # Evenly strided rather than truncated, so the sample spans every filer's table
    # instead of exhausting the first one. Deterministic, so reruns are comparable.
    if len(pool) > negatives:
        stride = len(pool) / negatives
        pool = [pool[int(i * stride)] for i in range(negatives)]
    rows.extend(pool)

    path = EVAL / "resolution_pairs.jsonl"
    jsonl.write(path, rows)
    counts = Counter(r["source"] for r in rows)
    print(f"{path}: {len(rows)} pairs ({kept_hand} pre-existing hand labels kept)")
    for source, n in counts.most_common():
        same = sum(1 for r in rows if r["source"] == source and r["same"])
        print(f"  {source:20} {n:5}  ({same} same / {n - same} different)")


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

"""`kgrag bakeoff` — extraction quality, cost, and latency across 3 candidate models.

Runs each model against the 20 hand-labeled gold chunks only, not the full corpus — the
production extraction already ran on `EXTRACT_MODEL`; this is a comparison, not a rerun.
Every call still goes through `extract.extract_one`, so bakeoff numbers reflect the same
schema-plus-evidence-check pipeline that actually shipped, not raw model output.
"""

from __future__ import annotations

import time
from typing import Any

from . import extract, fireworks, jsonl
from .config import CHUNKS, EVAL

MODELS = [
    ("gpt-oss-120b", fireworks.EXTRACT_MODEL, fireworks.BASE_URL),
    ("gpt-oss-20b", "accounts/fireworks/models/gpt-oss-20b", fireworks.BASE_URL),
    ("llama3.2:3b", "llama3.2:latest", fireworks.OLLAMA_BASE_URL),
]


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _score(predicted: set[tuple[str, ...]], gold: set[tuple[str, ...]]) -> tuple[int, int, int]:
    return len(predicted & gold), len(predicted - gold), len(gold - predicted)


def _mention_set(mentions: list[dict[str, Any]]) -> set[tuple[str, ...]]:
    return {(m["name"].casefold(), m["type"]) for m in mentions}


def _relation_set(relations: list[dict[str, Any]]) -> set[tuple[str, ...]]:
    return {(r["subject"].casefold(), r["predicate"], r["object"].casefold()) for r in relations}


def run(only: str | None = None) -> None:
    gold = list(jsonl.read(EVAL / "extraction_gold.jsonl"))
    if not gold:
        raise SystemExit("eval/extraction_gold.jsonl is empty.")
    chunks_by_id = {c["chunk_id"]: c for c in jsonl.read(CHUNKS)}
    missing = [g["chunk_id"] for g in gold if g["chunk_id"] not in chunks_by_id]
    if missing:
        raise SystemExit(f"{len(missing)} gold chunk_ids not in data/chunks.jsonl: {missing[:5]}")
    rows = [(g, chunks_by_id[g["chunk_id"]]) for g in gold]

    gold_mentions = sum(len(g["mentions"]) for g in gold)
    print(
        f"{len(gold)} gold chunks, {gold_mentions} labelled mentions.\n"
        "Read RECALL, not precision or F1. The gold set labels ~3x fewer mentions than the\n"
        "models extract -- on an Exhibit 21 chunk a labeller stops after a few rows while\n"
        "the model takes all 68 -- so most 'false positives' are real entities nobody wrote\n"
        "down. Gold is a subset of truth, which leaves recall meaningful and precision a\n"
        "measure of how thorough the labelling was. Precision here penalises the most\n"
        "thorough model hardest, so it inverts the ranking it appears to give.\n"
    )
    header = f"{'model':16} {'cost':>9} {'sec':>7} {'mP':>6} {'mR':>6} {'mF1':>6} {'rP':>6} {'rR':>6} {'rF1':>6}"
    print(header)
    for label, model, base_url in MODELS:
        if only and only not in label:
            continue
        before = fireworks.METER.usd
        start = time.monotonic()
        m_tp = m_fp = m_fn = r_tp = r_fp = r_fn = errors = 0

        for gold_row, chunk in rows:
            # Uncached on purpose: cost and latency are the point of this table, and the
            # incumbent's gold chunks are already cached from the production run.
            result = extract.extract_one(chunk, model=model, base_url=base_url, use_cache=False)
            if "error" in result:
                errors += 1
                predicted_mentions, predicted_relations = [], []
            else:
                predicted_mentions, predicted_relations = result["mentions"], result["relations"]

            tp, fp, fn = _score(_mention_set(predicted_mentions), _mention_set(gold_row["mentions"]))
            m_tp, m_fp, m_fn = m_tp + tp, m_fp + fp, m_fn + fn
            tp, fp, fn = _score(_relation_set(predicted_relations), _relation_set(gold_row["relations"]))
            r_tp, r_fp, r_fn = r_tp + tp, r_fp + fp, r_fn + fn

        elapsed = time.monotonic() - start
        cost = fireworks.METER.usd - before
        mp, mr, mf1 = _prf(m_tp, m_fp, m_fn)
        rp, rr, rf1 = _prf(r_tp, r_fp, r_fn)
        err = f"  ({errors} errors)" if errors else ""
        print(
            f"{label:16} ${cost:>8.4f} {elapsed:>7.1f} "
            f"{mp:>6.3f} {mr:>6.3f} {mf1:>6.3f} {rp:>6.3f} {rr:>6.3f} {rf1:>6.3f}{err}"
        )

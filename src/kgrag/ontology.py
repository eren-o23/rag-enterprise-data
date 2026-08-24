"""The ontology, and the extraction contract built from it.

This module is the single source of truth. The prompt, the JSON schema handed to
the model, the post-extraction validators, the Cypher writes, and the README table
are all generated from what is defined here. Change a type in one place.

The md's first failure mode is an open-ended ontology producing a graph nobody can
query. The defence is that `EntityType` and `RelationType` are enums inside the JSON
schema, so constrained decoding makes an off-ontology value structurally unreachable
rather than merely discouraged.
"""

from __future__ import annotations

import re
from collections import Counter
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class EntityType(StrEnum):
    COMPANY = "Company"
    PERSON = "Person"
    PRODUCT = "Product"
    LOCATION = "Location"
    REGULATOR = "Regulator"
    RISK_TOPIC = "RiskTopic"
    LEGAL_PROCEEDING = "LegalProceeding"


class RelationType(StrEnum):
    SUBSIDIARY_OF = "SUBSIDIARY_OF"
    ACQUIRED = "ACQUIRED"
    COMPETES_WITH = "COMPETES_WITH"
    SUPPLIES = "SUPPLIES"
    PARTNERS_WITH = "PARTNERS_WITH"
    AUDITED_BY = "AUDITED_BY"
    OFFICER_OF = "OFFICER_OF"
    DIRECTOR_OF = "DIRECTOR_OF"
    OFFERS = "OFFERS"
    INCORPORATED_IN = "INCORPORATED_IN"
    OPERATES_IN = "OPERATES_IN"
    REGULATED_BY = "REGULATED_BY"
    EXPOSED_TO = "EXPOSED_TO"
    PARTY_TO = "PARTY_TO"


C, P = EntityType.COMPANY, EntityType.PERSON

#: Every relation's (subject type, object type) signature. A relation whose endpoints
#: do not match its signature is dropped — this is what stops "Nvidia SUPPLIES Taiwan".
ALLOWED_EDGES: dict[RelationType, tuple[EntityType, EntityType]] = {
    RelationType.SUBSIDIARY_OF: (C, C),
    RelationType.ACQUIRED: (C, C),
    RelationType.COMPETES_WITH: (C, C),
    RelationType.SUPPLIES: (C, C),
    RelationType.PARTNERS_WITH: (C, C),
    RelationType.AUDITED_BY: (C, C),
    RelationType.OFFICER_OF: (P, C),
    RelationType.DIRECTOR_OF: (P, C),
    RelationType.OFFERS: (C, EntityType.PRODUCT),
    RelationType.INCORPORATED_IN: (C, EntityType.LOCATION),
    RelationType.OPERATES_IN: (C, EntityType.LOCATION),
    RelationType.REGULATED_BY: (C, EntityType.REGULATOR),
    RelationType.EXPOSED_TO: (C, EntityType.RISK_TOPIC),
    RelationType.PARTY_TO: (C, EntityType.LEGAL_PROCEEDING),
}

#: Written for the extraction prompt, which is generated from this dict so the prompt
#: can never drift from the schema.
RELATION_DOCS: dict[RelationType, str] = {
    RelationType.SUBSIDIARY_OF: "subject is owned or controlled by object (Exhibit 21 lists these directly)",
    RelationType.ACQUIRED: "subject acquired, merged with, or purchased object",
    RelationType.COMPETES_WITH: "the filing names object as a competitor of subject",
    RelationType.SUPPLIES: "subject supplies components, wafers, IP, or services to object",
    RelationType.PARTNERS_WITH: "a named collaboration, joint venture, or licensing arrangement",
    RelationType.AUDITED_BY: "object is subject's independent registered public accounting firm",
    RelationType.OFFICER_OF: "subject holds an executive office at object (CEO, CFO, EVP, ...)",
    RelationType.DIRECTOR_OF: "subject sits on object's board of directors. Proxy bios name "
    "directors' seats at OTHER companies — extract those too, they are the most valuable edges here",
    RelationType.OFFERS: "subject sells or markets object as a named product, platform, or service line",
    RelationType.INCORPORATED_IN: "subject's state or country of incorporation",
    RelationType.OPERATES_IN: "subject has named operations, fabs, offices, or material revenue in object",
    RelationType.REGULATED_BY: "object is a government body or agency with named authority over subject",
    RelationType.EXPOSED_TO: "the filing identifies object as a material risk to subject",
    RelationType.PARTY_TO: "subject is a named party to object (a lawsuit, investigation, or proceeding)",
}

#: How each relation reads as a sentence, for verbalising a graph path before it reaches
#: the synthesis prompt. Phase 4's spec is explicit that raw triples generate awkward text.
#: RELATION_DOCS cannot be reused for this: it documents the relation *to the extraction
#: model* ("subject is owned or controlled by object") and reads as a definition, not as a
#: statement about two named entities. Rendered subject-first, so the phrase must follow a
#: subject and precede an object with no other glue.
RELATION_PHRASE: dict[RelationType, str] = {
    RelationType.SUBSIDIARY_OF: "is a subsidiary of",
    RelationType.ACQUIRED: "acquired",
    RelationType.COMPETES_WITH: "competes with",
    RelationType.SUPPLIES: "supplies",
    RelationType.PARTNERS_WITH: "has a named partnership with",
    RelationType.AUDITED_BY: "is audited by",
    RelationType.OFFICER_OF: "is an executive officer of",
    RelationType.DIRECTOR_OF: "sits on the board of",
    RelationType.OFFERS: "offers",
    RelationType.INCORPORATED_IN: "is incorporated in",
    RelationType.OPERATES_IN: "has operations in",
    RelationType.REGULATED_BY: "is regulated by",
    RelationType.EXPOSED_TO: "is exposed to the risk of",
    RelationType.PARTY_TO: "is a party to",
}

#: What the *subjects* of a relation are called, collectively, from the object's side.
#: Aggregation needs this and RELATION_PHRASE cannot supply it: an inbound count has a
#: plural subject, so "32 distinct entities is a subsidiary of AMD" is what reusing the
#: phrase produces. A noun sidesteps agreement entirely -- "AMD has 32 subsidiaries".
#: Outbound aggregates need no noun: the phrase already agrees with a singular anchor.
RELATION_NOUN: dict[RelationType, str] = {
    RelationType.SUBSIDIARY_OF: "subsidiaries",
    RelationType.ACQUIRED: "acquirers",
    RelationType.COMPETES_WITH: "companies naming it as a competitor",
    RelationType.SUPPLIES: "suppliers",
    RelationType.PARTNERS_WITH: "named partners",
    RelationType.AUDITED_BY: "audit clients",
    RelationType.OFFICER_OF: "executive officers",
    RelationType.DIRECTOR_OF: "board directors",
    RelationType.OFFERS: "vendors offering it",
    RelationType.INCORPORATED_IN: "companies incorporated there",
    RelationType.OPERATES_IN: "companies with operations there",
    RelationType.REGULATED_BY: "companies it regulates",
    RelationType.EXPOSED_TO: "companies disclosing it as a material risk",
    RelationType.PARTY_TO: "named parties",
}

#: RiskTopic is a closed vocabulary, not free text. Left open, the model invents a new
#: risk phrasing per filing and every RiskTopic node has degree 1 — a graph that looks
#: full and answers nothing.
RISK_TOPICS: tuple[str, ...] = (
    "supply chain concentration",
    "customer concentration",
    "export controls and trade restrictions",
    "geopolitical tension",
    "intellectual property litigation",
    "cybersecurity incident",
    "talent retention",
    "manufacturing yield and capacity",
    "raw material and energy cost",
    "demand cyclicality",
    "foreign exchange",
    "environmental and climate regulation",
    "antitrust and competition regulation",
    "data privacy regulation",
    "acquisition integration",
)


# ---------------------------------------------------------------------------
# Extraction contract
# ---------------------------------------------------------------------------


class Mention(BaseModel):
    name: str = Field(description="The entity's surface form, copied verbatim from the chunk.")
    type: EntityType


class Relation(BaseModel):
    subject: str = Field(description="Must exactly match the `name` of a declared mention.")
    predicate: RelationType
    object: str = Field(description="Must exactly match the `name` of a declared mention.")
    confidence: Literal["high", "medium", "low"]
    evidence: str = Field(
        description="A verbatim span copied from the chunk that states this relation. "
        "Do not paraphrase; the span is checked against the source text."
    )


class Extraction(BaseModel):
    mentions: list[Mention]
    relations: list[Relation]


# ---------------------------------------------------------------------------
# Validation the JSON schema cannot express
# ---------------------------------------------------------------------------

DropReason = Literal["unknown_endpoint", "type_signature", "evidence_not_found", "self_loop", "unknown_risk_topic"]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def validate(extraction: Extraction, chunk_text: str) -> tuple[list[Relation], Counter[str]]:
    """Return the relations worth keeping, plus a per-reason count of what was dropped.

    Constrained decoding guarantees the *shape* of the output and the *vocabulary* of the
    type fields. It cannot check that a relation's endpoints were actually declared, that
    the endpoint types match the relation's signature, or that the quoted evidence exists
    in the source. Those three are the whole of this function.
    """
    haystack = _norm(chunk_text)
    types = {m.name: m.type for m in extraction.mentions}
    kept: list[Relation] = []
    dropped: Counter[str] = Counter()

    for rel in extraction.relations:
        if rel.subject not in types or rel.object not in types:
            dropped["unknown_endpoint"] += 1
            continue
        if rel.subject == rel.object:
            dropped["self_loop"] += 1
            continue
        if (types[rel.subject], types[rel.object]) != ALLOWED_EDGES[rel.predicate]:
            dropped["type_signature"] += 1
            continue
        # Cheap hallucination filter: a fabricated relation rarely carries a real span.
        if _norm(rel.evidence) not in haystack:
            dropped["evidence_not_found"] += 1
            continue
        if types[rel.object] is EntityType.RISK_TOPIC and rel.object not in RISK_TOPICS:
            dropped["unknown_risk_topic"] += 1
            continue
        kept.append(rel)

    return kept, dropped


def ontology_prompt_block() -> str:
    """The ontology section of the extraction prompt, rendered from the definitions above."""
    lines = ["ENTITY TYPES:"]
    lines += [f"- {t.value}" for t in EntityType]
    lines.append("")
    lines.append("RELATION TYPES (subject type -> object type):")
    for rel, (src, dst) in ALLOWED_EDGES.items():
        lines.append(f"- {rel.value} ({src.value} -> {dst.value}): {RELATION_DOCS[rel]}")
    lines.append("")
    lines.append("RiskTopic mentions MUST use one of these exact strings, or be omitted:")
    lines += [f"- {t}" for t in RISK_TOPICS]
    return "\n".join(lines)


if __name__ == "__main__":
    # Self-check: the ontology has to be internally consistent or everything downstream lies.
    assert set(ALLOWED_EDGES) == set(RelationType), "every relation needs a type signature"
    assert set(RELATION_DOCS) == set(RelationType), "every relation needs prompt documentation"
    assert set(RELATION_PHRASE) == set(RelationType), "every relation needs a readable phrase"
    assert set(RELATION_NOUN) == set(RelationType), "every relation needs an inbound noun"
    assert len(EntityType) == 7 and len(RelationType) == 14

    chunk = "Broadcom Inc. acquired VMware, Inc. in a transaction valued at $69 billion."
    good = Extraction(
        mentions=[
            Mention(name="Broadcom Inc.", type=EntityType.COMPANY),
            Mention(name="VMware, Inc.", type=EntityType.COMPANY),
        ],
        relations=[
            Relation(subject="Broadcom Inc.", predicate=RelationType.ACQUIRED, object="VMware, Inc.",
                     confidence="high", evidence="Broadcom Inc. acquired VMware, Inc."),
        ],
    )
    kept, dropped = validate(good, chunk)
    assert len(kept) == 1 and not dropped, (kept, dropped)

    bad = Extraction(
        mentions=[
            Mention(name="Broadcom Inc.", type=EntityType.COMPANY),
            Mention(name="VMware, Inc.", type=EntityType.COMPANY),
            Mention(name="California", type=EntityType.LOCATION),
        ],
        relations=[
            # evidence never appears in the chunk
            Relation(subject="Broadcom Inc.", predicate=RelationType.ACQUIRED, object="VMware, Inc.",
                     confidence="high", evidence="Broadcom purchased Intel's foundry unit."),
            # Company -> Location is not ACQUIRED's signature
            Relation(subject="Broadcom Inc.", predicate=RelationType.ACQUIRED, object="California",
                     confidence="low", evidence="Broadcom Inc. acquired"),
            # endpoint was never declared as a mention
            Relation(subject="Broadcom Inc.", predicate=RelationType.SUPPLIES, object="Apple",
                     confidence="low", evidence="Broadcom Inc. acquired"),
            # a company cannot acquire itself
            Relation(subject="Broadcom Inc.", predicate=RelationType.ACQUIRED, object="Broadcom Inc.",
                     confidence="low", evidence="Broadcom Inc. acquired"),
        ],
    )
    kept, dropped = validate(bad, chunk)
    assert kept == [], kept
    assert dropped == Counter(
        {"evidence_not_found": 1, "type_signature": 1, "unknown_endpoint": 1, "self_loop": 1}
    ), dropped

    assert "DIRECTOR_OF (Person -> Company)" in ontology_prompt_block()
    print("ontology self-check passed:", len(EntityType), "entity types,", len(RelationType), "relations")

"""Qdrant filter builders for specialty preference and strict case isolation."""

from __future__ import annotations

from typing import List, Optional

from qdrant_client.http import models as qm

SHARED_CASE_ID = "shared"
CASE_SOURCE_TYPES = ("case_attachment", "case_text")


def _empty_or_shared_case_id() -> List:
    """Conditions that mean 'not owned by a patient case'."""
    return [
        qm.IsEmptyCondition(is_empty=qm.PayloadField(key="case_id")),
        qm.IsNullCondition(is_null=qm.PayloadField(key="case_id")),
        qm.FieldCondition(key="case_id", match=qm.MatchValue(value="")),
        qm.FieldCondition(key="case_id", match=qm.MatchValue(value=SHARED_CASE_ID)),
    ]


def case_isolation_filter(case_id: Optional[str]) -> qm.Filter:
    """Strict isolation: shared KB OR current case only.

    Critical rule: ``case_attachment`` / ``case_text`` are NEVER treated as shared,
    even if ``case_id`` is missing (legacy leak). They must match the current case_id.
    """
    shared_kb = qm.Filter(
        must=[
            # Not a case-scoped source type
            qm.Filter(
                must_not=[
                    qm.FieldCondition(
                        key="source_type",
                        match=qm.MatchValue(value=st),
                    )
                    for st in CASE_SOURCE_TYPES
                ]
            ),
            # And not tagged as case-owned scope
            qm.Filter(
                must_not=[
                    qm.FieldCondition(
                        key="scope",
                        match=qm.MatchValue(value="case"),
                    )
                ]
            ),
            # case_id empty / shared
            qm.Filter(should=_empty_or_shared_case_id()),
        ]
    )

    if not case_id:
        # No active case → shared KB only
        return shared_kb

    current_case = qm.Filter(
        must=[
            qm.FieldCondition(
                key="case_id",
                match=qm.MatchValue(value=str(case_id)),
            )
        ]
    )

    # Also reject any point whose case_id is a *different* non-empty UUID/string
    # (defense in depth when combined with shared_kb should-branch)
    return qm.Filter(
        should=[shared_kb, current_case],
        must_not=[
            # Explicitly exclude other case_ids when the field is present and
            # is neither current nor shared/empty. MatchExcept: field value NOT IN list.
            qm.FieldCondition(
                key="case_id",
                match=qm.MatchExcept(**{"except": [str(case_id), SHARED_CASE_ID, ""]}),
            ),
        ],
    )


def specialty_prefer_filter(specialty: Optional[str]) -> Optional[qm.Filter]:
    """Soft specialty preference — does NOT open the door to foreign case docs.

    Case materials are handled exclusively by case_isolation_filter.
    """
    if not specialty or specialty == "general_internal_medicine":
        return None
    return qm.Filter(
        should=[
            qm.FieldCondition(key="specialty", match=qm.MatchValue(value=specialty)),
            qm.FieldCondition(key="tags", match=qm.MatchValue(value=specialty)),
            qm.IsEmptyCondition(is_empty=qm.PayloadField(key="specialty")),
            qm.IsNullCondition(is_null=qm.PayloadField(key="specialty")),
            qm.FieldCondition(key="specialty", match=qm.MatchValue(value="")),
            # Shared sample with matching section tag only — not all case_attachments
            qm.FieldCondition(
                key="source_type",
                match=qm.MatchValue(value="sample"),
            ),
        ],
    )


def merge_filters(*filters: Optional[qm.Filter]) -> Optional[qm.Filter]:
    """AND-merge Qdrant filters."""
    must: List = []
    for f in filters:
        if f is None:
            continue
        must.append(f)
    if not must:
        return None
    if len(must) == 1:
        return must[0]
    return qm.Filter(must=must)

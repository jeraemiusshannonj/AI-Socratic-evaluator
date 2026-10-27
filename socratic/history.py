"""Write interaction records to Atlas.

Replaces the SQLite logging in the old `main.py`, which wrote to a local
`evaluator.db` with four columns and no student identifier. Those 112 rows
are migrated and kept, tagged `origin: "evaluator"` or `"telemetry"`; rows
written here are tagged `origin: "live"` so the legacy data and new data
stay distinguishable in any query that reports on either.

The schema matches `scripts/migrate_sqlite.py` so both populations are
queryable together, with additions the old schema had no concept of:
whether the critic cited a real passage, and what it cited. Those fields
are what make citation accuracy measurable over actual sessions rather
than only over a fixed eval set.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from . import db

log = logging.getLogger(__name__)

LIVE_ORIGIN = "live"


def concept_family(concept: str) -> str:
  """Strip a parenthetical variant: "Binary Search (Off-by-One)" -> "Binary Search".

  Duplicated from the migration script on purpose -- the migration is a
  one-shot tool that should stay runnable on its own, and this is one line.
  """
  return re.sub(r"\s*\(.*?\)\s*$", "", concept).strip()


def new_session_id() -> str:
  """An opaque id for one student's run through one exercise.

  The original schema had no student identifier at all, which is why the
  entire migrated history sits under the single id "legacy" and why no
  per-student claim can be made about it. Anything logged from here on
  carries one, so `student_concept_history` and `weakest_concepts` mean
  something going forward.
  """
  return uuid.uuid4().hex[:16]


def log_interaction(
    *,
    concept: str,
    is_correct: bool,
    attempts: int,
    student_id: str,
    session_id: str | None = None,
    citations: list[str] | None = None,
    invalid_citations: list[str] | None = None,
    grounded: bool | None = None,
    passage_count: int = 0,
    usage: dict[str, Any] | None = None,
    judged_by: str = "model",
    model_is_correct: bool | None = None,
    model_agreed: bool | None = None,
    tests_passed: int | None = None,
    tests_total: int | None = None,
) -> str | None:
  """Record one evaluation turn. Returns the document id, or None on failure.

  A logging failure must never cost the student their answer, so this
  swallows its exceptions and reports them. The caller has already done the
  expensive work by the time this runs.
  """
  doc = {
      "student_id": student_id,
      "session_id": session_id,
      "concept": concept,
      "concept_family": concept_family(concept),
      "is_correct": bool(is_correct),
      "attempts": int(attempts),
      "ts": datetime.now(timezone.utc),
      "origin": LIVE_ORIGIN,
      "citations": list(citations or []),
      "invalid_citations": list(invalid_citations or []),
      "grounded": bool(grounded) if grounded is not None else None,
      "passage_count": passage_count,
      "usage": usage or {},
      # Whether the verdict came from running the code or from the model,
      # and whether the two agreed. Stored per turn so the agreement rate
      # can be computed over real sessions rather than only over an eval
      # set -- the eval set is fixed, and this is not.
      "judged_by": judged_by,
      "model_is_correct": model_is_correct,
      "model_agreed": model_agreed,
      "tests_passed": tests_passed,
      "tests_total": tests_total,
  }

  try:
    result = db.interactions().insert_one(doc)
    return str(result.inserted_id)
  except Exception as exc:  # noqa: BLE001
    log.warning("Could not log interaction for %r: %s", concept, exc)
    return None

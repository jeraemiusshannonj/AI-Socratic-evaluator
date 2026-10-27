"""Move the SQLite interaction history and the CSV taxonomy into Atlas.

Run once, after bootstrap_db.py:

    python -m scripts.migrate_sqlite

Notes on the data, from inspecting it rather than assuming:

  * evaluator.db holds 95 rows, telemetry.db holds 17. Both have the same
    schema and overlapping timestamps, so they are two runs of the same app,
    not two different things. Both are migrated, tagged by origin.

  * 72 of the 112 rows are logged against the concept "Binary Search", which
    is not a concept in misconceptions.csv or in the app's dropdown -- the
    taxonomy was refined later and the old rows were never backfilled. They
    are kept with their original label and given a `concept_family` so
    structured queries can roll the variants up without inventing a
    specificity the data does not have.

  * There is no student identifier anywhere in the original schema. Every
    legacy row is attributed to the synthetic student "legacy". Do not claim
    per-student personalization on the strength of this data alone.
"""

from __future__ import annotations

import csv
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from pymongo import UpdateOne

from socratic import db

ROOT = Path(__file__).resolve().parent.parent
SQLITE_SOURCES = [("evaluator.db", "evaluator"), ("telemetry.db", "telemetry")]
CSV_PATH = ROOT / "misconceptions.csv"

LEGACY_STUDENT = "legacy"


def concept_family(concept: str) -> str:
  """"Binary Search (Off-by-One)" -> "Binary Search"."""
  return re.sub(r"\s*\(.*?\)\s*$", "", concept).strip()


def parse_ts(raw: str) -> datetime:
  try:
    dt = datetime.fromisoformat(raw)
  except ValueError:
    dt = datetime.strptime(raw.split(".")[0], "%Y-%m-%d %H:%M:%S")
  return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def migrate_interactions() -> int:
  ops = []
  for filename, origin in SQLITE_SOURCES:
    path = ROOT / filename
    if not path.exists():
      print(f"  skip {filename} (not found)")
      continue
    conn = sqlite3.connect(path)
    rows = conn.execute(
        "SELECT id, concept, is_correct, attempts, timestamp FROM interactions"
    ).fetchall()
    conn.close()

    for row_id, concept, is_correct, attempts, ts in rows:
      # Deterministic _id so re-running the migration is idempotent rather
      # than duplicating the whole history.
      doc_id = f"{origin}:{row_id}"
      ops.append(
          UpdateOne(
              {"_id": doc_id},
              {
                  "$set": {
                      "student_id": LEGACY_STUDENT,
                      "concept": concept,
                      "concept_family": concept_family(concept),
                      "is_correct": bool(is_correct),
                      "attempts": int(attempts),
                      "ts": parse_ts(ts),
                      "origin": origin,
                  }
              },
              upsert=True,
          )
      )
    print(f"  {filename}: {len(rows)} rows queued")

  if not ops:
    return 0
  result = db.interactions().bulk_write(ops, ordered=False)
  return result.upserted_count + result.modified_count


def migrate_taxonomy() -> int:
  if not CSV_PATH.exists():
    print(f"  {CSV_PATH.name} not found; skipping taxonomy")
    return 0

  ops = []
  families: dict[str, list[str]] = {}
  with CSV_PATH.open(newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
      concept = row["Concept"].strip()
      families.setdefault(concept_family(concept), []).append(concept)
      ops.append(
          UpdateOne(
              {"concept": concept},
              {
                  "$set": {
                      "concept": concept,
                      "concept_family": concept_family(concept),
                      "misconception": row["Misconception"].strip(),
                      # Signature and contract pin what the generated
                      # function is called and what it must do. Without
                      # them each generation invents its own interface --
                      # one Linear Search returns a bool, the next an
                      # index -- and no executable test case can assert
                      # anything about either.
                      "signature": (row.get("Signature") or "").strip(),
                      "contract": (row.get("Contract") or "").strip(),
                  },
                  "$addToSet": {"aliases": concept_family(concept)},
              },
              upsert=True,
          )
      )

  db.misconceptions().bulk_write(ops, ordered=False)
  print(f"  taxonomy: {len(ops)} concepts")
  for family, variants in sorted(families.items()):
    if len(variants) > 1:
      print(f"    family '{family}' -> {len(variants)} variants")
  return len(ops)


def report_gaps() -> None:
  """Name the mismatches out loud instead of letting them fail silently."""
  logged = set(db.interactions().distinct("concept"))
  known = set(db.misconceptions().distinct("concept"))
  orphans = sorted(logged - known)
  if orphans:
    print("\n  Logged concepts with no taxonomy entry:")
    for concept in orphans:
      n = db.interactions().count_documents({"concept": concept})
      family = concept_family(concept)
      resolves = db.misconceptions().count_documents({"concept_family": family})
      print(
          f"    {concept!r}: {n} rows"
          f" -> family {family!r} matches {resolves} taxonomy entries"
      )


def main() -> int:
  print("Connecting to Atlas...")
  print(" ", db.ping())

  print("\nMigrating interaction history:")
  n = migrate_interactions()
  print(f"  wrote {n} interaction documents")

  print("\nMigrating misconception taxonomy:")
  migrate_taxonomy()

  report_gaps()

  print("\nCollection counts:")
  for name in ("interactions", "misconceptions", "cs_chunks", "test_cases"):
    print(f"  {name:16s} {db.db()[name].count_documents({})}")
  return 0


if __name__ == "__main__":
  sys.exit(main())

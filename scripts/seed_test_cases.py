"""Load data/test_cases.json into the test_cases collection.

    python -m scripts.seed_test_cases
    python -m scripts.seed_test_cases --check    # validate, write nothing

The file on disk is the source of truth; this mirrors it into Atlas so the
structured retrieval path can serve cases by concept the same way it serves
the taxonomy. Deterministic ids keep it idempotent.

`--check` validates without a database, which is the useful mode in CI and
before a commit: it catches a case naming a concept that is not in the
taxonomy, a malformed kind, or a concept left with no case that actually
targets its misconception.
"""

from __future__ import annotations

import argparse
import sys

from socratic import config, verify

VALID_KINDS = {"value", "terminates", "raises", "operations"}

# Phrases in a rationale that mark a case as the one written to fail on the
# concept's specific misconception. Every concept needs at least one.
TARGET_MARKERS = ("catches the misconception", "catches what is catchable")


def case_id(case: dict, index: int) -> str:
  return f"{case['concept']}#{index}"


def validate(cases: list[dict]) -> list[str]:
  problems: list[str] = []
  by_concept: dict[str, list[dict]] = {}

  for index, case in enumerate(cases):
    where = f"case {index} ({case.get('name', 'unnamed')!r})"

    for field in ("concept", "name", "call", "kind", "rationale"):
      if not case.get(field):
        problems.append(f"{where}: missing {field}")

    kind = case.get("kind")
    if kind and kind not in VALID_KINDS:
      problems.append(f"{where}: unknown kind {kind!r}")

    if kind == "operations":
      if not case.get("counts_from"):
        problems.append(f"{where}: operations case needs counts_from")
      if "reads_under" not in case and "writes_under" not in case:
        problems.append(
            f"{where}: operations case needs reads_under or writes_under")
    if kind in {"value", "operations"} and "expected" not in case:
      problems.append(f"{where}: {kind} case needs expected")

    by_concept.setdefault(case.get("concept", ""), []).append(case)

  for concept, group in sorted(by_concept.items()):
    targeted = [
        c for c in group
        if any(m in c.get("rationale", "").lower() for m in TARGET_MARKERS)
    ]
    if not targeted:
      problems.append(
          f"{concept}: no case marked as catching the misconception. Every "
          f"concept needs one, or the suite only proves the code runs.")
    if len(group) < 3:
      problems.append(f"{concept}: only {len(group)} case(s)")

  return problems


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--check", action="store_true",
                      help="validate the file without touching the database")
  args = parser.parse_args()

  cases = verify.load_cases()
  if not cases:
    print(f"no cases found at {verify.DATA_PATH}")
    return 1

  coverage = verify.coverage()
  print(f"{len(cases)} cases across {len(coverage)} concepts")

  problems = validate(cases)
  if problems:
    print(f"\n{len(problems)} problem(s):")
    for problem in problems:
      print(f"  - {problem}")
    return 2

  for concept in sorted(coverage):
    print(f"  {coverage[concept]:>2}  {concept}")

  if args.check:
    print("\n--check: nothing written")
    return 0

  from pymongo import UpdateOne

  from socratic import db

  print(f"\nmongo: {db.ping()}")

  known = set(db.misconceptions().distinct("concept"))
  unknown = sorted({c["concept"] for c in cases} - known)
  if unknown:
    print("\ncases name concepts with no taxonomy entry:")
    for concept in unknown:
      print(f"  - {concept}")
    print("run `python -m scripts.migrate_sqlite` first")
    return 3

  counters: dict[str, int] = {}
  operations = []
  valid_ids = set()
  for case in cases:
    concept = case["concept"]
    index = counters.get(concept, 0)
    counters[concept] = index + 1
    doc = dict(case)
    doc["_id"] = case_id(case, index)
    doc["case_index"] = index
    valid_ids.add(doc["_id"])
    operations.append(
        UpdateOne({"_id": doc["_id"]}, {"$set": doc}, upsert=True))

  result = db.test_cases().bulk_write(operations, ordered=False)
  removed = db.test_cases().delete_many(
      {"_id": {"$nin": sorted(valid_ids)}}).deleted_count

  print(f"upserted {len(result.upserted_ids or {})}, "
        f"modified {result.modified_count}, pruned {removed}")
  print(f"{db.test_cases().count_documents({})} cases now in "
        f"{config.MONGO_DB}.{config.COLL_TEST_CASES}")
  return 0


if __name__ == "__main__":
  sys.exit(main())

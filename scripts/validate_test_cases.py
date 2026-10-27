"""Check that the test cases actually discriminate.

    python -m scripts.validate_test_cases

Runs every case in data/test_cases.json against both reference
implementations in eval/references.py:

  correct implementation -> every case must pass. A case that fails here
                            is broken, not strict.
  buggy implementation   -> the case marked as catching that concept's
                            misconception must fail. If it passes, the
                            suite proves only that the code runs.

Needs no credentials and touches no network: it executes Python locally
through socratic.execute. Worth running before any commit that changes
data/test_cases.json or misconceptions.csv.

This found five problems in the first draft of the case file. Four cases
compared a tuple against a JSON list, so they failed correct
implementations for a reason that had nothing to do with the code. And
every heap-sort case passed the buggy implementation, because 1-indexed
child arithmetic still sorts roughly 80% of random inputs -- the hand
-picked examples all landed in the 80%.
"""

from __future__ import annotations

import argparse
import sys

from eval import references
from socratic import verify

# A case whose rationale contains one of these is the one written to fail
# on the concept's misconception.
TARGET_MARKERS = ("catches the misconception", "catches what is catchable")


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--concept", help="check a single concept")
  parser.add_argument("-v", "--verbose", action="store_true",
                      help="list every case, not only failures")
  args = parser.parse_args()

  coverage = verify.coverage()
  concepts = [args.concept] if args.concept else sorted(coverage)
  if args.concept and args.concept not in coverage:
    print(f"no cases for {args.concept!r}")
    return 1

  print(f"{len(concepts)} concept(s), {sum(coverage.values())} cases total\n")

  problems: list[str] = []
  for concept in concepts:
    cases = verify.load_cases(concept)
    targeted = [
        c["name"] for c in cases
        if any(m in c.get("rationale", "").lower() for m in TARGET_MARKERS)
    ]

    if concept not in references.CORRECT:
      problems.append(f"{concept}: no reference implementation")
      print(f"SKIP {concept} -- no reference implementation")
      continue

    good = verify.verify(references.CORRECT[concept], concept)
    bad = verify.verify(references.BUGGY[concept], concept)
    failed_on_bug = {c.name for c in bad.cases if not c.passed}
    caught = bool(targeted) and all(t in failed_on_bug for t in targeted)

    status = "OK  " if (good.all_passed and caught) else "BAD "
    print(f"{status} {concept}")
    print(f"       correct: {good.passed}/{good.total} pass    "
          f"buggy: {bad.passed}/{bad.total} pass "
          f"({len(failed_on_bug)} caught)")

    if args.verbose:
      for case in good.cases:
        print(f"         {'pass' if case.passed else 'FAIL'}  {case.name}")

    if not good.all_passed:
      problems.append(f"{concept}: the correct implementation fails cases")
      for case in good.cases:
        if not case.passed:
          print(f"         FALSE FAILURE  {case.name}: {case.detail}")

    if not targeted:
      problems.append(f"{concept}: no case targets the misconception")
    elif not caught:
      problems.append(f"{concept}: the buggy implementation passes "
                      f"the targeted case")
      for name in targeted:
        if name not in failed_on_bug:
          print(f"         NOT CAUGHT     {name}")

  print()
  if problems:
    print(f"{len(problems)} problem(s):")
    for problem in problems:
      print(f"  - {problem}")
    return 1

  print("every concept: the correct implementation passes and the "
        "misconception is caught")
  return 0


if __name__ == "__main__":
  sys.exit(main())

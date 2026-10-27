"""Run executable test cases against a student's code.

The critic judges code by reading it, which is exactly the thing language
models are least reliable at: a plausible-looking fix that does not work is
the failure mode. Running the code answers the question instead of
estimating it, and the answer goes into the critic's prompt as fact rather
than as something for it to assess.

Each case is a `setup` statement and a `call` expression evaluated in the
namespace the student's code defines. Expressions rather than argument
lists because three of the twenty-two concepts are not plain functions --
one defines a class, two take a sqlite3 connection -- and an args/expected
schema cannot express those without a special case per concept.

Four kinds of case:

  value       the call must equal `expected`
  terminates  the call must finish at all; this is what catches the
              binary-search off-by-one, which loops forever rather than
              returning anything wrong
  raises      the call must raise `expected` (an exception type name)
  operations  the call must equal `expected` AND touch the input fewer
              than `reads_under` / `writes_under` times. The only way to
              detect a missing early exit or a swap in the wrong loop:
              both return a perfectly correct answer and differ only in
              how much work they did

A case marked `isolate` runs in a process of its own. Several of these
bugs hang rather than return, and one hanging case in a shared batch
would report every later case as untested.

Execution reuses `socratic.execute`, so the same caveat applies: this is
isolation against accidents, not against an attacker.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from . import execute

log = logging.getLogger(__name__)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "test_cases.json"

# Per-case wall clock. A case that is meant to terminate gets the same
# budget as any other; only `terminates` cases read a timeout as a failure.
CASE_TIMEOUT = 6.0


@dataclass
class CaseResult:
  name: str
  passed: bool
  detail: str = ""
  rationale: str = ""

  def to_dict(self) -> dict[str, Any]:
    return {
        "name": self.name,
        "passed": self.passed,
        "detail": self.detail,
        "rationale": self.rationale,
    }


@dataclass
class VerificationResult:
  concept: str
  cases: list[CaseResult] = field(default_factory=list)
  ran: bool = True
  error: str = ""

  @property
  def passed(self) -> int:
    return sum(1 for c in self.cases if c.passed)

  @property
  def total(self) -> int:
    return len(self.cases)

  @property
  def all_passed(self) -> bool:
    return self.ran and bool(self.cases) and self.passed == self.total

  def to_dict(self) -> dict[str, Any]:
    return {
        "concept": self.concept,
        "ran": self.ran,
        "error": self.error,
        "passed": self.passed,
        "total": self.total,
        "all_passed": self.all_passed,
        "cases": [c.to_dict() for c in self.cases],
    }

  def as_prompt_block(self) -> str:
    """How the result is shown to the critic."""
    if not self.ran:
      return f"Test cases could not be run ({self.error})."
    if not self.cases:
      return "No test cases exist for this concept."

    lines = [f"{self.passed} of {self.total} test cases pass."]
    for case in self.cases:
      mark = "PASS" if case.passed else "FAIL"
      lines.append(f"  [{mark}] {case.name}")
      if not case.passed and case.detail:
        lines.append(f"         {case.detail}")
    return "\n".join(lines)


def load_cases(concept: str | None = None) -> list[dict[str, Any]]:
  """Read the case file. Returns every case, or those for one concept."""
  if not DATA_PATH.exists():
    log.warning("No test case file at %s", DATA_PATH)
    return []
  data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
  cases = data.get("cases", [])
  if concept is None:
    return cases
  return [c for c in cases if c.get("concept") == concept]


# The driver runs inside the student's process. It is written as source
# rather than imported because the student's code is executed as a
# standalone script with no package available to it.
_DRIVER = '''

# ---- test harness (appended; not part of the student's code) --------------
import json as _json
import traceback as _traceback


class CountingList(list):
    """A list that records how many times it was indexed.

    Needed for the early-exit cases: a bubble sort without the swapped
    flag returns a perfectly correct answer and is only distinguishable
    by how much work it did.
    """

    def __init__(self, *a):
        super().__init__(*a)
        self.reads = 0
        self.writes = 0

    def __getitem__(self, i):
        self.reads += 1
        return super().__getitem__(i)

    def __setitem__(self, i, v):
        self.writes += 1
        return super().__setitem__(i, v)


_CASES = _json.loads({cases_json!r})
_RESULTS = []

for _case in _CASES:
    _name = _case["name"]
    _kind = _case.get("kind", "value")
    _ns = dict(globals())
    try:
        if _case.get("setup"):
            exec(_case["setup"], _ns)
        _value = eval(_case["call"], _ns)

        if _kind == "raises":
            _RESULTS.append({{
                "name": _name, "passed": False,
                "detail": "expected {{}} but the call returned {{!r}}".format(
                    _case["expected"], _value),
            }})
        elif _kind == "terminates":
            _RESULTS.append({{"name": _name, "passed": True, "detail": ""}})
        elif _kind == "operations":
            _probe = _ns.get(_case["counts_from"])
            _reads = getattr(_probe, "reads", None)
            _writes = getattr(_probe, "writes", None)
            _n = len(_case["expected"])
            if _value != _case["expected"]:
                _RESULTS.append({{
                    "name": _name, "passed": False,
                    "detail": "returned {{!r}}, expected {{!r}}".format(
                        _value, _case["expected"]),
                }})
            elif _reads is None:
                _RESULTS.append({{
                    "name": _name, "passed": False,
                    "detail": (
                        "could not count operations -- the function copied "
                        "its input instead of working on it in place"),
                }})
            elif ("reads_under" in _case
                  and _reads >= _case["reads_under"]):
                _RESULTS.append({{
                    "name": _name, "passed": False,
                    "detail": (
                        "correct, but read the list {{}} times for {{}} "
                        "elements (under {{}} expected)").format(
                            _reads, _n, _case["reads_under"]),
                }})
            elif ("writes_under" in _case
                  and _writes >= _case["writes_under"]):
                _RESULTS.append({{
                    "name": _name, "passed": False,
                    "detail": (
                        "correct, but wrote to the list {{}} times for {{}} "
                        "elements (under {{}} expected)").format(
                            _writes, _n, _case["writes_under"]),
                }})
            else:
                _RESULTS.append({{"name": _name, "passed": True, "detail": ""}})
        else:
            _passed = _value == _case["expected"]
            _RESULTS.append({{
                "name": _name, "passed": _passed,
                "detail": "" if _passed else
                          "returned {{!r}}, expected {{!r}}".format(
                              _value, _case["expected"]),
            }})

    except Exception as _exc:
        _got = type(_exc).__name__
        if _kind == "raises" and _got == _case["expected"]:
            _RESULTS.append({{"name": _name, "passed": True, "detail": ""}})
        else:
            _RESULTS.append({{
                "name": _name, "passed": False,
                "detail": "{{}}: {{}}".format(_got, _exc),
            }})

print("__TEST_RESULTS__" + _json.dumps(_RESULTS))
'''


def verify(
    code: str, concept: str, *, cases: Sequence[dict] | None = None
) -> VerificationResult:
  """Run the test cases for `concept` against `code`."""
  selected = list(cases) if cases is not None else load_cases(concept)
  if not selected:
    return VerificationResult(concept=concept, cases=[], ran=True)

  rationales = {c["name"]: c.get("rationale", "") for c in selected}

  # Cases that may hang run on their own: one infinite loop would
  # otherwise take the whole batch down and report every later case as
  # untested rather than as passing or failing on its own merits.
  def _alone(case: dict) -> bool:
    return case.get("kind") == "terminates" or bool(case.get("isolate"))

  batched = [c for c in selected if not _alone(c)]
  solo = [c for c in selected if _alone(c)]

  results: list[CaseResult] = []
  if batched:
    results.extend(_run_batch(code, batched, rationales))
  for case in solo:
    results.extend(_run_batch(code, [case], rationales))

  order = {c["name"]: i for i, c in enumerate(selected)}
  results.sort(key=lambda r: order.get(r.name, 0))
  return VerificationResult(concept=concept, cases=results)


def _run_batch(
    code: str, cases: Sequence[dict], rationales: dict[str, str]
) -> list[CaseResult]:
  driver = _DRIVER.format(cases_json=json.dumps(list(cases)))
  outcome = execute.run_student_code(code, driver, timeout=CASE_TIMEOUT)

  marker = "__TEST_RESULTS__"
  if marker in outcome.output:
    payload = outcome.output.split(marker, 1)[1].splitlines()[0]
    try:
      raw = json.loads(payload)
      return [
          CaseResult(
              name=item["name"],
              passed=item["passed"],
              detail=item.get("detail", ""),
              rationale=rationales.get(item["name"], ""),
          )
          for item in raw
      ]
    except json.JSONDecodeError:
      log.warning("Unparseable test results: %s", payload[:200])

  # No marker: the process died before printing. For a `terminates` case
  # that is the finding, not an error.
  if outcome.timed_out:
    return [
        CaseResult(
            name=case["name"],
            passed=False,
            detail=(
                f"did not finish within {CASE_TIMEOUT:g}s -- the call does "
                f"not terminate"
            ),
            rationale=rationales.get(case["name"], ""),
        )
        for case in cases
    ]

  detail = outcome.output.strip().splitlines()
  summary = detail[-1] if detail else "no output"
  return [
      CaseResult(
          name=case["name"],
          passed=False,
          detail=f"the code did not run: {summary[:200]}",
          rationale=rationales.get(case["name"], ""),
      )
      for case in cases
  ]


def coverage() -> dict[str, int]:
  """Cases per concept, for the README table and for seeding checks."""
  counts: dict[str, int] = {}
  for case in load_cases():
    counts[case["concept"]] = counts.get(case["concept"], 0) + 1
  return counts

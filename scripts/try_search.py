"""Exercise both retrieval paths from the command line.

    python -m scripts.try_search "why does high = mid loop forever"
    python -m scripts.try_search "off by one" --concept "Binary Search (Off-by-One)"
    python -m scripts.try_search --stats
    python -m scripts.try_search --structured "Binary Search (Off-by-One)"
    python -m scripts.try_search --selftest

This exists because the first thing worth knowing about a retrieval system
is whether it returns the right passage, and that question cannot be
answered by counting documents. `--selftest` runs a fixed set of questions
whose correct source document is known in advance and reports how often the
right one came back, which is the number the README needs.
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from socratic import config, retrieval

# Questions paired with the source a correct answer must come from. Chosen so
# the expected source is unambiguous: each one names a mechanism that only
# one document in the corpus explains.
SELFTEST: tuple[tuple[str, str, str | None], ...] = (
    (
        "why does setting high = mid cause an infinite loop in binary search",
        "opendsa:Searching_BinarySearch",
        "Binary Search (Off-by-One)",
    ),
    (
        "what happens when a mutable list is used as a default argument",
        "cpython:",
        "Python Default Arguments",
    ),
    (
        "how do I find the parent and child of a node in a heap stored in an array",
        "opendsa:",
        "Heap Sort (0-Indexed Parent/Child)",
    ),
    (
        "what does a LEFT OUTER JOIN return when there is no matching row",
        "sqlalchemy:",
        "SQL Left Join",
    ),
    (
        "how does GROUP BY work with aggregate functions and HAVING",
        "sqlalchemy:",
        "SQL Group By",
    ),
    (
        "why does a hash table need to handle collisions",
        "opendsa:Hashing",
        "Hash Tables",
    ),
    (
        "what is a base case and why does recursion need one",
        "opendsa:",
        "Recursion",
    ),
    (
        "why track visited vertices during a breadth-first traversal",
        "opendsa:Graph",
        "Graph Traversal (BFS Visited Tracking)",
    ),
    (
        "is a tuple mutable in Python",
        "cpython:",
        "Python Object Mutability",
    ),
    (
        "why does bubble sort need a swapped flag to exit early",
        "opendsa:",
        "Bubble Sort (Early Exit)",
    ),
)


def _show(passages: list[retrieval.Passage], *, width: int = 76) -> None:
  if not passages:
    print("  no passages returned")
    return
  for i, p in enumerate(passages, 1):
    print(f"\n  {i}. score {p.score:.4f}  [{p.chunk_id}]")
    if p.section:
      print(f"     section: {p.section}")
    print(f"     source : {p.source_title}")
    if p.concept_tags:
      print(f"     tags   : {', '.join(p.concept_tags)}")
    body = " ".join(p.text.split())[:360]
    for line in textwrap.wrap(body, width=width):
      print(f"     {line}")


def run_query(query: str, *, concept: str | None, k: int,
              min_score: float) -> None:
  print(f"query  : {query!r}")
  print(f"concept: {concept or '(unfiltered)'}")
  print(f"k      : {k}   min_score: {min_score}")
  passages = retrieval.semantic_search(
      query, k=k, concept=concept, min_score=min_score)
  print(f"\n{len(passages)} passages:")
  _show(passages)


def run_structured(concept: str) -> None:
  """Path B, for contrast: exact lookups, no embedding involved."""
  print(f"=== structured lookups for {concept!r} ===\n")

  misconception = retrieval.lookup_misconception(concept)
  print("lookup_misconception:")
  if misconception:
    for line in textwrap.wrap(
        misconception.get("misconception", ""), width=74):
      print(f"  {line}")
    if misconception.get("concept") != concept:
      print(f"  (matched via alias on {misconception.get('concept')!r})")
  else:
    print("  no taxonomy entry -- this is one of the 11 missing concepts")

  tests = retrieval.lookup_test_cases(concept)
  print(f"\nlookup_test_cases: {len(tests)}")

  history = retrieval.student_concept_history("legacy", concept)
  print(f"\nstudent_concept_history('legacy', {concept!r}):")
  print(f"  total sessions: {history['total_sessions']}")
  for row in history["per_concept"]:
    print(f"  {row['concept']}: {row['sessions']} sessions, "
          f"{row['success_rate']:.1%} solved, "
          f"{row['avg_attempts']} avg attempts")

  print("\nweakest_concepts('legacy'):")
  for row in retrieval.weakest_concepts("legacy"):
    print(f"  {row['concept']}: {row['success_rate']:.1%} solved "
          f"over {row['sessions']} sessions")


def run_selftest(k: int) -> int:
  """Does the right document come back? Reports hit@1 and hit@k."""
  print(f"Retrieval self-test: {len(SELFTEST)} questions, k={k}\n")
  hit1 = hitk = 0
  failures: list[str] = []

  for query, expected_prefix, concept in SELFTEST:
    passages = retrieval.semantic_search(query, k=k, concept=concept)
    ids = [p.chunk_id for p in passages]
    top1 = bool(ids) and ids[0].startswith(expected_prefix)
    topk = any(i.startswith(expected_prefix) for i in ids)
    hit1 += top1
    hitk += topk

    mark = "PASS" if topk else "MISS"
    rank = next(
        (n for n, i in enumerate(ids, 1) if i.startswith(expected_prefix)),
        None,
    )
    position = f"rank {rank}" if rank else "not in top-k"
    print(f"  {mark}  {position:<12} expected {expected_prefix!r}")
    print(f"        {query[:68]}")
    if not topk:
      failures.append(f"{query!r}\n          wanted {expected_prefix!r}, "
                      f"got {ids[:3]}")

  total = len(SELFTEST)
  print(f"\nhit@1: {hit1}/{total} ({hit1/total:.0%})")
  print(f"hit@{k}: {hitk}/{total} ({hitk/total:.0%})")

  if failures:
    print("\nmisses:")
    for failure in failures:
      print(f"  - {failure}")

  # Not a hard gate: these are hand-written expectations, and a miss can
  # mean the expectation was wrong as easily as the retrieval was. It is a
  # number to look at, not a build to break.
  return 0


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("query", nargs="?", help="natural-language question")
  parser.add_argument("--concept", default=None,
                      help="scope the search to one concept tag")
  parser.add_argument("-k", type=int, default=config.DEFAULT_K)
  parser.add_argument("--min-score", type=float, default=0.0)
  parser.add_argument("--structured", metavar="CONCEPT",
                      help="run the structured lookups for a concept")
  parser.add_argument("--stats", action="store_true",
                      help="print collection counts")
  parser.add_argument("--selftest", action="store_true",
                      help="run the fixed question set and report hit rate")
  args = parser.parse_args()

  if args.stats:
    for key, value in retrieval.corpus_stats().items():
      print(f"  {key:<20} {value}")
    return 0

  if args.structured:
    run_structured(args.structured)
    return 0

  if args.selftest:
    return run_selftest(args.k)

  if not args.query:
    parser.print_help()
    return 1

  run_query(args.query, concept=args.concept, k=args.k,
            min_score=args.min_score)
  return 0


if __name__ == "__main__":
  sys.exit(main())

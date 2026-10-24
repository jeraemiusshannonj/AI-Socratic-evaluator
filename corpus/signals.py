"""Per-chunk concept filtering.

`sources.py` tags whole documents. That is the right granularity for saying
"this page is about hashing", and the wrong granularity for the
`concept_tags` filter on the vector index.

CPython's `faq/programming.rst` is a legitimate source for Python Lists,
Python Object Mutability and Python Default Arguments. It is also 71,000
characters and 56 chunks, most of which are about threading, string
formatting, import semantics and performance. Tagging all 56 with all three
concepts means a search scoped to "Python Lists" can return a chunk about
`sys.setrecursionlimit`, score it highly because the filter admitted it,
and hand the critic a passage with nothing to cite. That failure reads as a
bad embedding model. It is bad metadata.

So a document-level tag is treated as a *candidate*, and each chunk has to
earn it:

  * one **strong** term is enough -- these are phrases that do not appear
    by accident ("mutable default", "breadth-first", "GROUP BY")
  * otherwise two distinct **weak** terms are needed -- individually common
    words ("list", "swap", "index") that only mean something together

A chunk that earns nothing keeps an empty `concept_tags` and is still
ingested. It stays reachable by unfiltered semantic search, which is the
honest treatment: the text is real corpus content, it just is not evidence
for any particular concept.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Signals:
  """Evidence that a chunk is actually about a concept."""

  strong: tuple[str, ...] = ()
  weak: tuple[str, ...] = ()
  # Minimum distinct weak terms needed when no strong term is present.
  weak_threshold: int = 2
  _compiled: dict = field(default_factory=dict, compare=False, repr=False)


def _rx(term: str) -> re.Pattern[str]:
  """Word-boundary match, case-insensitive, tolerant of punctuation.

  `count(` and `[]` are legitimate terms and have no word boundary, so the
  boundary is only applied where the term starts/ends alphanumerically.
  """
  escaped = re.escape(term)
  prefix = r"\b" if term[:1].isalnum() else ""
  suffix = r"\b" if term[-1:].isalnum() else ""
  return re.compile(prefix + escaped + suffix, re.IGNORECASE)


_CACHE: dict[str, re.Pattern[str]] = {}


def _match(term: str, text: str) -> bool:
  pattern = _CACHE.get(term)
  if pattern is None:
    pattern = _CACHE[term] = _rx(term)
  return bool(pattern.search(text))


# ---------------------------------------------------------------------------
# One entry per concept in sources.CONCEPTS.
# ---------------------------------------------------------------------------
CONCEPT_SIGNALS: dict[str, Signals] = {
    "Linear Search": Signals(
        strong=("sequential search", "linear search", "unsuccessful search"),
        weak=("unsorted", "search", "array", "element", "found"),
    ),
    "Binary Search (Off-by-One)": Signals(
        strong=("binary search",),
        weak=("mid", "midpoint", "low", "high", "halve", "half", "sorted"),
    ),
    "Binary Search (Loop Condition)": Signals(
        strong=("binary search",),
        weak=("mid", "low", "high", "loop", "while", "terminate", "range"),
    ),
    "Binary Search (Prerequisite)": Signals(
        strong=("binary search", "sorted array"),
        weak=("sorted", "order", "precondition", "requires", "search"),
    ),
    "Bubble Sort (Early Exit)": Signals(
        strong=("bubble sort", "bubblesort", "exchange sort"),
        weak=("swap", "pass", "adjacent", "already sorted", "best case"),
    ),
    "Selection Sort (Min Index Swap)": Signals(
        strong=("selection sort",),
        weak=("smallest", "minimum", "swap", "unsorted", "pass"),
    ),
    "Insertion Sort (Shifting Bug)": Signals(
        strong=("insertion sort",),
        weak=("shift", "insert", "sorted portion", "sublist", "key"),
    ),
    "Merge Sort (Slice Overhead)": Signals(
        strong=("mergesort", "merge sort"),
        weak=("merge", "sublist", "auxiliary", "temporary", "divide",
              "space"),
    ),
    "Merge Sort (Two-Pointer Comparison)": Signals(
        strong=("mergesort", "merge sort"),
        weak=("merge", "pointer", "compare", "sublist", "index",
              "advance"),
    ),
    "Quick Sort (Partition Off-By-One)": Signals(
        strong=("quicksort", "quick sort"),
        weak=("pivot", "partition", "recursive", "subarray", "index"),
    ),
    "Quick Sort (Lomuto Partition)": Signals(
        strong=("quicksort", "quick sort", "lomuto"),
        weak=("pivot", "partition", "swap", "boundary", "index"),
    ),
    "Heap Sort (0-Indexed Parent/Child)": Signals(
        strong=("heapsort", "heap sort", "siftdown", "sift down",
                "complete binary tree"),
        weak=("heap", "parent", "child", "root", "index", "array"),
    ),
    "Graph Traversal (BFS Visited Tracking)": Signals(
        strong=("breadth-first", "breadth first", "bfs"),
        weak=("graph", "queue", "visited", "traversal", "vertex",
              "neighbor", "adjacent"),
    ),
    "Graph Traversal (DFS Cycle Detection)": Signals(
        strong=("depth-first", "depth first", "dfs", "topological sort",
                "cycle"),
        weak=("graph", "visited", "traversal", "vertex", "recursive",
              "stack", "edge"),
    ),
    "Recursion": Signals(
        strong=("base case", "recursive call", "recursion", "tail recursion"),
        weak=("recursive", "recursively", "call itself", "stack",
              "subproblem"),
    ),
    "Python Lists": Signals(
        # `list` alone is in nearly every Python document, so it is weak and
        # needs company.
        strong=("list comprehension", "list method", "list object",
                "list slice", "lists are mutable"),
        weak=("list", "append", "slice", "index", "len(", "[]", "pop",
              "insert", "sequence", "element"),
        weak_threshold=3,
    ),
    "Python Default Arguments": Signals(
        strong=("default argument", "default value", "mutable default",
                "keyword argument", "default parameter"),
        weak=("default", "argument", "parameter", "shared", "evaluated",
              "def "),
        weak_threshold=3,
    ),
    "Python Object Mutability": Signals(
        strong=("immutable", "mutable", "in place", "in-place",
                "deepcopy", "identity test"),
        weak=("copy", "reference", "alias", "modify", "object", "is ",
              "shared", "tuple"),
        weak_threshold=3,
    ),
    "Hash Tables": Signals(
        strong=("hash table", "hash function", "hashing", "collision",
                "load factor", "open addressing", "probe sequence"),
        weak=("hash", "bucket", "slot", "probe", "key", "dictionary",
              "chaining"),
    ),
    "SQL Left Join": Signals(
        strong=("left outer join", "left join", "outer join", "inner join",
                "join condition", "on clause"),
        weak=("join", "null", "foreign key", "relation", "table", "row",
              "match"),
    ),
    "SQL Group By": Signals(
        strong=("group by", "having", "aggregate function"),
        weak=("aggregate", "count(", "sum(", "avg(", "max(", "min(",
              "grouping", "group"),
    ),
    "Object-Oriented Programming": Signals(
        strong=("inheritance", "polymorphism", "encapsulation", "subclass",
                "abstract data type", "object-oriented", "class definition"),
        weak=("class", "object", "method", "instance", "attribute",
              "interface", "inherit", "self"),
        weak_threshold=3,
    ),
}


def earns(concept: str, text: str, heading_trail: tuple[str, ...] = ()) -> bool:
  """Does this chunk carry evidence for `concept`?

  Headings count double in effect: a heading naming the concept is treated
  as a strong hit, because a section titled "Aggregate functions with GROUP
  BY / HAVING" is about grouping even if the body is mostly a code sample.
  """
  signals = CONCEPT_SIGNALS.get(concept)
  if signals is None:
    # No signal set defined: fall back to trusting the document tag rather
    # than silently dropping a concept someone just added to sources.py.
    return True

  heading_text = " ".join(heading_trail)
  for term in signals.strong:
    if _match(term, heading_text) or _match(term, text):
      return True

  for term in signals.weak:
    if _match(term, heading_text):
      return True

  hits = sum(1 for term in signals.weak if _match(term, text))
  return hits >= signals.weak_threshold


def filter_concepts(
    candidates: tuple[str, ...],
    text: str,
    heading_trail: tuple[str, ...] = (),
) -> list[str]:
  """Narrow a document's concept tags to the ones this chunk supports."""
  return [c for c in candidates if earns(c, text, heading_trail)]


def missing_signal_sets(concepts: tuple[str, ...]) -> list[str]:
  """Concepts in the taxonomy with no signal set defined here."""
  return [c for c in concepts if c not in CONCEPT_SIGNALS]

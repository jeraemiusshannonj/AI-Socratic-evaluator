"""Which documents form the corpus, and which concept each one answers for.

This mapping is hand-curated, and deliberately so. Tagging by keyword match
would put every page containing the word "sort" under all six sorting
concepts, and the point of the `concept_tags` filter on the vector index is
that scoping a search to "Heap Sort (0-Indexed Parent/Child)" returns the
pages about parent/child index arithmetic rather than everything in the
Sorting chapter.

Licensing
---------
Both sources are permissively licensed, which matters because this
repository is public:

  * OpenDSA -- MIT. Covers the prose as well as the code ("this software and
    associated documentation files"). Copyright (c) 2011-2025 Ville
    Karavirta and Cliff Shaffer.
  * CPython documentation -- PSF License Agreement, which permits
    redistribution of modified versions with the copyright notice retained.
    Copyright (c) 2001-2025 Python Software Foundation.

OpenDSA also vendors the "Python for Everybody" chapters by Charles
Severance under `RST/en/PY4E`. Those are CC BY-NC-SA, not MIT -- OpenDSA
cannot relicense someone else's work by including it. The non-commercial
and share-alike terms are both unwanted here, so PY4E is excluded and the
Python language concepts are served from CPython's own documentation
instead. That is the better source anyway: `tutorial/controlflow.rst`
documents the mutable-default-argument trap directly, which is exactly the
misconception the app teaches.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# The 22 concepts the app's topic selector offers (app.py). Kept here as the
# single source of truth so build_corpus can fail loudly when a tag is
# misspelled, rather than producing a corpus nothing retrieves.
# ---------------------------------------------------------------------------
CONCEPTS: tuple[str, ...] = (
    "Linear Search",
    "Binary Search (Off-by-One)",
    "Binary Search (Loop Condition)",
    "Binary Search (Prerequisite)",
    "Bubble Sort (Early Exit)",
    "Selection Sort (Min Index Swap)",
    "Insertion Sort (Shifting Bug)",
    "Merge Sort (Slice Overhead)",
    "Merge Sort (Two-Pointer Comparison)",
    "Quick Sort (Partition Off-By-One)",
    "Quick Sort (Lomuto Partition)",
    "Heap Sort (0-Indexed Parent/Child)",
    "Graph Traversal (BFS Visited Tracking)",
    "Graph Traversal (DFS Cycle Detection)",
    "Recursion",
    "Python Lists",
    "Python Default Arguments",
    "Python Object Mutability",
    "Hash Tables",
    "SQL Left Join",
    "SQL Group By",
    "Object-Oriented Programming",
)


@dataclass(frozen=True)
class Repo:
  """An upstream repository the corpus is built from."""

  key: str
  clone_url: str
  # Subdirectory the documents are addressed relative to.
  root: str
  license_id: str
  attribution: str
  blob_url: str  # formatted with the path relative to `root`
  # Sparse-checkout paths, or () to clone the whole tree shallowly.
  sparse: tuple[str, ...] = ()

  def url_for(self, rel_path: str) -> str:
    return self.blob_url.format(path=rel_path)


OPENDSA = Repo(
    key="opendsa",
    clone_url="https://github.com/OpenDSA/OpenDSA.git",
    root="RST/en",
    license_id="MIT",
    attribution=(
        "OpenDSA eTextbook Project, Copyright (c) 2011-2025 "
        "Ville Karavirta and Cliff Shaffer, MIT License. https://opendsa.org"
    ),
    blob_url="https://github.com/OpenDSA/OpenDSA/blob/master/RST/en/{path}",
)

# OpenDSA's SQL chapter turned out to be interactive JavaScript slideshows
# with no prose at all -- `Database/SQLJoin.rst` is three `inlineav`
# directives and a heading, and converts to 18 characters. The explanatory
# text lives in `AV/Database/Join.js` as slideshow captions, which is not
# retrievable material. SQLAlchemy's narrative documentation covers both of
# our SQL concepts (a dedicated "OUTER and FULL join" section and
# "Aggregate functions with GROUP BY / HAVING") and shows the rendered SQL
# alongside the Python, so a retrieved passage still contains real SQL.
SQLALCHEMY = Repo(
    key="sqlalchemy",
    clone_url="https://github.com/sqlalchemy/sqlalchemy.git",
    root="doc/build",
    license_id="MIT",
    attribution=(
        "SQLAlchemy documentation, Copyright 2005-2026 SQLAlchemy authors "
        "and contributors, MIT License. https://www.sqlalchemy.org/"
    ),
    blob_url=(
        "https://github.com/sqlalchemy/sqlalchemy/blob/main/doc/build/{path}"
    ),
    sparse=("doc/build/tutorial", "doc/build/orm"),
)

CPYTHON = Repo(
    key="cpython",
    clone_url="https://github.com/python/cpython.git",
    root="Doc",
    license_id="PSF-2.0",
    attribution=(
        "Python documentation, Copyright (c) 2001-2025 Python Software "
        "Foundation, PSF License Agreement. https://docs.python.org/3/"
    ),
    blob_url="https://github.com/python/cpython/blob/main/Doc/{path}",
    sparse=("Doc",),
)

REPOS: tuple[Repo, ...] = (OPENDSA, CPYTHON, SQLALCHEMY)


@dataclass(frozen=True)
class Document:
  repo: Repo
  path: str                   # relative to repo.root
  concepts: tuple[str, ...]

  @property
  def source_id(self) -> str:
    """Stable identifier used for `_id` prefixes and citation display."""
    stem = self.path.rsplit(".", 1)[0].replace("/", "_")
    return f"{self.repo.key}:{stem}"

  @property
  def url(self) -> str:
    return self.repo.url_for(self.path)


def _d(repo: Repo, path: str, *concepts: str) -> Document:
  return Document(repo=repo, path=path, concepts=concepts)


# ---------------------------------------------------------------------------
# Searching
# ---------------------------------------------------------------------------
_SEARCH = [
    _d(OPENDSA, "Searching/SearchIntro.rst",
       "Linear Search", "Binary Search (Prerequisite)"),
    _d(OPENDSA, "Searching/UnsortedSearch.rst",
       "Linear Search"),
    # The canonical treatment: sequential search, then binary search with the
    # mid/low/high invariant the off-by-one and loop-condition bugs violate.
    _d(OPENDSA, "Searching/BinarySearch.rst",
       "Linear Search",
       "Binary Search (Off-by-One)",
       "Binary Search (Loop Condition)",
       "Binary Search (Prerequisite)"),
    _d(OPENDSA, "Searching/SortedSearch.rst",
       "Binary Search (Off-by-One)",
       "Binary Search (Loop Condition)",
       "Binary Search (Prerequisite)"),
]

# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------
_SORTING = [
    _d(OPENDSA, "Sorting/SortNotation.rst",
       "Bubble Sort (Early Exit)",
       "Selection Sort (Min Index Swap)",
       "Insertion Sort (Shifting Bug)"),
    _d(OPENDSA, "Sorting/InsertionSort.rst",
       "Insertion Sort (Shifting Bug)"),
    _d(OPENDSA, "Sorting/InSort.rst",
       "Insertion Sort (Shifting Bug)"),
    _d(OPENDSA, "Sorting/BubbleSort.rst",
       "Bubble Sort (Early Exit)"),
    _d(OPENDSA, "Sorting/ExchangeSort.rst",
       "Bubble Sort (Early Exit)",
       "Selection Sort (Min Index Swap)",
       "Insertion Sort (Shifting Bug)"),
    # Why skipping already-sorted passes matters -- the Bubble Sort early-exit
    # misconception is about exactly this cost.
    _d(OPENDSA, "Sorting/SortOpt.rst",
       "Bubble Sort (Early Exit)",
       "Selection Sort (Min Index Swap)",
       "Insertion Sort (Shifting Bug)"),
    _d(OPENDSA, "Sorting/SelectionSort.rst",
       "Selection Sort (Min Index Swap)"),
    # The three quadratic sorts measured against each other. OpenDSA's
    # selection-sort page is short once the interactive code is stripped, and
    # this is where the swap-count argument that the misconception turns on
    # actually lives.
    _d(OPENDSA, "Sorting/SortingEmpirical.rst",
       "Bubble Sort (Early Exit)",
       "Selection Sort (Min Index Swap)",
       "Insertion Sort (Shifting Bug)"),
    # Shellsort is insertion sort over strided sublists, so its page spends
    # most of its length explaining what insertion sort's shifting costs.
    _d(OPENDSA, "Sorting/Shellsort.rst",
       "Insertion Sort (Shifting Bug)"),
    _d(OPENDSA, "Sorting/Mergesort.rst",
       "Merge Sort (Slice Overhead)",
       "Merge Sort (Two-Pointer Comparison)"),
    _d(OPENDSA, "Sorting/MergesortImpl.rst",
       "Merge Sort (Slice Overhead)",
       "Merge Sort (Two-Pointer Comparison)"),
    _d(OPENDSA, "Sorting/Quicksort.rst",
       "Quick Sort (Partition Off-By-One)",
       "Quick Sort (Lomuto Partition)"),
    _d(OPENDSA, "Sorting/Heapsort.rst",
       "Heap Sort (0-Indexed Parent/Child)"),
    # Heaps.rst carries the parent/child index arithmetic itself, which is
    # the whole substance of the heapsort misconception.
    _d(OPENDSA, "Binary/Heaps.rst",
       "Heap Sort (0-Indexed Parent/Child)"),
    _d(OPENDSA, "Binary/CompleteTree.rst",
       "Heap Sort (0-Indexed Parent/Child)"),
]

# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------
_GRAPHS = [
    _d(OPENDSA, "Graph/GraphIntro.rst",
       "Graph Traversal (BFS Visited Tracking)",
       "Graph Traversal (DFS Cycle Detection)"),
    _d(OPENDSA, "Graph/GraphImpl.rst",
       "Graph Traversal (BFS Visited Tracking)",
       "Graph Traversal (DFS Cycle Detection)"),
    _d(OPENDSA, "Graph/GraphTraversal.rst",
       "Graph Traversal (BFS Visited Tracking)",
       "Graph Traversal (DFS Cycle Detection)"),
    # Topological sort is where "have I already visited this" becomes "is
    # there a cycle", which is the DFS misconception.
    _d(OPENDSA, "Graph/GraphTopsort.rst",
       "Graph Traversal (DFS Cycle Detection)"),
]

# ---------------------------------------------------------------------------
# Recursion
# ---------------------------------------------------------------------------
_RECURSION = [
    _d(OPENDSA, "RecurTutor/RecIntro.rst", "Recursion"),
    _d(OPENDSA, "RecurTutor/Trace.rst", "Recursion"),
    _d(OPENDSA, "RecurTutor/Write.rst", "Recursion"),
    _d(OPENDSA, "Binary/RecursiveDS.rst", "Recursion"),
    # Tree traversal is where recursion stops being a trick and starts being
    # the natural shape of the problem, and these pages reason explicitly
    # about base cases.
    _d(OPENDSA, "Binary/BinaryTreeTraversal.rst", "Recursion"),
    _d(OPENDSA, "Binary/WritingTraversals.rst", "Recursion"),
    # The call stack made concrete: what recursion costs and what iteration
    # replaces it with.
    _d(OPENDSA, "List/StackRecur.rst", "Recursion"),
]

# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
_HASHING = [
    _d(OPENDSA, "Hashing/HashIntro.rst", "Hash Tables"),
    _d(OPENDSA, "Hashing/HashFunc.rst", "Hash Tables"),
    _d(OPENDSA, "Hashing/HashFuncExamp.rst", "Hash Tables"),
    _d(OPENDSA, "Hashing/OpenHash.rst", "Hash Tables"),
    _d(OPENDSA, "Hashing/BucketHash.rst", "Hash Tables"),
    _d(OPENDSA, "Hashing/HashCSimple.rst", "Hash Tables"),
    _d(OPENDSA, "Hashing/HashCImproved.rst", "Hash Tables"),
    _d(OPENDSA, "Hashing/HashDel.rst", "Hash Tables"),
    _d(OPENDSA, "Hashing/HashAnal.rst", "Hash Tables"),
]

# ---------------------------------------------------------------------------
# Lists, ADTs, object orientation
# ---------------------------------------------------------------------------
_STRUCTURES = [
    _d(OPENDSA, "List/ListIntro.rst", "Python Lists"),
    _d(OPENDSA, "List/ListADT.rst", "Python Lists",
       "Object-Oriented Programming"),
    _d(OPENDSA, "List/ListArray.rst", "Python Lists"),
    _d(OPENDSA, "Design/IntroOO.rst", "Object-Oriented Programming"),
    _d(OPENDSA, "Design/ADT.rst", "Object-Oriented Programming",
       "Python Lists"),
]

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------
_SQL = [
    # The one OpenDSA database page with real prose: relations, tuples,
    # keys. Everything a join argument rests on.
    _d(OPENDSA, "Database/RelationalModel.rst",
       "SQL Left Join", "SQL Group By"),
    # "Explicit FROM clauses and JOINs", "Setting the ON Clause",
    # "OUTER and FULL join", "Aggregate functions with GROUP BY / HAVING".
    _d(SQLALCHEMY, "tutorial/data_select.rst",
       "SQL Left Join", "SQL Group By"),
    _d(SQLALCHEMY, "tutorial/data.rst",
       "SQL Left Join", "SQL Group By"),
    # Relationship loading: why a join multiplies rows, which is the
    # substance of the LEFT JOIN misconception.
    _d(SQLALCHEMY, "orm/queryguide/relationships.rst",
       "SQL Left Join"),
]

# ---------------------------------------------------------------------------
# Python language semantics -- from CPython's own docs, not PY4E
# ---------------------------------------------------------------------------
_PYTHON = [
    _d(CPYTHON, "tutorial/introduction.rst", "Python Lists"),
    _d(CPYTHON, "tutorial/datastructures.rst",
       "Python Lists", "Python Object Mutability"),
    # Documents the mutable-default trap explicitly, and recursion by example.
    _d(CPYTHON, "tutorial/controlflow.rst",
       "Python Default Arguments", "Recursion"),
    _d(CPYTHON, "tutorial/classes.rst",
       "Object-Oriented Programming", "Python Object Mutability"),
    # "Why are default values shared between objects?" and the
    # mutable-argument questions live here.
    _d(CPYTHON, "faq/programming.rst",
       "Python Default Arguments",
       "Python Object Mutability",
       "Python Lists"),
    _d(CPYTHON, "faq/design.rst",
       "Python Object Mutability", "Python Lists"),
]

DOCUMENTS: tuple[Document, ...] = tuple(
    _SEARCH + _SORTING + _GRAPHS + _RECURSION + _HASHING + _STRUCTURES
    + _SQL + _PYTHON
)


def validate() -> list[str]:
  """Check the mapping for problems a silent corpus would hide."""
  problems: list[str] = []
  known = set(CONCEPTS)

  for doc in DOCUMENTS:
    if not doc.concepts:
      problems.append(f"{doc.source_id}: no concept tags")
    for concept in doc.concepts:
      if concept not in known:
        problems.append(f"{doc.source_id}: unknown concept {concept!r}")

  seen: dict[str, str] = {}
  for doc in DOCUMENTS:
    if doc.source_id in seen:
      problems.append(f"duplicate source_id {doc.source_id}")
    seen[doc.source_id] = doc.path

  covered = {c for doc in DOCUMENTS for c in doc.concepts}
  for concept in CONCEPTS:
    if concept not in covered:
      problems.append(f"concept has no documents: {concept!r}")

  return problems


def documents_for(concept: str) -> tuple[Document, ...]:
  return tuple(d for d in DOCUMENTS if concept in d.concepts)

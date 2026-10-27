"""The two retrieval paths, as tools the agent can choose between.

Path A -- semantic_search: real $vectorSearch over embedded CS prose.
          "explain why this loop boundary is wrong"

Path B -- the structured_* functions: ordinary MongoDB filters over the
          taxonomy, the student's attempt history, and test cases.
          "has this student missed this concept before"

Keeping these as separate, individually callable functions is what lets the
coordinator make a routing decision that is visible in a trace, rather than
stuffing everything into one query and calling it hybrid.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from . import config, db, vertex


@dataclass
class Passage:
  """One retrieved chunk, carrying the citation id the critic will check."""

  chunk_id: str
  text: str
  score: float
  source_title: str
  source_url: str
  concept_tags: list[str]
  heading_trail: list[str] = field(default_factory=list)

  def as_prompt_block(self) -> str:
    return f"[{self.chunk_id}] (source: {self.source_title})\n{self.text}"

  @property
  def section(self) -> str:
    """Where in the source document this passage came from."""
    return " > ".join(self.heading_trail)

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


# ---------------------------------------------------------------------------
# Path A: unstructured / semantic
# ---------------------------------------------------------------------------
def semantic_search(
    query: str,
    *,
    k: int = config.DEFAULT_K,
    concept: str | None = None,
    min_score: float = 0.0,
) -> list[Passage]:
  """Find CS explanations that answer a conceptual question.

  Use for: "why", "explain", "how does", "what is the difference between" --
  anything whose answer is prose that has to be understood, not looked up.

  Args:
    query: the student's question, in natural language.
    k: how many passages to return.
    concept: optional concept tag to scope the search. Narrows recall, so
      leave it unset when the question may span concepts.
    min_score: drop passages below this cosine score. Useful for the
      unanswerable set -- if nothing clears the bar, the honest answer is
      that the corpus does not cover it.
  """
  query_vector = vertex.embed_query(query)

  stage: dict[str, Any] = {
      "index": config.VECTOR_INDEX_NAME,
      "path": "embedding",
      "queryVector": query_vector,
      "numCandidates": k * config.NUM_CANDIDATES_MULTIPLIER,
      "limit": k,
  }
  if concept:
    stage["filter"] = {"concept_tags": {"$in": [concept]}}

  # Field names here must match what scripts/ingest_corpus.py writes: the
  # chunk id is the document `_id` ("{source_id}#{index}"), and the source
  # title and link are `title` and `url`. Projecting `chunk_id`,
  # `source_title` and `source_url` instead returns passages whose text and
  # score are correct and whose citation is empty -- which looks like a
  # working pipeline right up until something tries to cite a source.
  pipeline = [
      {"$vectorSearch": stage},
      {
          "$project": {
              "_id": 1,
              "text": 1,
              "concept_tags": 1,
              "title": 1,
              "url": 1,
              "source_id": 1,
              "heading_trail": 1,
              "score": {"$meta": "vectorSearchScore"},
          }
      },
  ]

  results = []
  for doc in db.chunks().aggregate(pipeline):
    if doc.get("score", 0.0) < min_score:
      continue
    results.append(
        Passage(
            chunk_id=str(doc.get("_id", "")),
            text=doc.get("text", ""),
            score=round(float(doc.get("score", 0.0)), 4),
            source_title=doc.get("title") or doc.get("source_id", "unknown"),
            source_url=doc.get("url", ""),
            concept_tags=doc.get("concept_tags", []),
            heading_trail=doc.get("heading_trail", []),
        )
    )
  return results


# ---------------------------------------------------------------------------
# Path B: structured
# ---------------------------------------------------------------------------
def lookup_misconception(concept: str) -> dict[str, Any] | None:
  """Get the canonical misconception for a concept from the taxonomy.

  Use for: generating a flawed exercise, or naming the bug the student is
  supposed to find. This is an exact record lookup, not a search.
  """
  doc = db.misconceptions().find_one(
      {"concept": concept}, {"_id": 0, "embedding": 0}
  )
  if doc:
    return doc
  # Fall back to the alias list before giving up -- historical interaction
  # rows use coarser concept names than the current dropdown.
  return db.misconceptions().find_one(
      {"aliases": concept}, {"_id": 0, "embedding": 0}
  )


def student_concept_history(
    student_id: str, concept: str | None = None, *, limit: int = 20
) -> dict[str, Any]:
  """Has this student struggled with this concept before?

  Use for: deciding how much scaffolding to give, or answering questions
  about the student's own past performance. Pure structured aggregation --
  no embeddings involved.
  """
  match: dict[str, Any] = {"student_id": student_id}
  if concept:
    match["concept"] = concept

  attempts = list(
      db.interactions()
      .find(match, {"_id": 0})
      .sort("ts", -1)
      .limit(limit)
  )

  summary = list(
      db.interactions().aggregate([
          {"$match": match},
          {
              "$group": {
                  "_id": "$concept",
                  "sessions": {"$sum": 1},
                  "solved": {"$sum": {"$cond": ["$is_correct", 1, 0]}},
                  "avg_attempts": {"$avg": "$attempts"},
                  "last_seen": {"$max": "$ts"},
              }
          },
          {"$sort": {"sessions": -1}},
      ])
  )

  for row in summary:
    row["concept"] = row.pop("_id")
    row["success_rate"] = (
        round(row["solved"] / row["sessions"], 3) if row["sessions"] else 0.0
    )
    row["avg_attempts"] = round(row.get("avg_attempts") or 0.0, 2)

  return {
      "student_id": student_id,
      "concept_filter": concept,
      "per_concept": summary,
      "recent_attempts": attempts,
      "total_sessions": sum(r["sessions"] for r in summary),
  }


def weakest_concepts(
    student_id: str, *, limit: int = 3, min_sessions: int = 2
) -> list[dict[str, Any]]:
  """Rank the concepts this student fails most often.

  Use for: "what should I practice next" -- a ranking question that a vector
  search cannot answer, because the answer is a computation over records.
  """
  rows = list(
      db.interactions().aggregate([
          {"$match": {"student_id": student_id}},
          {
              "$group": {
                  "_id": "$concept",
                  "sessions": {"$sum": 1},
                  "solved": {"$sum": {"$cond": ["$is_correct", 1, 0]}},
                  "avg_attempts": {"$avg": "$attempts"},
              }
          },
          {"$match": {"sessions": {"$gte": min_sessions}}},
          {
              "$addFields": {
                  "success_rate": {"$divide": ["$solved", "$sessions"]}
              }
          },
          {"$sort": {"success_rate": 1, "avg_attempts": -1}},
          {"$limit": limit},
      ])
  )
  for row in rows:
    row["concept"] = row.pop("_id")
    row["success_rate"] = round(row["success_rate"], 3)
    row["avg_attempts"] = round(row.get("avg_attempts") or 0.0, 2)
  return rows


def lookup_test_cases(concept: str) -> list[dict[str, Any]]:
  """Fetch executable test cases for a concept.

  Use for: verifying a student's fix actually works, rather than asking a
  model whether it looks right.
  """
  return list(db.test_cases().find({"concept": concept}, {"_id": 0}))


def corpus_stats() -> dict[str, Any]:
  """Counts for the README results table and for sanity checks after ingest."""
  cutoff = datetime.now(timezone.utc) - timedelta(days=365 * 5)
  return {
      "chunks": db.chunks().count_documents({}),
      "misconceptions": db.misconceptions().count_documents({}),
      "interactions": db.interactions().count_documents({}),
      "test_cases": db.test_cases().count_documents({}),
      # distinct() on an array field emits a null for every document whose
      # array is empty, and 156 chunks legitimately carry no concept tag.
      # Counting the raw result reports one concept more than exists --
      # which is how this was found, as a 23 against a taxonomy of 22.
      "distinct_concepts": len(
          [c for c in db.chunks().distinct("concept_tags") if c]
      ),
      "untagged_chunks": db.chunks().count_documents({"concept_tags": []}),
      "distinct_sources": len(db.chunks().distinct("source_id")),
      "licenses": sorted(db.chunks().distinct("license")),
      "interactions_since": cutoff.isoformat(),
  }


# ---------------------------------------------------------------------------
# Tool declarations -- consumed by the coordinator agent
# ---------------------------------------------------------------------------
TOOL_REGISTRY = {
    "semantic_search": semantic_search,
    "lookup_misconception": lookup_misconception,
    "student_concept_history": student_concept_history,
    "weakest_concepts": weakest_concepts,
    "lookup_test_cases": lookup_test_cases,
}

SEMANTIC_TOOLS = {"semantic_search"}
STRUCTURED_TOOLS = set(TOOL_REGISTRY) - SEMANTIC_TOOLS

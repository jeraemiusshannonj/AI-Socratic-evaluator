"""MongoDB Atlas connection, collection handles, and index bootstrap.

One Atlas cluster backs both retrieval paths:

  cs_chunks       unstructured CS content + embeddings -> $vectorSearch
  misconceptions  concept taxonomy                     -> normal filters
  interactions    student attempt history              -> normal filters
  test_cases      per-concept executable checks        -> normal filters

The split is deliberate and is the thing that makes "pipelines for structured
and unstructured data" an accurate claim rather than a hopeful one.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.operations import SearchIndexModel

from . import config

log = logging.getLogger(__name__)

_client: MongoClient | None = None


def client() -> MongoClient:
  global _client
  if _client is None:
    _client = MongoClient(
        config.require_mongo_uri(),
        appname="socratic-evaluator",
        serverSelectionTimeoutMS=10_000,
    )
  return _client


def db() -> Database:
  return client()[config.MONGO_DB]


def chunks() -> Collection:
  return db()[config.COLL_CHUNKS]


def misconceptions() -> Collection:
  return db()[config.COLL_MISCONCEPTIONS]


def interactions() -> Collection:
  return db()[config.COLL_INTERACTIONS]


def test_cases() -> Collection:
  return db()[config.COLL_TEST_CASES]


def ping() -> dict[str, Any]:
  """Cheap connectivity + tier check. Run this before anything else."""
  info = client().admin.command("ping")
  build = client().admin.command("buildInfo")
  return {
      "ok": info.get("ok") == 1,
      "server_version": build.get("version"),
      "database": config.MONGO_DB,
  }


# ---------------------------------------------------------------------------
# Index bootstrap
# ---------------------------------------------------------------------------
def ensure_standard_indexes() -> list[str]:
  """Ordinary B-tree indexes for the structured retrieval path.

  These do not count against the Atlas Free search-index cap -- only
  Search/Vector Search indexes do.
  """
  created = []
  created.append(
      interactions().create_index(
          [("student_id", ASCENDING), ("concept", ASCENDING), ("ts", DESCENDING)],
          name="student_concept_recency",
      )
  )
  created.append(
      interactions().create_index([("ts", DESCENDING)], name="recency")
  )
  created.append(
      misconceptions().create_index(
          [("concept", ASCENDING)], name="concept_unique", unique=True
      )
  )
  created.append(
      test_cases().create_index([("concept", ASCENDING)], name="concept")
  )
  created.append(
      chunks().create_index([("concept_tags", ASCENDING)], name="concept_tags")
  )
  return created


def vector_index_definition() -> dict[str, Any]:
  """The one Vector Search index.

  `concept_tags` is declared as a filter field so the retrieval agent can
  scope a semantic search to a concept. $vectorSearch can only pre-filter on
  fields declared here with type "filter" -- adding the filter later means
  rebuilding the index.
  """
  return {
      "fields": [
          {
              "type": "vector",
              "path": "embedding",
              "numDimensions": config.EMBED_DIM,
              "similarity": "cosine",
          },
          {"type": "filter", "path": "concept_tags"},
          {"type": "filter", "path": "source_id"},
      ]
  }


def ensure_vector_index(*, wait: bool = True, timeout_s: int = 600) -> str:
  """Create the Vector Search index if absent, optionally waiting for READY.

  Index build on M0 is not instant. Nothing you query before it reports
  READY will return results, which looks exactly like a broken pipeline.
  """
  coll = chunks()
  existing = {i["name"]: i for i in coll.list_search_indexes()}

  if config.VECTOR_INDEX_NAME in existing:
    log.info("Vector index %s already exists.", config.VECTOR_INDEX_NAME)
  else:
    coll.create_search_index(
        model=SearchIndexModel(
            definition=vector_index_definition(),
            name=config.VECTOR_INDEX_NAME,
            type="vectorSearch",
        )
    )
    log.info("Created vector index %s.", config.VECTOR_INDEX_NAME)

  if not wait:
    return "CREATED"

  deadline = time.time() + timeout_s
  while time.time() < deadline:
    for idx in coll.list_search_indexes():
      if idx["name"] != config.VECTOR_INDEX_NAME:
        continue
      status = idx.get("status", "UNKNOWN")
      if idx.get("queryable"):
        log.info("Vector index is queryable (status=%s).", status)
        return status
      if status == "FAILED":
        raise RuntimeError(
            f"Vector index build FAILED: {idx.get('error', 'no detail given')}"
        )
    time.sleep(5)
  raise TimeoutError(
      f"Vector index not queryable after {timeout_s}s. Check the Atlas UI."
  )


def search_index_count() -> int:
  """Total Search/Vector Search indexes across the database.

  Atlas Free (M0) caps this at 3 for the whole cluster, regardless of type.
  """
  total = 0
  for name in db().list_collection_names():
    try:
      total += len(list(db()[name].list_search_indexes()))
    except Exception:  # noqa: BLE001 - views and system collections
      continue
  return total

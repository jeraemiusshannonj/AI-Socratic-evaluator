"""Create the Atlas indexes. Run this first, before any ingest.

    python -m scripts.bootstrap_db

This is the script that answers Saturday's open question: whether an M0 free
cluster will actually build a Vector Search index. If this exits 0, the plan
holds. If it fails here, stop and rethink before writing ingest code.
"""

from __future__ import annotations

import sys

from socratic import config, db


def main() -> int:
  print("1. Connectivity")
  info = db.ping()
  print(f"   {info}")

  print("\n2. Standard indexes (structured retrieval path)")
  for name in db.ensure_standard_indexes():
    print(f"   created/confirmed: {name}")

  print("\n3. Vector Search index (unstructured retrieval path)")
  print(f"   dimensions: {config.EMBED_DIM}  similarity: cosine")
  print(f"   filter fields: concept_tags, source_id")
  print("   building (M0 index builds take a few minutes)...")
  status = db.ensure_vector_index(wait=True)
  print(f"   status: {status}")

  used = db.search_index_count()
  print(
      f"\n4. Search indexes in use: {used}"
      f" / {config.MAX_FREE_TIER_SEARCH_INDEXES} (Atlas Free cap)"
  )

  print("\nBootstrap complete. Vector search is available on M0.")
  return 0


if __name__ == "__main__":
  sys.exit(main())

"""Chunk the built corpus, embed it, and upsert it into `cs_chunks`.

    python -m scripts.ingest_corpus --dry-run   # no credentials needed
    python -m scripts.ingest_corpus             # embed and upsert
    python -m scripts.ingest_corpus --force     # re-embed everything

Run `python -m scripts.build_corpus` first; this reads `corpus/_build/`.

`--dry-run` imports nothing beyond the chunker, so it works before any
credentials exist and before the database driver is installed. It is the
check worth running while the Atlas cluster is still being created.

Two properties matter more than speed here.

**Idempotence.** Every chunk gets a deterministic `_id` of
`{source_id}#{index}`, and writes go through `UpdateOne(upsert=True)`. Run
this twice and the collection is identical, so a half-finished ingest is
fixed by running it again rather than by truncating the collection.

**Not paying twice.** Each document stores the SHA-256 of its chunk text
along with the embedding model and dimension used. A re-run embeds only
chunks whose text or model actually changed. Without this, every retry
re-embeds the whole corpus, and the obvious workaround -- dropping the
collection first -- also drops the vector index and waits out another
index build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from corpus import signals
from socratic import config
from socratic.chunking import chunk_markdown

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = REPO_ROOT / "corpus" / "_build"

# The provenance comment build_corpus writes at the top of each file. It is
# there for a human reading the build output; embedding it would put the
# same licence boilerplate into all 477 vectors.
_HEADER = re.compile(r"\A<!--.*?-->\s*", re.S)

# Mongo's own limit is 16MB per document; a chunk is ~1.5KB of text plus a
# 768-float vector, so this only trips if something upstream went wrong.
BATCH_WRITE_SIZE = 200

# A concept with fewer chunks than this behind it retrieves the same passage
# for every question asked about it, which is worse than an honest gap
# because it looks like it is working.
MIN_CHUNKS_PER_CONCEPT = 4


@dataclass
class Record:
  id: str
  source_id: str
  chunk_index: int
  text: str
  content_hash: str
  concept_tags: list[str]
  title: str
  url: str
  license_id: str
  attribution: str
  heading_trail: list[str]
  has_code: bool
  char_count: int

  def metadata(self) -> dict:
    """Everything about the chunk except the vector.

    Kept separate because metadata and embeddings go stale independently.
    Widening a concept's signal set in `corpus/signals.py` changes
    `concept_tags` without changing a single character of chunk text, so a
    re-run must be able to correct the tags without paying to re-embed --
    and, more importantly, must not skip the correction just because the
    text hash still matches.
    """
    return {
        "source_id": self.source_id,
        "chunk_index": self.chunk_index,
        "text": self.text,
        "content_hash": self.content_hash,
        "concept_tags": self.concept_tags,
        "title": self.title,
        "url": self.url,
        "license": self.license_id,
        "attribution": self.attribution,
        "heading_trail": self.heading_trail,
        "has_code": self.has_code,
        "char_count": self.char_count,
    }

  def to_document(self, embedding: list[float]) -> dict:
    return {
        **self.metadata(),
        "_id": self.id,
        "embedding": embedding,
        "embed_model": config.EMBED_MODEL,
        "embed_dim": config.EMBED_DIM,
    }


def load_manifest() -> dict:
  path = BUILD_DIR / "manifest.json"
  if not path.exists():
    sys.exit(
        f"{path.relative_to(REPO_ROOT)} not found.\n"
        "Run: python -m scripts.build_corpus"
    )
  return json.loads(path.read_text(encoding="utf-8"))


def build_records(manifest: dict) -> tuple[list[Record], dict[str, int]]:
  """Chunk every document into upsertable records.

  Returns the records plus a count of how many candidate tags were dropped
  per concept, so the caller can report what chunk-level filtering removed
  rather than leaving it invisible.
  """
  records: list[Record] = []
  dropped: dict[str, int] = {}
  for entry in manifest["documents"]:
    md_path = BUILD_DIR / entry["file"]
    if not md_path.exists():
      sys.exit(f"missing built file {md_path.relative_to(REPO_ROOT)}")

    body = _HEADER.sub("", md_path.read_text(encoding="utf-8"))
    chunks = chunk_markdown(body)

    candidates = tuple(entry["concepts"])
    for index, chunk in enumerate(chunks):
      tags = signals.filter_concepts(candidates, chunk.text,
                                     chunk.heading_trail)
      for concept in candidates:
        if concept not in tags:
          dropped[concept] = dropped.get(concept, 0) + 1

      records.append(
          Record(
              id=f"{entry['source_id']}#{index}",
              source_id=entry["source_id"],
              chunk_index=index,
              text=chunk.text,
              content_hash=hashlib.sha256(
                  chunk.text.encode("utf-8")).hexdigest(),
              concept_tags=tags,
              title=entry.get("title") or entry["source_id"],
              url=entry["url"],
              license_id=entry["license"],
              attribution=entry["attribution"],
              heading_trail=list(chunk.heading_trail),
              has_code=chunk.has_code,
              char_count=chunk.char_count,
          )
      )
  return records, dropped


def existing_state() -> dict[str, tuple[str, str, int]]:
  """Map _id -> (content_hash, embed_model, embed_dim) already stored."""
  from socratic import db

  state: dict[str, tuple[str, str, int]] = {}
  projection = {"content_hash": 1, "embed_model": 1, "embed_dim": 1}
  for doc in db.chunks().find({}, projection):
    state[doc["_id"]] = (
        doc.get("content_hash", ""),
        doc.get("embed_model", ""),
        doc.get("embed_dim", 0),
    )
  return state


def needs_embedding(
    record: Record, state: dict[str, tuple[str, str, int]]
) -> bool:
  stored = state.get(record.id)
  if stored is None:
    return True
  return stored != (record.content_hash, config.EMBED_MODEL, config.EMBED_DIM)


def _fake_vector(record: Record) -> list[float]:
  """A deterministic unit vector for dry runs.

  Derived from the content hash so the same chunk always gets the same
  vector. This exercises chunking, record construction, the write path and
  the index, without spending anything on the embedding API. Nothing
  retrieved against these vectors means anything, which is why the dry run
  refuses to write by default.
  """
  import struct

  # Derived from unsigned ints, not from reinterpreting the hash bytes as
  # IEEE-754 floats: random bit patterns decode to infinities and NaNs
  # roughly 2% of the time, and a single NaN makes the whole vector's norm
  # NaN -- which Atlas would accept and then never match.
  digest = hashlib.sha256(record.content_hash.encode()).digest()
  vec: list[float] = []
  while len(vec) < config.EMBED_DIM:
    digest = hashlib.sha256(digest).digest()
    for value in struct.unpack(">8I", digest[:32]):
      vec.append(value / 2147483647.5 - 1.0)  # uint32 -> [-1, 1)
  vec = vec[: config.EMBED_DIM]
  norm = sum(v * v for v in vec) ** 0.5 or 1.0
  return [v / norm for v in vec]


def _chunked(items: list, size: int) -> Iterable[list]:
  for start in range(0, len(items), size):
    yield items[start : start + size]


def write(
    embedded: list[dict], metadata_only: list[tuple[str, dict]]
) -> tuple[int, int]:
  """Upsert full documents and refresh metadata on the rest.

  `embedded` carries newly computed vectors and is upserted whole.
  `metadata_only` is `(chunk_id, fields)` for chunks whose text and model
  are unchanged: their tags, titles and headings are refreshed without
  touching `embedding`, so a signal-set change propagates for free.
  """
  from pymongo import UpdateOne
  from pymongo.errors import BulkWriteError

  from socratic import db

  operations = [
      UpdateOne({"_id": doc["_id"]}, {"$set": doc}, upsert=True)
      for doc in embedded
  ]
  operations += [
      UpdateOne({"_id": chunk_id}, {"$set": fields})
      for chunk_id, fields in metadata_only
  ]

  upserted = modified = 0
  for batch in _chunked(operations, BATCH_WRITE_SIZE):
    try:
      result = db.chunks().bulk_write(batch, ordered=False)
    except BulkWriteError as exc:
      sys.exit(f"bulk write failed: {exc.details}")
    upserted += len(result.upserted_ids or {})
    modified += result.modified_count
  return upserted, modified


def prune(valid_ids: set[str]) -> int:
  """Delete chunks that no longer correspond to anything in the corpus.

  Shrinking a document leaves its tail chunks orphaned -- `#41` still sits
  in the collection after a rewrite produced only 38 chunks, and it stays
  retrievable. Deleting by "not in the current set" handles that without
  dropping the collection and its index.
  """
  from socratic import db

  result = db.chunks().delete_many({"_id": {"$nin": sorted(valid_ids)}})
  return result.deleted_count


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dry-run", action="store_true",
                      help="chunk and report; no embedding, no writes")
  parser.add_argument("--force", action="store_true",
                      help="re-embed every chunk, ignoring stored hashes")
  parser.add_argument("--limit", type=int, default=0,
                      help="only process the first N chunks (smoke test)")
  parser.add_argument("--no-prune", action="store_true",
                      help="keep chunks that are no longer in the corpus")
  args = parser.parse_args()

  from corpus import sources

  gaps = signals.missing_signal_sets(sources.CONCEPTS)
  if gaps:
    print("concepts with no signal set in corpus/signals.py "
          "(document tags will be trusted as-is):")
    for concept in gaps:
      print(f"  - {concept}")
    print()

  manifest = load_manifest()
  records, dropped_tags = build_records(manifest)
  if args.limit:
    records = records[: args.limit]

  sizes = [r.char_count for r in records]
  print(f"corpus: {manifest['totals']['documents']} documents, "
        f"{len(records)} chunks")
  if sizes:
    print(f"chunk chars: min {min(sizes)}, mean {sum(sizes)//len(sizes)}, "
          f"max {max(sizes)}")
  oversize = [r for r in records if r.char_count > config.MAX_CHUNK_CHARS]
  if oversize:
    print(f"\n{len(oversize)} chunks exceed MAX_CHUNK_CHARS "
          f"({config.MAX_CHUNK_CHARS}); the embedder would truncate them.")
    for record in oversize[:5]:
      print(f"  {record.id}: {record.char_count}")
    return 2

  untagged = sum(1 for r in records if not r.concept_tags)
  print(f"{untagged} chunks carry no concept tag "
        f"(reachable by unfiltered search only)")

  per_concept: dict[str, int] = {}
  for record in records:
    for concept in record.concept_tags:
      per_concept[concept] = per_concept.get(concept, 0) + 1

  print("\nchunks per concept (kept / candidates):")
  starved: list[str] = []
  for concept in sources.CONCEPTS:
    kept = per_concept.get(concept, 0)
    candidates = kept + dropped_tags.get(concept, 0)
    share = f"{kept}/{candidates}"
    print(f"  {share:>9}  {concept}")
    if kept < MIN_CHUNKS_PER_CONCEPT:
      starved.append(f"{concept} ({kept})")

  if starved and not args.limit:
    print(f"\nCONCEPTS BELOW {MIN_CHUNKS_PER_CONCEPT} CHUNKS:")
    for item in starved:
      print(f"  - {item}")
    print("\nA concept with almost nothing behind it will retrieve the same"
          "\none or two passages for every question. Either widen its signal"
          "\nset in corpus/signals.py or add a document in corpus/sources.py.")
    return 3

  if args.dry_run:
    print(f"\nembedding model would be {config.EMBED_MODEL} "
          f"at {config.EMBED_DIM} dimensions")
    print(f"would embed {len(records)} chunks and upsert into "
          f"{config.MONGO_DB}.{config.COLL_CHUNKS}")
    print("\ndry run: nothing embedded, nothing written")
    return 0

  from socratic import db, vertex

  print(f"\nmongo: {db.ping()}")

  state = {} if args.force else existing_state()
  pending = [r for r in records if needs_embedding(r, state)]
  unchanged = len(records) - len(pending)
  print(f"{len(pending)} chunks to embed, {unchanged} unchanged")

  documents: list[dict] = []
  if pending:
    started = time.perf_counter()
    vectors = vertex.embed_documents([r.text for r in pending],
                                     show_progress=True)
    elapsed = time.perf_counter() - started
    if len(vectors) != len(pending):
      sys.exit(f"embedder returned {len(vectors)} vectors for "
               f"{len(pending)} chunks")
    bad = [i for i, v in enumerate(vectors) if len(v) != config.EMBED_DIM]
    if bad:
      sys.exit(f"{len(bad)} vectors have the wrong dimension; expected "
               f"{config.EMBED_DIM}, first offender had {len(vectors[bad[0]])}")
    print(f"embedded {len(pending)} chunks in {elapsed:.1f}s "
          f"({len(pending)/elapsed:.1f}/s)")
    documents = [r.to_document(v) for r, v in zip(pending, vectors)]

  pending_ids = {r.id for r in pending}
  refresh = [(r.id, r.metadata()) for r in records
             if r.id not in pending_ids and r.id in state]
  upserted, modified = write(documents, refresh)
  print(f"upserted {upserted}, modified {modified} "
        f"({len(refresh)} metadata-only refreshes)")

  if not args.no_prune and not args.limit:
    deleted = prune({r.id for r in records})
    if deleted:
      print(f"pruned {deleted} stale chunks")

  print(f"\n{db.chunks().count_documents({})} chunks now in "
        f"{config.MONGO_DB}.{config.COLL_CHUNKS}")
  print("\nNext: python -m scripts.bootstrap_db  (builds the vector index)")
  return 0


if __name__ == "__main__":
  sys.exit(main())

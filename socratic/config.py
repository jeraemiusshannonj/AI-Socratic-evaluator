"""Central configuration. Every tunable lives here so nothing is hardcoded twice.

Model IDs on Vertex move fast (Gemini 2.5 was retired on 2026-10-20). Keep them
in one place and verify against your own project with:

    python -m scripts.check_models
"""

import os

from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
  value = os.getenv(name)
  if not value:
    raise RuntimeError(
        f"Missing required environment variable {name}. "
        "Copy .env.example to .env and fill it in."
    )
  return value


# ---------------------------------------------------------------------------
# Google Cloud / Vertex AI
# ---------------------------------------------------------------------------
GCP_PROJECT = os.getenv("GCP_PROJECT", "")
# gemini-3.8-flash is served from the `global`, `us` and `eu` endpoints.
# A single region such as us-central1 may not have it; `global` is the safe
# default and costs nothing extra.
GCP_LOCATION = os.getenv("GCP_LOCATION", "global")

# Generation model. Replaces the previous Groq-hosted openai/gpt-oss-20b.
GEN_MODEL = os.getenv("GEN_MODEL", "gemini-3.8-flash")

# Embedding model. gemini-embedding-001 is GA; gemini-embedding-2-preview
# exists but is preview and can change under you before November.
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")

# gemini-embedding-001 defaults to 3072 dimensions and supports Matryoshka
# truncation. 768 is the documented quality/size sweet spot and keeps the
# Atlas M0 512MB storage cap irrelevant at our corpus size.
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

# Vertex rejects oversized embedding batches, and the per-request instance
# limit differs by model and has changed before. We batch at this size and
# fall back to one-at-a-time on failure (see vertex.embed_documents).
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))

# ---------------------------------------------------------------------------
# MongoDB Atlas
# ---------------------------------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI", "")
MONGO_DB = os.getenv("MONGO_DB", "socratic")

COLL_CHUNKS = "cs_chunks"          # unstructured: vector-searched
COLL_MISCONCEPTIONS = "misconceptions"  # structured: taxonomy
COLL_INTERACTIONS = "interactions"      # structured: student history
COLL_TEST_CASES = "test_cases"          # structured: per-concept tests

VECTOR_INDEX_NAME = "chunks_vector_index"

# Atlas Free (M0) allows at most 3 search indexes of any type across the
# cluster. We use exactly one. Do not add a second without checking.
MAX_FREE_TIER_SEARCH_INDEXES = 3

# ---------------------------------------------------------------------------
# Retrieval defaults
# ---------------------------------------------------------------------------
DEFAULT_K = 5
# Atlas guidance: numCandidates should be 10-20x the limit for good recall.
NUM_CANDIDATES_MULTIPLIER = 20

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_TARGET_CHARS = 1400
CHUNK_OVERLAP_CHARS = 200

# gemini-embedding-001 accepts up to 2048 input tokens and silently truncates
# past that when auto_truncate is on. At ~4 chars/token that is ~8000 chars.
# We hard-split well below the ceiling so no chunk is ever embedded as a
# partial, which would look like a retrieval miss and be near-impossible to
# diagnose from the output.
MAX_CHUNK_CHARS = 6000

# A chunk shorter than this carries too little context to be worth retrieving
# on its own -- it wins on cosine similarity for the wrong reasons (short
# texts embed tightly) and then supplies nothing the critic can cite. Tails
# below this are merged back into the preceding chunk.
MIN_CHUNK_CHARS = 300


# Values straight out of .env.example. Left in place they produce failures
# that look like infrastructure problems: a project ID of
# "your-gcp-project-id" comes back from Vertex as a 404 on the publisher
# model, which reads as a wrong model ID rather than an unedited config file.
_PLACEHOLDERS = frozenset({
    "your-gcp-project-id",
    "your_gcp_project_id",
    "your-project-id",
    "your_project_id",
    "YOUR_PROJECT_ID",
    "changeme",
    "mongodb+srv://USER:PASSWORD@cluster0.xxxxx.mongodb.net/"
    "?retryWrites=true&w=majority",
})


def _reject_placeholder(name: str, value: str) -> str:
  if value in _PLACEHOLDERS or "USER:PASSWORD" in value:
    raise RuntimeError(
        f"{name} is still the placeholder from .env.example ({value!r}). "
        f"Open .env and set a real value."
    )
  return value


def require_mongo_uri() -> str:
  return _reject_placeholder(
      "MONGO_URI", MONGO_URI or _require("MONGO_URI")
  )


def require_gcp_project() -> str:
  return _reject_placeholder(
      "GCP_PROJECT", GCP_PROJECT or _require("GCP_PROJECT")
  )

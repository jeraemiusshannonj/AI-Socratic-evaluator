"""Vertex AI access: one client, embeddings, and generation.

Replaces the previous ChatGroq usage. Everything that talks to a model goes
through here so model IDs, retries and token accounting have one home.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from google import genai
from google.genai import types

from . import config

log = logging.getLogger(__name__)

_client: genai.Client | None = None


def client() -> genai.Client:
  """Lazily build the Vertex-backed client.

  Uses Application Default Credentials. Locally that means:
      gcloud auth application-default login
  On Cloud Run it is the attached service account (Vertex AI User only).
  """
  global _client
  if _client is None:
    _client = genai.Client(
        vertexai=True,
        project=config.require_gcp_project(),
        location=config.GCP_LOCATION,
    )
  return _client


# ---------------------------------------------------------------------------
# Usage accounting -- feeds the tokens/sec and cost-per-request metrics
# ---------------------------------------------------------------------------
@dataclass
class Usage:
  """Rolling counters for one request or one eval run."""

  prompt_tokens: int = 0
  output_tokens: int = 0
  calls: int = 0
  latencies_ms: list[float] = field(default_factory=list)

  def add(self, prompt: int, output: int, latency_ms: float) -> None:
    self.prompt_tokens += prompt
    self.output_tokens += output
    self.calls += 1
    self.latencies_ms.append(latency_ms)

  @property
  def total_tokens(self) -> int:
    return self.prompt_tokens + self.output_tokens

  def percentile(self, p: float) -> float:
    if not self.latencies_ms:
      return 0.0
    ordered = sorted(self.latencies_ms)
    idx = min(len(ordered) - 1, max(0, math.ceil(p / 100 * len(ordered)) - 1))
    return ordered[idx]

  def summary(self) -> dict[str, Any]:
    wall_s = sum(self.latencies_ms) / 1000 or 1e-9
    return {
        "calls": self.calls,
        "prompt_tokens": self.prompt_tokens,
        "output_tokens": self.output_tokens,
        "total_tokens": self.total_tokens,
        "tokens_per_sec": round(self.total_tokens / wall_s, 1),
        "p50_latency_ms": round(self.percentile(50), 1),
        "p95_latency_ms": round(self.percentile(95), 1),
    }


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def _embed_call(texts: Sequence[str], task_type: str) -> list[list[float]]:
  resp = client().models.embed_content(
      model=config.EMBED_MODEL,
      contents=list(texts),
      config=types.EmbedContentConfig(
          task_type=task_type,
          output_dimensionality=config.EMBED_DIM,
          # Explicit rather than relying on the default: a silent truncation
          # here produces a valid-looking vector for partial text, which is
          # the hardest class of retrieval bug to find later. The chunker
          # already guarantees we stay under the limit.
          auto_truncate=False,
      ),
  )
  return [list(e.values) for e in resp.embeddings]


def _normalize(vec: list[float]) -> list[float]:
  """Gemini embeddings below the native 3072 dims are not unit-normalized.

  Cosine similarity in Atlas tolerates unnormalized vectors, but normalizing
  makes dotProduct and cosine agree and makes scores comparable across runs.
  """
  norm = math.sqrt(sum(v * v for v in vec))
  if norm == 0:
    return vec
  return [v / norm for v in vec]


def embed_documents(
    texts: Sequence[str], *, show_progress: bool = False
) -> list[list[float]]:
  """Embed corpus chunks for storage. Batches, with per-item fallback.

  The per-request instance limit for Gemini embedding models on Vertex has
  changed between releases, so a batch rejection falls back to singles rather
  than failing the whole ingest.
  """
  out: list[list[float]] = []
  total = len(texts)
  for start in range(0, total, config.EMBED_BATCH_SIZE):
    batch = texts[start : start + config.EMBED_BATCH_SIZE]
    try:
      vectors = _embed_call(batch, "RETRIEVAL_DOCUMENT")
    except Exception as exc:  # noqa: BLE001 - we genuinely want any failure
      log.warning(
          "Batch embed of %d failed (%s); falling back to one at a time.",
          len(batch),
          type(exc).__name__,
      )
      vectors = []
      for text in batch:
        vectors.extend(_retry(lambda t=text: _embed_call([t], "RETRIEVAL_DOCUMENT")))
    out.extend(_normalize(v) for v in vectors)
    if show_progress:
      print(f"  embedded {min(start + len(batch), total)}/{total}", flush=True)
  return out


def embed_query(text: str) -> list[float]:
  """Embed a search query. Must use RETRIEVAL_QUERY, not RETRIEVAL_DOCUMENT.

  Asymmetric task types are the whole point: a question and the passage that
  answers it are not the same kind of text, and using one task type for both
  measurably degrades recall.
  """
  return _normalize(_retry(lambda: _embed_call([text], "RETRIEVAL_QUERY"))[0])


def _retry(fn, attempts: int = 4, base_delay: float = 1.0):
  """Retry with exponential backoff, and never hide why it failed.

  Two kinds of failure arrive through the same exception type here: a
  transient one worth retrying (rate limit, backend hiccup) and a
  permanent one that will fail identically four times in a row (a project
  that does not exist, a model ID that was retired, a missing permission).
  Logging only the class name turns the second kind into a silent 7-second
  wait followed by "failed after 4 attempts", which says nothing about the
  misconfiguration that caused it. So the message is carried through, and
  the permanent cases are not retried at all.
  """
  last: Exception | None = None
  for i in range(attempts):
    try:
      return fn()
    except Exception as exc:  # noqa: BLE001
      last = exc
      detail = str(exc).strip().replace("\n", " ")[:400]

      if _is_permanent(exc):
        raise RuntimeError(
            f"Vertex call failed and will not succeed on retry: {detail}"
        ) from exc

      if i == attempts - 1:
        break
      delay = base_delay * (2**i)
      log.warning("Retrying after %s in %.1fs: %s",
                  type(exc).__name__, delay, detail)
      time.sleep(delay)

  detail = str(last).strip().replace("\n", " ")[:400] if last else "no detail"
  raise RuntimeError(
      f"Vertex call failed after {attempts} attempts: {detail}"
  ) from last


# Substrings that mean the request is wrong, not that the service is busy.
# Retrying these wastes seven seconds and then reports nothing useful.
_PERMANENT_MARKERS = (
    "was not found",
    "does not exist",
    "permission_denied",
    "permission denied",
    "has not been used in project",
    "is not enabled",
    "api_key_invalid",
    "invalid_argument",
    "unauthenticated",
    "billing",
    "not supported",
    "publisher model",
)


def _is_permanent(exc: Exception) -> bool:
  text = str(exc).lower()
  if any(marker in text for marker in _PERMANENT_MARKERS):
    # A 429 can mention quota alongside these words; quota is retryable.
    return "resource_exhausted" not in text and "rate limit" not in text
  return False


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate(
    prompt: str,
    *,
    system_instruction: str | None = None,
    temperature: float = 0.2,
    response_schema: Any | None = None,
    usage: Usage | None = None,
    model: str | None = None,
) -> str:
  """One generation call. Returns raw text.

  Pass `response_schema` to force structured JSON out of the model instead of
  parsing fenced blocks out of prose.
  """
  cfg: dict[str, Any] = {"temperature": temperature}
  if system_instruction:
    cfg["system_instruction"] = system_instruction
  if response_schema is not None:
    cfg["response_mime_type"] = "application/json"
    cfg["response_schema"] = response_schema

  started = time.perf_counter()
  resp = _retry(
      lambda: client().models.generate_content(
          model=model or config.GEN_MODEL,
          contents=prompt,
          config=types.GenerateContentConfig(**cfg),
      )
  )
  elapsed_ms = (time.perf_counter() - started) * 1000

  if usage is not None:
    meta = getattr(resp, "usage_metadata", None)
    usage.add(
        getattr(meta, "prompt_token_count", 0) or 0,
        getattr(meta, "candidates_token_count", 0) or 0,
        elapsed_ms,
    )
  return resp.text or ""

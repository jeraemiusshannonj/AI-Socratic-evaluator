"""Verify the configured Vertex model IDs are live in YOUR project.

    python -m scripts.check_models

Gemini model IDs on Vertex move fast -- the 2.5 family was retired on
2026-10-20. Run this on day one and again before you submit. Two minutes
here beats a demo that 404s during an interview.
"""

from __future__ import annotations

import sys

from socratic import config, vertex


def main() -> int:
  print(f"project : {config.GCP_PROJECT}")
  print(f"location: {config.GCP_LOCATION}")
  print(f"gen     : {config.GEN_MODEL}")
  print(f"embed   : {config.EMBED_MODEL} @ {config.EMBED_DIM} dims\n")

  ok = True

  print("Testing generation...")
  try:
    usage = vertex.Usage()
    out = vertex.generate(
        "Reply with exactly: OK", temperature=0.0, usage=usage
    )
    print(f"  response: {out.strip()[:60]!r}")
    print(f"  usage   : {usage.summary()}")
  except Exception as exc:  # noqa: BLE001
    ok = False
    print(f"  FAILED: {type(exc).__name__}: {exc}")

  print("\nTesting embeddings...")
  try:
    vec = vertex.embed_query("binary search off by one error")
    norm = sum(v * v for v in vec) ** 0.5
    print(f"  dimensions: {len(vec)} (expected {config.EMBED_DIM})")
    print(f"  L2 norm   : {norm:.4f} (should be ~1.0 after normalization)")
    if len(vec) != config.EMBED_DIM:
      ok = False
      print("  FAILED: dimension mismatch -- fix EMBED_DIM before ingesting,")
      print("          or the vector index will reject every document.")
  except Exception as exc:  # noqa: BLE001
    ok = False
    print(f"  FAILED: {type(exc).__name__}: {exc}")

  print("\nOK" if ok else "\nFAILURES above -- do not proceed to ingest.")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())

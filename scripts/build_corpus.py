"""Fetch the upstream sources and convert them to a retrieval-ready corpus.

    python -m scripts.build_corpus            # fetch (or reuse) and convert
    python -m scripts.build_corpus --refetch   # discard clones and re-pull
    python -m scripts.build_corpus --no-fetch  # convert what is on disk

Output goes to `corpus/_build/`, which is gitignored. The converted text is
deliberately *not* committed: the mapping in `corpus/sources.py` plus this
script reproduce it exactly, so the repository carries the method rather
than a copy of three upstream projects' documentation. `manifest.json` is
written alongside so the corpus is auditable -- per-document character
counts, licences, upstream URLs -- without the text itself.

Every source is permissively licensed (MIT, PSF, MIT). See the module
docstring in `corpus/sources.py` for why PY4E is excluded.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

from corpus import sources
from corpus.rst import rst_to_markdown

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = REPO_ROOT / "corpus" / "_sources"
BUILD_DIR = REPO_ROOT / "corpus" / "_build"

# A document that converts to less than this is a failure, not a short page.
# OpenDSA's SQL chapter looked fine in the mapping and converted to 18
# characters, because its prose lives in JavaScript slideshow captions. That
# would have shipped as a corpus with two concepts silently unanswerable, so
# the build refuses to be quiet about it.
MIN_USEFUL_CHARS = 600


def _run(cmd: list[str], cwd: Path | None = None) -> None:
  result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
  if result.returncode != 0:
    sys.exit(
        f"command failed: {' '.join(cmd)}\n"
        f"{result.stderr.strip() or result.stdout.strip()}"
    )


def fetch(repo: sources.Repo, *, refetch: bool) -> Path:
  """Clone (or reuse) one upstream repository, shallow and sparse."""
  dest = SOURCE_DIR / repo.key
  if dest.exists():
    if not refetch:
      print(f"  {repo.key}: reusing {dest.relative_to(REPO_ROOT)}")
      return dest
    shutil.rmtree(dest)

  dest.parent.mkdir(parents=True, exist_ok=True)
  print(f"  {repo.key}: cloning {repo.clone_url}")

  if repo.sparse:
    # blob:none + sparse keeps CPython from pulling the whole interpreter
    # history for a dozen documentation files.
    _run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
          repo.clone_url, str(dest)])
    _run(["git", "sparse-checkout", "set", *repo.sparse], cwd=dest)
  else:
    _run(["git", "clone", "--depth", "1", repo.clone_url, str(dest)])

  return dest


def build(*, fetch_sources: bool, refetch: bool) -> int:
  problems = sources.validate()
  if problems:
    print("source mapping is invalid:")
    for problem in problems:
      print(f"  - {problem}")
    return 1

  print(f"Sources ({len(sources.REPOS)} repositories):")
  roots: dict[str, Path] = {}
  for repo in sources.REPOS:
    if fetch_sources:
      roots[repo.key] = fetch(repo, refetch=refetch)
    else:
      roots[repo.key] = SOURCE_DIR / repo.key
      if not roots[repo.key].exists():
        return _fail_missing(repo)

  if BUILD_DIR.exists():
    shutil.rmtree(BUILD_DIR)
  BUILD_DIR.mkdir(parents=True)

  print(f"\nConverting {len(sources.DOCUMENTS)} documents:")
  entries: list[dict] = []
  dropped_total: Counter[str] = Counter()
  thin: list[tuple[str, int]] = []
  total_chars = 0

  for doc in sources.DOCUMENTS:
    src_path = roots[doc.repo.key] / doc.repo.root / doc.path
    if not src_path.exists():
      print(f"  !! missing {doc.repo.key}:{doc.path}")
      thin.append((doc.source_id, -1))
      continue

    raw = src_path.read_text(encoding="utf-8", errors="replace")
    conv = rst_to_markdown(raw, source_name=doc.path)
    dropped_total.update(conv.dropped)

    if conv.char_count < MIN_USEFUL_CHARS:
      thin.append((doc.source_id, conv.char_count))

    out_name = f"{doc.source_id.replace(':', '__')}.md"
    header = (
        f"<!-- source: {doc.url}\n"
        f"     license: {doc.repo.license_id}\n"
        f"     attribution: {doc.repo.attribution}\n"
        f"     concepts: {'; '.join(doc.concepts)} -->\n\n"
    )
    (BUILD_DIR / out_name).write_text(header + conv.markdown + "\n",
                                      encoding="utf-8")

    total_chars += conv.char_count
    entries.append({
        "source_id": doc.source_id,
        "file": out_name,
        "path": doc.path,
        "repo": doc.repo.key,
        "license": doc.repo.license_id,
        "attribution": doc.repo.attribution,
        "url": doc.url,
        "concepts": list(doc.concepts),
        "title": conv.title,
        "chars": conv.char_count,
        "code_blocks": conv.code_blocks,
        "headings": conv.headings,
    })
    print(f"  {conv.char_count:>7,} chars  code={conv.code_blocks:>3}  "
          f"{doc.source_id}")

  manifest = {
      "documents": entries,
      "totals": {
          "documents": len(entries),
          "chars": total_chars,
          "code_blocks": sum(e["code_blocks"] for e in entries),
      },
      "licenses": sorted({e["license"] for e in entries}),
      "attributions": sorted({e["attribution"] for e in entries}),
      "concept_coverage": {
          concept: sorted(
              e["source_id"] for e in entries if concept in e["concepts"]
          )
          for concept in sources.CONCEPTS
      },
      "dropped_directives": dict(dropped_total.most_common()),
  }
  (BUILD_DIR / "manifest.json").write_text(
      json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

  print(f"\n{len(entries)} documents, {total_chars:,} characters")
  print(f"wrote {BUILD_DIR.relative_to(REPO_ROOT)}/ "
        f"({len(entries)} .md + manifest.json)")

  unknown = {k: v for k, v in dropped_total.items() if k.startswith("unknown:")}
  if unknown:
    print("\nDirectives this converter does not know (bodies were kept):")
    for name, count in sorted(unknown.items(), key=lambda kv: -kv[1]):
      print(f"  {count:>4}x {name.split(':', 1)[1]}")

  empty = [c for c in sources.CONCEPTS
           if not manifest["concept_coverage"][c]]
  if empty:
    print("\nCONCEPTS WITH NO DOCUMENTS:")
    for concept in empty:
      print(f"  - {concept}")

  if thin:
    print(f"\nTHIN OR MISSING DOCUMENTS (under {MIN_USEFUL_CHARS} chars):")
    for source_id, count in thin:
      label = "missing file" if count < 0 else f"{count} chars"
      print(f"  - {source_id}: {label}")
    print("\nA document this short converted to almost nothing. Usually the"
          "\nupstream page is interactive rather than prose. Check it, then"
          "\neither drop it from corpus/sources.py or find a real source for"
          "\nthat concept -- do not ship a concept nothing can answer.")
    return 2

  return 0


def _fail_missing(repo: sources.Repo) -> int:
  print(f"missing {SOURCE_DIR / repo.key} -- run without --no-fetch first")
  return 1


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--refetch", action="store_true",
                      help="discard existing clones and pull again")
  parser.add_argument("--no-fetch", action="store_true",
                      help="convert the clones already on disk")
  args = parser.parse_args()
  return build(fetch_sources=not args.no_fetch, refetch=args.refetch)


if __name__ == "__main__":
  sys.exit(main())

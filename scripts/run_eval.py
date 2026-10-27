"""Measure the system and write the numbers down.

    python -m scripts.run_eval --retrieval        # cheap: embeddings only
    python -m scripts.run_eval --critic           # costs generation calls
    python -m scripts.run_eval --all --out eval/results/run.json

Three measurements, each answering a question the project would otherwise
only be asserting an answer to.

**Retrieval.** Does the right document come back? Document-level, against
eval/retrieval_questions.json, run twice: once with the concept filter and
once without. The unfiltered run is the one that matters, because with a
filter applied the corpus is already narrowed to a handful of documents
and a hit is close to free. Reported as hit@1 and hit@k with the gap
between filtered and unfiltered stated, since that gap is the honest
measure of how much work the metadata is doing versus the embeddings.

**The critic.** Over the 44 reference implementations in eval/references.py
-- one correct and one deliberately buggy per concept -- three things:
whether the verdict is right (it comes from running the code, so this
mostly measures the test cases), whether the model's own reading agreed,
and whether its citations point at passages it was actually given. The
agreement rate is the interesting one. A model that reads code correctly
and a model that sounds confident are different things, and this separates
them.

**Cost and latency.** Tokens, throughput and p50/p95 per call, taken from
the same runs rather than from a separate benchmark.

Nothing here is a pass/fail gate. These are numbers to look at and to put
in the README, and a gate would only create pressure to tune them.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval import references
from socratic import config, retrieval, tutor, vertex

QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "eval" / "retrieval_questions.json"


def _pct(numerator: int, denominator: int) -> str:
  if not denominator:
    return "n/a"
  return f"{numerator}/{denominator} ({numerator / denominator:.0%})"


@dataclass
class RetrievalOutcome:
  hit_at_1: int = 0
  hit_at_k: int = 0
  total: int = 0
  misses: list[dict] = field(default_factory=list)
  latencies_ms: list[float] = field(default_factory=list)

  def summary(self) -> dict[str, Any]:
    return {
        "questions": self.total,
        "hit_at_1": self.hit_at_1,
        "hit_at_k": self.hit_at_k,
        "hit_at_1_rate": round(self.hit_at_1 / self.total, 3) if self.total else 0,
        "hit_at_k_rate": round(self.hit_at_k / self.total, 3) if self.total else 0,
        "p50_latency_ms": round(statistics.median(self.latencies_ms), 1)
                          if self.latencies_ms else 0,
        "misses": self.misses,
    }


def run_retrieval(k: int, *, filtered: bool) -> RetrievalOutcome:
  questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
  outcome = RetrievalOutcome(total=len(questions))

  for item in questions:
    concept = item["concept"] if filtered else None
    expected = set(item["expect"])

    started = time.perf_counter()
    passages = retrieval.semantic_search(item["query"], k=k, concept=concept)
    outcome.latencies_ms.append((time.perf_counter() - started) * 1000)

    # chunk_id is "{source_id}#{index}"; the document is the part before #.
    documents = [p.chunk_id.rsplit("#", 1)[0] for p in passages]
    rank = next(
        (i for i, doc in enumerate(documents, 1) if doc in expected), None)

    if rank == 1:
      outcome.hit_at_1 += 1
    if rank is not None:
      outcome.hit_at_k += 1
    else:
      outcome.misses.append({
          "query": item["query"],
          "concept": item["concept"],
          "expected": sorted(expected),
          "returned": documents,
      })

  return outcome


@dataclass
class CriticOutcome:
  total: int = 0
  verdict_correct: int = 0
  model_agreed: int = 0
  model_judged: int = 0
  grounded: int = 0
  with_passages: int = 0
  fabricated_citations: int = 0
  total_citations: int = 0
  disagreements: list[dict] = field(default_factory=list)
  fabrications: list[dict] = field(default_factory=list)

  def summary(self) -> dict[str, Any]:
    return {
        "evaluations": self.total,
        "verdict_correct": self.verdict_correct,
        "verdict_accuracy": round(self.verdict_correct / self.total, 3)
                            if self.total else 0,
        "model_agreement": round(self.model_agreed / self.model_judged, 3)
                           if self.model_judged else None,
        "model_judged_on": self.model_judged,
        "grounded": self.grounded,
        "grounded_rate": round(self.grounded / self.with_passages, 3)
                         if self.with_passages else None,
        "citations_total": self.total_citations,
        "citations_fabricated": self.fabricated_citations,
        "citation_validity": round(
            1 - self.fabricated_citations / self.total_citations, 3)
            if self.total_citations else None,
        "disagreements": self.disagreements,
        "fabrications": self.fabrications,
    }


def run_critic(
    concepts: list[str], usage: vertex.Usage, *, attempt: int = 2
) -> CriticOutcome:
  """Evaluate a correct and a buggy implementation of each concept."""
  outcome = CriticOutcome()

  for concept in concepts:
    for label, code, should_be_correct in (
        ("correct", references.CORRECT[concept], True),
        ("buggy", references.BUGGY[concept], False),
    ):
      print(f"  {concept} [{label}]", flush=True)
      try:
        result = tutor.evaluate_student(
            code, [], attempt, concept=concept, usage=usage)
      except Exception as exc:  # noqa: BLE001
        print(f"    failed: {exc}")
        continue

      outcome.total += 1
      if result.is_correct == should_be_correct:
        outcome.verdict_correct += 1

      if result.model_agreed is not None:
        outcome.model_judged += 1
        if result.model_agreed:
          outcome.model_agreed += 1
        else:
          outcome.disagreements.append({
              "concept": concept,
              "implementation": label,
              "tests_said": result.is_correct,
              "model_said": result.model_is_correct,
              "tests": f"{result.verification.get('passed')}/"
                       f"{result.verification.get('total')}",
          })

      if result.passages:
        outcome.with_passages += 1
        if result.grounded:
          outcome.grounded += 1

      outcome.total_citations += len(result.citations)
      outcome.fabricated_citations += len(result.invalid_citations)
      if result.invalid_citations:
        outcome.fabrications.append({
            "concept": concept,
            "implementation": label,
            "invented": result.invalid_citations,
            "supplied": [p.chunk_id for p in result.passages],
        })

  return outcome


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--retrieval", action="store_true")
  parser.add_argument("--critic", action="store_true")
  parser.add_argument("--all", action="store_true")
  parser.add_argument("-k", type=int, default=config.DEFAULT_K)
  parser.add_argument("--limit", type=int, default=0,
                      help="only this many concepts in the critic run")
  parser.add_argument("--out", help="write the full report as JSON")
  args = parser.parse_args()

  if not (args.retrieval or args.critic or args.all):
    parser.print_help()
    return 1

  report: dict[str, Any] = {
      "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
      "models": {
          "generation": config.GEN_MODEL,
          "embedding": config.EMBED_MODEL,
          "dimensions": config.EMBED_DIM,
          "location": config.GCP_LOCATION,
      },
      "k": args.k,
  }

  try:
    report["corpus"] = retrieval.corpus_stats()
  except Exception as exc:  # noqa: BLE001
    print(f"could not read corpus stats: {exc}")
    return 1

  print("Corpus")
  for key, value in report["corpus"].items():
    print(f"  {key:<20} {value}")

  if args.retrieval or args.all:
    print(f"\nRetrieval (k={args.k})")
    filtered = run_retrieval(args.k, filtered=True)
    unfiltered = run_retrieval(args.k, filtered=False)
    report["retrieval"] = {
        "filtered": filtered.summary(),
        "unfiltered": unfiltered.summary(),
    }

    print(f"  with concept filter:")
    print(f"    hit@1   {_pct(filtered.hit_at_1, filtered.total)}")
    print(f"    hit@{args.k}   {_pct(filtered.hit_at_k, filtered.total)}")
    print(f"  without concept filter:")
    print(f"    hit@1   {_pct(unfiltered.hit_at_1, unfiltered.total)}")
    print(f"    hit@{args.k}   {_pct(unfiltered.hit_at_k, unfiltered.total)}")
    print(f"  median query latency: "
          f"{statistics.median(filtered.latencies_ms):.0f} ms")

    if unfiltered.misses:
      print(f"\n  unfiltered misses ({len(unfiltered.misses)}):")
      for miss in unfiltered.misses:
        print(f"    {miss['query'][:66]}")
        print(f"      wanted one of {miss['expected']}")
        print(f"      got {miss['returned'][:3]}")

  if args.critic or args.all:
    concepts = sorted(references.CORRECT)
    if args.limit:
      concepts = concepts[: args.limit]
    print(f"\nCritic ({len(concepts)} concepts, "
          f"{len(concepts) * 2} evaluations)")

    usage = vertex.Usage()
    outcome = run_critic(concepts, usage)
    report["critic"] = outcome.summary()
    report["usage"] = usage.summary()

    print(f"\n  verdict correct   {_pct(outcome.verdict_correct, outcome.total)}")
    if outcome.model_judged:
      print(f"  model agreed with the tests "
            f"{_pct(outcome.model_agreed, outcome.model_judged)}")
    if outcome.with_passages:
      print(f"  cited a real passage "
            f"{_pct(outcome.grounded, outcome.with_passages)}")
    print(f"  citations {outcome.total_citations}, "
          f"fabricated {outcome.fabricated_citations}")

    if outcome.disagreements:
      print(f"\n  the model read these differently from how they ran:")
      for row in outcome.disagreements:
        print(f"    {row['concept']} [{row['implementation']}]: "
              f"tests {row['tests']} -> {row['tests_said']}, "
              f"model said {row['model_said']}")

    if outcome.fabrications:
      print(f"\n  invented citations:")
      for row in outcome.fabrications:
        print(f"    {row['concept']} [{row['implementation']}]: "
              f"{row['invented']}")

    print(f"\n  tokens {usage.total_tokens:,} over {usage.calls} calls")
    print(f"  {usage.summary()['tokens_per_sec']} tokens/sec, "
          f"p50 {usage.summary()['p50_latency_ms']:.0f} ms, "
          f"p95 {usage.summary()['p95_latency_ms']:.0f} ms")

  if args.out:
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {path}")

  return 0


if __name__ == "__main__":
  sys.exit(main())

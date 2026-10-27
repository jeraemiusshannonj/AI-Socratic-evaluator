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
from socratic import config, coordinator, retrieval, tutor, vertex

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
QUESTIONS_PATH = EVAL_DIR / "retrieval_questions.json"
UNANSWERABLE_PATH = EVAL_DIR / "unanswerable_questions.json"
ROUTING_PATH = EVAL_DIR / "routing_questions.json"


def _min_score() -> float:
  from socratic import tutor as _t
  return _t.MIN_PASSAGE_SCORE


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


# ---------------------------------------------------------------------------
# Does it refuse what the corpus cannot answer?
# ---------------------------------------------------------------------------
_JUDGE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "refused": {"type": "BOOLEAN"},
        "why": {"type": "STRING"},
    },
    "required": ["refused", "why"],
}

_JUDGE_PROMPT = """\
A tutoring system was asked a question its reference material does not
cover. Here is what it said.

Question: {question}

Response: {answer}

Did the response decline, on the grounds that the available material does
not cover the question? Answer true only if it actually said the material
is insufficient. A response that answers the question correctly from
general knowledge, without saying the material does not cover it, is NOT
a refusal -- set refused to false."""


@dataclass
class RefusalOutcome:
  total: int = 0
  refused: int = 0
  declined_structurally: int = 0
  revisions_spent: int = 0
  fabricated: int = 0
  answered_anyway: list[dict] = field(default_factory=list)
  top_scores: list[float] = field(default_factory=list)

  def summary(self) -> dict[str, Any]:
    return {
        "questions": self.total,
        "refused": self.refused,
        "correct_refusal_rate": round(self.refused / self.total, 3)
                                if self.total else 0,
        "declared_material_irrelevant": self.declined_structurally,
        "revisions_spent": self.revisions_spent,
        "fabricated_citations": self.fabricated,
        "top_score_median": round(statistics.median(self.top_scores), 4)
                            if self.top_scores else None,
        "top_score_max": round(max(self.top_scores), 4)
                         if self.top_scores else None,
        "answered_anyway": self.answered_anyway,
    }


def run_refusal(k: int, usage: vertex.Usage) -> RefusalOutcome:
  questions = json.loads(
      UNANSWERABLE_PATH.read_text(encoding="utf-8"))["questions"]
  outcome = RefusalOutcome(total=len(questions))

  for item in questions:
    print(f"  {item['query'][:66]}", flush=True)
    try:
      answer = coordinator.answer_question(
          item["query"], concept=item.get("concept", ""), k=k, usage=usage)
    except Exception as exc:  # noqa: BLE001
      print(f"    failed: {exc}")
      outcome.total -= 1
      continue

    if answer.passages:
      outcome.top_scores.append(max(p.score for p in answer.passages))
    if answer.declined:
      outcome.declined_structurally += 1
    outcome.revisions_spent += answer.revisions
    if answer.invalid_citations:
      outcome.fabricated += len(answer.invalid_citations)

    verdict = vertex.generate(
        _JUDGE_PROMPT.format(question=item["query"], answer=answer.text),
        temperature=0.0, response_schema=_JUDGE_SCHEMA, usage=usage)
    try:
      refused = bool(json.loads(verdict).get("refused"))
    except Exception:  # noqa: BLE001
      refused = False

    if refused:
      outcome.refused += 1
    else:
      outcome.answered_anyway.append({
          "query": item["query"],
          "why_absent": item.get("why_absent", ""),
          "answer": answer.text[:220],
          "cited": answer.valid_citations,
      })

  return outcome


# ---------------------------------------------------------------------------
# Does the coordinator route to the right path?
# ---------------------------------------------------------------------------
@dataclass
class RoutingOutcome:
  total: int = 0
  path_correct: int = 0
  tool_correct: int = 0
  tool_checked: int = 0
  mistakes: list[dict] = field(default_factory=list)

  def summary(self) -> dict[str, Any]:
    return {
        "questions": self.total,
        "path_correct": self.path_correct,
        "path_accuracy": round(self.path_correct / self.total, 3)
                         if self.total else 0,
        "tool_correct": self.tool_correct,
        "tool_accuracy": round(self.tool_correct / self.tool_checked, 3)
                         if self.tool_checked else None,
        "mistakes": self.mistakes,
    }


def run_routing(usage: vertex.Usage) -> RoutingOutcome:
  questions = json.loads(ROUTING_PATH.read_text(encoding="utf-8"))["questions"]
  outcome = RoutingOutcome(total=len(questions))

  for item in questions:
    plan = coordinator.plan_turn(
        item["query"], concept=item.get("concept", ""), usage=usage)
    chosen = [c.name for c in plan.calls]
    paths = set(plan.paths)
    want = item["expect_path"]

    if want == "both":
      path_ok = {"structured", "unstructured"} <= paths
    else:
      path_ok = paths == {want}

    outcome.path_correct += path_ok

    tool_ok = None
    if item.get("expect_tool"):
      outcome.tool_checked += 1
      tool_ok = item["expect_tool"] in chosen
      outcome.tool_correct += tool_ok

    print(f"  {'ok  ' if path_ok else 'MISS'} {item['query'][:56]}")
    print(f"       wanted {want}, chose {sorted(paths) or ['nothing']} "
          f"via {chosen}")

    if not path_ok or tool_ok is False:
      outcome.mistakes.append({
          "query": item["query"],
          "expected_path": want,
          "expected_tool": item.get("expect_tool"),
          "chose": chosen,
          "reasoning": plan.reasoning,
      })

  return outcome


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--retrieval", action="store_true")
  parser.add_argument("--critic", action="store_true")
  parser.add_argument("--routing", action="store_true",
                      help="does the coordinator pick the right path")
  parser.add_argument("--refusal", action="store_true",
                      help="does it decline what the corpus cannot answer")
  parser.add_argument("--all", action="store_true")
  parser.add_argument("-k", type=int, default=config.DEFAULT_K)
  parser.add_argument("--limit", type=int, default=0,
                      help="only this many concepts in the critic run")
  parser.add_argument("--out", help="write the full report as JSON")
  args = parser.parse_args()

  if not (args.retrieval or args.critic or args.routing
          or args.refusal or args.all):
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

  if args.routing or args.all:
    print("\nRouting")
    usage = vertex.Usage()
    routing = run_routing(usage)
    report["routing"] = routing.summary()
    report.setdefault("usage_by_stage", {})["routing"] = usage.summary()
    print(f"\n  path correct  {_pct(routing.path_correct, routing.total)}")
    if routing.tool_checked:
      print(f"  tool correct  "
            f"{_pct(routing.tool_correct, routing.tool_checked)}")

  if args.refusal or args.all:
    print("\nRefusal on questions the corpus cannot answer")
    usage = vertex.Usage()
    refusal = run_refusal(args.k, usage)
    report["refusal"] = refusal.summary()
    report.setdefault("usage_by_stage", {})["refusal"] = usage.summary()

    print(f"\n  correct refusal  "
          f"{_pct(refusal.refused, refusal.total)}")
    print(f"  declared the material irrelevant  "
          f"{_pct(refusal.declined_structurally, refusal.total)}")
    print(f"  revisions spent on refusals  {refusal.revisions_spent} "
          f"(0 is correct; a refusal is not an ungrounded answer)")
    print(f"  fabricated citations  {refusal.fabricated}")
    if refusal.top_scores:
      print(f"  top passage score on unanswerable questions: "
            f"median {statistics.median(refusal.top_scores):.4f}, "
            f"max {max(refusal.top_scores):.4f}")
      print(f"  (compare with MIN_PASSAGE_SCORE = {_min_score()}; a "
            f"threshold between these and the answerable scores is what "
            f"would let retrieval itself reject an off-corpus question)")
    for row in refusal.answered_anyway:
      print(f"\n  ANSWERED ANYWAY: {row['query'][:66]}")
      print(f"    absent because: {row['why_absent'][:90]}")
      print(f"    said: {row['answer'][:110]}")

  # Cost per request, if prices are configured.
  totals = report.get("usage") or {}
  stage_usage = report.get("usage_by_stage", {})
  calls = totals.get("calls", 0) + sum(
      u.get("calls", 0) for u in stage_usage.values())
  prompt_tokens = totals.get("prompt_tokens", 0) + sum(
      u.get("prompt_tokens", 0) for u in stage_usage.values())
  output_tokens = totals.get("output_tokens", 0) + sum(
      u.get("output_tokens", 0) for u in stage_usage.values())

  if calls:
    cost = config.cost_of(prompt_tokens, output_tokens)
    report["cost"] = {
        "calls": calls,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_cost_usd": round(cost, 6) if cost is not None else None,
        "cost_per_call_usd": round(cost / calls, 6)
                             if cost is not None else None,
        "pricing_configured": cost is not None,
    }
    print(f"\nCost")
    print(f"  {calls} calls, {prompt_tokens:,} in / {output_tokens:,} out")
    if cost is None:
      print("  cost per request: not configured. Put the current prices for")
      print(f"  {config.GEN_MODEL} in .env as PRICE_PER_1M_INPUT_TOKENS and")
      print("  PRICE_PER_1M_OUTPUT_TOKENS, then re-run.")
    else:
      print(f"  ${cost:.4f} total, ${cost / calls:.6f} per request")

  if args.out:
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {path}")

  return 0


if __name__ == "__main__":
  sys.exit(main())

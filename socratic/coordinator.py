"""Coordinator: route the turn, then critique, then revise once if ungrounded.

Until now `retrieval.py` described tools "the agent can choose between"
and a "coordinator agent" that did not exist. The pipeline always did the
same thing in the same order, and the citation check detected a fabricated
source and then discarded the finding. This closes both gaps.

Three steps, each a separate decision rather than a fixed sequence:

**Route.** The coordinator asks the model which retrieval tools this turn
actually needs. "Why does this loop boundary never terminate?" wants the
corpus. "Have I got this wrong before?" wants an aggregation over the
student's records and no embedding at all. "What should I practise next?"
is a ranking over records that a vector search cannot answer however well
it retrieves. Routing wrong is cheap and visible; not routing at all means
every question pays for every path.

**Critique.** The retrieved material goes to the critic with chunk ids,
and the answer must cite the ids it relied on.

**Revise, once.** If the critic cites an id it was never given, or claims
grounding while citing nothing, it gets exactly one more attempt with the
specific failure named and the same passages re-supplied. Bounded at one:
a model that fabricates twice will fabricate ten times, and an unbounded
loop turns a bad answer into an expensive bad answer.

Deliberately hand-written rather than built on an agent framework. The
control flow is three steps and a conditional; what a framework would add
here is indirection, not capability.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from . import retrieval, tutor, vertex
from .retrieval import Passage

log = logging.getLogger(__name__)

MAX_REVISIONS = 1
MAX_TOOL_CALLS = 3

# What the router is allowed to choose from. The description is what the
# model sees, so each says what the tool is *for* rather than what it does
# -- "has this student struggled before" routes better than "queries the
# interactions collection".
TOOL_SPECS: dict[str, dict[str, Any]] = {
    "semantic_search": {
        "path": "unstructured",
        "use_for": (
            "Explaining a concept or a mechanism. Anything whose answer is "
            "prose that has to be read and understood: why something "
            "behaves the way it does, what a term means, how an algorithm "
            "works."
        ),
        "args": {"query": "the question, rephrased for retrieval"},
    },
    "lookup_misconception": {
        "path": "structured",
        "use_for": (
            "Naming the specific bug this exercise contains. An exact "
            "record lookup, not a search."
        ),
        "args": {},
    },
    "student_concept_history": {
        "path": "structured",
        "use_for": (
            "Whether this student has attempted this concept before and how "
            "it went. A computation over their records."
        ),
        "args": {},
    },
    "weakest_concepts": {
        "path": "structured",
        "use_for": (
            "What the student should practise next, or which concepts they "
            "fail most. A ranking over records; no amount of semantic "
            "search can answer it."
        ),
        "args": {},
    },
    "lookup_test_cases": {
        "path": "structured",
        "use_for": "What this concept's code is checked against.",
        "args": {},
    },
}

_ROUTE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "tools": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "query": {
                        "type": "STRING",
                        "description": "Only for semantic_search.",
                    },
                },
                "required": ["name"],
            },
        },
        "reasoning": {"type": "STRING"},
    },
    "required": ["tools", "reasoning"],
}

_ROUTE_SYSTEM = (
    "You route a student's question to the retrieval tools that can answer "
    "it. You pick the fewest tools that suffice. You never answer the "
    "question yourself."
)


def _tool_catalogue() -> str:
  lines = []
  for name, spec in TOOL_SPECS.items():
    lines.append(f"- {name} ({spec['path']}): {spec['use_for']}")
  return "\n".join(lines)


_ROUTE_PROMPT = """\
Student's concept: {concept}
Student asked: {question}

Available tools:
{catalogue}

Choose the tools that answer this question. Most questions need one. Use
semantic_search only when the answer is an explanation; use the structured
tools when the answer is a fact about this student or this exercise.

Return at most {max_tools}."""


@dataclass
class ToolCall:
  name: str
  query: str = ""
  ok: bool = True
  error: str = ""
  result: Any = None

  @property
  def path(self) -> str:
    return TOOL_SPECS.get(self.name, {}).get("path", "unknown")

  def to_dict(self) -> dict[str, Any]:
    summary: Any = self.result
    if isinstance(self.result, list) and self.result and isinstance(
        self.result[0], Passage):
      summary = [p.chunk_id for p in self.result]
    return {
        "name": self.name,
        "path": self.path,
        "query": self.query,
        "ok": self.ok,
        "error": self.error,
        "result": summary,
    }


@dataclass
class Plan:
  calls: list[ToolCall] = field(default_factory=list)
  reasoning: str = ""
  routed_by: str = "model"

  @property
  def paths(self) -> list[str]:
    return sorted({c.path for c in self.calls})

  def to_dict(self) -> dict[str, Any]:
    return {
        "reasoning": self.reasoning,
        "routed_by": self.routed_by,
        "paths": self.paths,
        "calls": [c.to_dict() for c in self.calls],
    }


@dataclass
class Answer:
  text: str
  citations: list[str] = field(default_factory=list)
  valid_citations: list[str] = field(default_factory=list)
  invalid_citations: list[str] = field(default_factory=list)
  passages: list[Passage] = field(default_factory=list)
  plan: Plan = field(default_factory=Plan)
  revisions: int = 0
  revision_reason: str = ""
  usage: dict[str, Any] = field(default_factory=dict)

  @property
  def grounded(self) -> bool:
    return bool(self.valid_citations)

  def to_dict(self) -> dict[str, Any]:
    return {
        "answer": self.text,
        "citations": self.citations,
        "valid_citations": self.valid_citations,
        "invalid_citations": self.invalid_citations,
        "grounded": self.grounded,
        "revisions": self.revisions,
        "revision_reason": self.revision_reason,
        "plan": self.plan.to_dict(),
        "sources": [
            {
                "chunk_id": p.chunk_id,
                "title": p.source_title,
                "section": p.section,
                "url": p.source_url,
                "score": p.score,
            }
            for p in self.passages
        ],
        "usage": self.usage,
    }


# ---------------------------------------------------------------------------
# Step 1: route
# ---------------------------------------------------------------------------
def plan_turn(
    question: str, *, concept: str, usage: vertex.Usage | None = None
) -> Plan:
  """Ask the model which tools this turn needs."""
  prompt = _ROUTE_PROMPT.format(
      concept=concept or "(none selected)",
      question=question,
      catalogue=_tool_catalogue(),
      max_tools=MAX_TOOL_CALLS,
  )

  try:
    raw = vertex.generate(
        prompt,
        system_instruction=_ROUTE_SYSTEM,
        temperature=0.0,
        response_schema=_ROUTE_SCHEMA,
        usage=usage,
    )
    data = json.loads(raw)
  except Exception as exc:  # noqa: BLE001
    # A routing failure should not lose the turn. Semantic search is the
    # right default: it is the path that answers "explain this", which is
    # most of what students ask.
    log.warning("Routing failed, defaulting to semantic search: %s", exc)
    return Plan(
        calls=[ToolCall(name="semantic_search", query=question)],
        reasoning=f"router unavailable ({type(exc).__name__})",
        routed_by="fallback",
    )

  calls: list[ToolCall] = []
  for item in data.get("tools", [])[:MAX_TOOL_CALLS]:
    name = str(item.get("name", "")).strip()
    if name not in TOOL_SPECS:
      log.warning("Router chose unknown tool %r; ignoring", name)
      continue
    calls.append(ToolCall(name=name, query=str(item.get("query") or question)))

  if not calls:
    calls = [ToolCall(name="semantic_search", query=question)]

  return Plan(calls=calls, reasoning=str(data.get("reasoning", "")))


# ---------------------------------------------------------------------------
# Step 2: execute
# ---------------------------------------------------------------------------
def execute_plan(
    plan: Plan, *, concept: str, student_id: str, k: int = 3
) -> list[Passage]:
  """Run the chosen tools, returning any passages they produced."""
  passages: list[Passage] = []

  for call in plan.calls:
    try:
      if call.name == "semantic_search":
        found = retrieval.semantic_search(
            call.query or concept, k=k, concept=concept or None)
        call.result = found
        passages.extend(found)
      elif call.name == "lookup_misconception":
        call.result = retrieval.lookup_misconception(concept)
      elif call.name == "student_concept_history":
        call.result = retrieval.student_concept_history(student_id, concept)
      elif call.name == "weakest_concepts":
        call.result = retrieval.weakest_concepts(student_id)
      elif call.name == "lookup_test_cases":
        call.result = retrieval.lookup_test_cases(concept)
      else:
        call.ok, call.error = False, "no handler"
    except Exception as exc:  # noqa: BLE001
      log.warning("Tool %s failed: %s", call.name, exc)
      call.ok, call.error = False, str(exc)

  return passages


def _format_structured(plan: Plan) -> str:
  """Render the non-passage tool results for the critic's prompt."""
  blocks = []
  for call in plan.calls:
    if call.name == "semantic_search" or call.result in (None, [], {}):
      continue
    if not call.ok:
      blocks.append(f"{call.name}: unavailable ({call.error})")
      continue
    blocks.append(f"{call.name}: {json.dumps(call.result, default=str)[:1200]}")
  return "\n\n".join(blocks) or "None."


# ---------------------------------------------------------------------------
# Step 3: critique, and revise once if ungrounded
# ---------------------------------------------------------------------------
_ANSWER_SYSTEM = (
    "You are a Socratic computer science tutor. You answer from the "
    "material you are given and you cite it. You do not hand over "
    "solutions, and you never invent a source."
)

_ANSWER_PROMPT = """\
Concept: {concept}
Student asked: {question}

### Retrieved passages
{passages}

### Structured results
{structured}

Answer the student from the material above. Be direct and short. If the
material does not answer the question, say so plainly rather than filling
the gap from memory.

In `citations`, list the ids in square brackets of the passages you
actually used, exactly as written above. Cite only ids shown above. If you
used none, return an empty list."""

_REVISION_NOTE = """\

### Your previous attempt was rejected
{problem}

The passages above are the only ones available to you. Answer again using
them. If none of them support an answer, say the material does not cover
it and return an empty citations list -- that is a correct response, not a
failure."""

_ANSWER_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "answer": {"type": "STRING"},
        "citations": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["answer", "citations"],
}


def answer_question(
    question: str,
    *,
    concept: str = "",
    student_id: str = "anonymous",
    k: int = 3,
    usage: vertex.Usage | None = None,
) -> Answer:
  """Route, retrieve, answer, and revise once if the answer is not grounded."""
  usage = usage or vertex.Usage()

  plan = plan_turn(question, concept=concept, usage=usage)
  passages = execute_plan(plan, concept=concept, student_id=student_id, k=k)
  supplied = {p.chunk_id for p in passages}

  base_prompt = _ANSWER_PROMPT.format(
      concept=concept or "(none selected)",
      question=question,
      passages=tutor._format_passages(passages),
      structured=_format_structured(plan),
  )

  prompt = base_prompt
  revisions = 0
  revision_reason = ""
  text, citations = "", []

  while True:
    raw = vertex.generate(
        prompt,
        system_instruction=_ANSWER_SYSTEM,
        temperature=0.1,
        response_schema=_ANSWER_SCHEMA,
        usage=usage,
    )
    data = tutor._parse_evaluation(raw)
    text = str(data.get("answer") or data.get("feedback") or "")
    citations = tutor._normalise_citations(data.get("citations"), text)

    problem = tutor.check_grounding(citations, supplied, bool(passages))
    if not problem or revisions >= MAX_REVISIONS:
      if problem:
        log.warning("Answer still ungrounded after %d revision(s): %s",
                    revisions, problem)
      break

    revisions += 1
    revision_reason = problem
    log.info("Revising answer (attempt %d): %s", revisions + 1, problem)
    prompt = base_prompt + _REVISION_NOTE.format(problem=problem)

  return Answer(
      text=text,
      citations=citations,
      valid_citations=[c for c in citations if c in supplied],
      invalid_citations=[c for c in citations if c not in supplied],
      passages=passages,
      plan=plan,
      revisions=revisions,
      revision_reason=revision_reason,
      usage=usage.summary(),
  )

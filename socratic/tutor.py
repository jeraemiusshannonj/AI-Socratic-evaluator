"""Exercise generation and Socratic evaluation, both grounded in retrieval.

Replaces the previous `chains.py`, which used ChromaDB and Groq. Two
substantive changes beyond swapping providers:

**The critic now retrieves.** The old evaluator was handed only the
expected-fix string and asked to judge the student's code against it. No
corpus was consulted, so its explanations came entirely from the model's
own knowledge and there was nothing to cite. Here, each evaluation pulls
passages from `cs_chunks` scoped to the concept, puts them in the prompt
with their chunk ids, and requires the model to cite the ones it used.
Citations are then checked against the passages actually supplied, so a
fabricated id is detected rather than trusted. That check is the basis of
the citation-accuracy metric.

**A concept with no taxonomy entry is an error, not a default.** The old
`generate_flaw` fell back to "introduce a subtle logic error related to
loop termination" whenever the concept was missing from the CSV. Eleven of
the twenty-two topics in the selector are missing, so half of them
generated a loop-termination bug no matter what the student picked, while
appearing to work. Now it raises.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from . import config, retrieval, vertex
from .retrieval import Passage

log = logging.getLogger(__name__)

# How many passages to put in front of the critic. More is not better: the
# prompt grows, latency grows, and a weakly related passage invites a
# citation that is technically valid and substantively irrelevant.
EVAL_PASSAGE_COUNT = 3

# Passages below this cosine score are dropped rather than padded in. If
# nothing clears it, the critic is told explicitly that the corpus has
# nothing for this question, which is better than it quietly reasoning
# unsourced while the response still shows a citations field.
MIN_PASSAGE_SCORE = 0.45

_CODE_FENCE = re.compile(r"```(?:python)?\s*(.*?)\s*```", re.DOTALL | re.I)
_CITATION = re.compile(r"\[([^\]\s]+#\d+)\]")


class ConceptNotInTaxonomy(KeyError):
  """Raised when a concept has no misconception entry.

  Carries the available concepts, because the useful question when this
  fires is "what should I have asked for".
  """

  def __init__(self, concept: str, available: Sequence[str]) -> None:
    self.concept = concept
    self.available = list(available)
    super().__init__(
        f"No misconception entry for {concept!r}. The taxonomy has "
        f"{len(self.available)} concepts: {', '.join(sorted(self.available))}. "
        f"Add a row to misconceptions.csv and re-run "
        f"`python -m scripts.migrate_sqlite`."
    )


@dataclass
class Exercise:
  concept: str
  flawed_code: str
  expected_fix: str
  # "exact" when the concept matched a taxonomy row directly, "alias" when
  # it matched through the alias list. Worth surfacing: an alias match means
  # the bug injected is for a related variant, not this exact one.
  taxonomy_match: str

  def to_dict(self) -> dict[str, Any]:
    return {
        "concept": self.concept,
        "flawed_code": self.flawed_code,
        "expected_fix": self.expected_fix,
        "taxonomy_match": self.taxonomy_match,
    }


@dataclass
class Evaluation:
  is_correct: bool
  reasoning: str
  feedback: str
  # Chunk ids the model said it used.
  citations: list[str] = field(default_factory=list)
  # Of those, the ones that were actually in the prompt.
  valid_citations: list[str] = field(default_factory=list)
  # Ids the model produced that were never supplied. A non-empty list here
  # means the model invented a source, which is the thing worth catching.
  invalid_citations: list[str] = field(default_factory=list)
  passages: list[Passage] = field(default_factory=list)
  usage: dict[str, Any] = field(default_factory=dict)

  @property
  def grounded(self) -> bool:
    """Did the critic cite at least one passage it was actually given?"""
    return bool(self.valid_citations)

  def to_dict(self) -> dict[str, Any]:
    return {
        "is_correct": self.is_correct,
        "reasoning": self.reasoning,
        "feedback": self.feedback,
        "citations": self.citations,
        "valid_citations": self.valid_citations,
        "invalid_citations": self.invalid_citations,
        "grounded": self.grounded,
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
# Exercise generation -- structured retrieval only
# ---------------------------------------------------------------------------
_GENERATOR_SYSTEM = (
    "You generate deliberately flawed Python functions for debugging "
    "exercises. You output a function definition and nothing else."
)

_GENERATOR_PROMPT = """\
TOPIC: {concept}
BUG YOU MUST INJECT: {misconception}

Rules:
1. Output only a Python function definition starting with `def`.
2. The function must contain the bug described above, introduced naturally
   as a plausible mistake a student would make.
3. No driver code, no test data, no prints outside the function.
4. No comments, and nothing that hints at where the bug is.
5. The code must be syntactically valid and must run -- it should produce
   wrong answers or loop forever, not raise SyntaxError.

Generate the flawed function for {concept}."""


def generate_flaw(concept: str, *, usage: vertex.Usage | None = None) -> Exercise:
  """Build a flawed exercise for a concept, using the taxonomy as the spec.

  This is the structured path on its own: an exact record lookup keyed by
  concept, with no embeddings involved. The misconception text is the
  specification handed to the generator, which is why a missing entry has
  to be an error -- without it there is nothing to specify.
  """
  entry = retrieval.lookup_misconception(concept)
  if not entry:
    from . import db  # local import keeps module import side-effect free
    available = db.misconceptions().distinct("concept")
    raise ConceptNotInTaxonomy(concept, available)

  misconception = entry.get("misconception", "")
  matched = entry.get("concept", concept)
  match_kind = "exact" if matched == concept else "alias"
  if match_kind == "alias":
    log.warning(
        "Concept %r matched taxonomy entry %r by alias; the injected bug is "
        "for the related variant.", concept, matched)

  raw = vertex.generate(
      _GENERATOR_PROMPT.format(concept=concept, misconception=misconception),
      system_instruction=_GENERATOR_SYSTEM,
      temperature=0.3,
      usage=usage,
  )

  return Exercise(
      concept=concept,
      flawed_code=extract_code(raw),
      expected_fix=misconception,
      taxonomy_match=match_kind,
  )


def extract_code(text: str) -> str:
  """Pull Python out of a model response.

  Unlike the previous implementation this does not strip comments. That
  stripping used a regex that could not tell a `#` inside a string literal
  from a real comment, and the generator is already told not to emit
  comments, so removing them afterwards only risked corrupting valid code.
  """
  match = _CODE_FENCE.search(text)
  body = match.group(1) if match else text
  lines = [line.rstrip() for line in body.splitlines()]
  # Drop leading prose before the first `def`, which the model occasionally
  # adds despite instructions.
  for index, line in enumerate(lines):
    if line.lstrip().startswith(("def ", "import ", "from ", "class ")):
      lines = lines[index:]
      break
  while lines and not lines[-1].strip():
    lines.pop()
  return "\n".join(lines)


# ---------------------------------------------------------------------------
# Socratic evaluation -- both paths
# ---------------------------------------------------------------------------
_EVAL_SYSTEM = (
    "You are a Socratic computer science tutor. You never hand over the "
    "fix. You judge the student's code structurally, and you ground your "
    "explanation in the reference passages you are given."
)

_EVAL_PROMPT = """\
### Topic
{concept}

### The bug the student is looking for
{expected_fix}

### Reference passages
{passages}

### Student's current code
```python
{code}
```

### Recent conversation
{history}

### How to respond
{scaffolding}

Judge the CODE, not the conversation. If the code no longer contains the
bug described above, set is_correct to true even if the student has not
explained it. If the bug is still there, set is_correct to false and give
one hint at the level described above.

Ground your reasoning in the reference passages. In `citations`, list the
ids in square brackets of the passages you actually relied on, exactly as
written. Cite only passages shown above. If none of them are relevant,
return an empty list rather than citing something unrelated."""

_EVAL_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "reasoning": {
            "type": "STRING",
            "description": "Why the code is or is not fixed, by line.",
        },
        "is_correct": {"type": "BOOLEAN"},
        "feedback": {
            "type": "STRING",
            "description": "What the student reads. Socratic, one hint.",
        },
        "citations": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "Chunk ids of passages relied on.",
        },
    },
    "required": ["reasoning", "is_correct", "feedback", "citations"],
}


def _scaffolding_for(attempt_count: int) -> str:
  if attempt_count <= 1:
    return (
        "Ask one open-ended question that pushes the student to re-read the "
        "logic themselves. Name no line numbers."
    )
  if attempt_count == 2:
    return (
        "Point at the specific line or operator where the logic goes wrong, "
        "without stating the correction."
    )
  return (
      "Say plainly what needs to change and why, but let the student write "
      "the change."
  )


def _format_history(chat_history: Sequence[dict], limit: int = 6) -> str:
  if not chat_history:
    return "No conversation yet."
  lines = []
  for message in list(chat_history)[-limit:]:
    role = "Student" if message.get("role") == "user" else "Tutor"
    lines.append(f"[{role}]: {message.get('content', '')}")
  return "\n".join(lines)


def _retrieval_query(
    concept: str, misconception: str, chat_history: Sequence[dict]
) -> str:
  """Compose one query for the semantic path.

  The concept and the misconception describe the mechanism to explain; the
  student's latest message, when they have actually asked something, says
  what they are confused about. Both matter, and one combined query costs
  one embedding call instead of two.
  """
  parts = [concept, misconception]
  for message in reversed(list(chat_history)):
    if message.get("role") == "user":
      text = (message.get("content") or "").strip()
      if len(text) >= 20:
        parts.append(text)
      break
  return " ".join(parts)[:1500]


def _format_passages(passages: Sequence[Passage]) -> str:
  if not passages:
    return (
        "None. The corpus has nothing relevant for this question, so reason "
        "from the code alone and return an empty citations list."
    )
  blocks = []
  for passage in passages:
    header = f"[{passage.chunk_id}] {passage.source_title}"
    if passage.section:
      header += f" -- {passage.section}"
    blocks.append(f"{header}\n{passage.text}")
  return "\n\n".join(blocks)


def evaluate_student(
    current_code: str,
    chat_history: Sequence[dict],
    attempt_count: int,
    *,
    concept: str,
    expected_fix: str = "",
    usage: vertex.Usage | None = None,
    k: int = EVAL_PASSAGE_COUNT,
) -> Evaluation:
  """Judge the student's code and explain it from retrieved sources.

  Both paths are used, in order: the taxonomy supplies the bug being looked
  for (structured), then the corpus supplies the explanation (semantic,
  scoped to the concept). The concept filter matters here -- an unfiltered
  search on "binary search mid low high" returns passages from every
  chapter that happens to discuss array indices.
  """
  if not expected_fix:
    entry = retrieval.lookup_misconception(concept)
    expected_fix = (entry or {}).get("misconception", "") or (
        "Find and correct the logic bug."
    )

  query = _retrieval_query(concept, expected_fix, chat_history)
  try:
    passages = retrieval.semantic_search(
        query, k=k, concept=concept, min_score=MIN_PASSAGE_SCORE
    )
  except Exception as exc:  # noqa: BLE001
    # A retrieval failure should degrade the answer, not lose the student's
    # turn. The response records that it was ungrounded.
    log.warning("Retrieval failed, evaluating without passages: %s", exc)
    passages = []

  prompt = _EVAL_PROMPT.format(
      concept=concept,
      expected_fix=expected_fix,
      passages=_format_passages(passages),
      code=current_code,
      history=_format_history(chat_history),
      scaffolding=_scaffolding_for(attempt_count),
  )

  raw = vertex.generate(
      prompt,
      system_instruction=_EVAL_SYSTEM,
      temperature=0.0,
      response_schema=_EVAL_SCHEMA,
      usage=usage,
  )

  data = _parse_evaluation(raw)
  supplied = {p.chunk_id for p in passages}
  cited = _normalise_citations(data.get("citations"), data.get("feedback", ""))

  return Evaluation(
      is_correct=bool(data.get("is_correct", False)),
      reasoning=str(data.get("reasoning", "")),
      feedback=str(data.get("feedback", "")),
      citations=cited,
      valid_citations=[c for c in cited if c in supplied],
      invalid_citations=[c for c in cited if c not in supplied],
      passages=list(passages),
      usage=usage.summary() if usage else {},
  )


def _parse_evaluation(raw: str) -> dict[str, Any]:
  """Parse the structured response, tolerating a fenced or prefixed body.

  `response_schema` makes well-formed JSON the overwhelming case, but a
  model that ignores it once should not cost the student their turn, so a
  fenced or surrounded object is recovered rather than raising.
  """
  import json

  text = raw.strip()
  try:
    return json.loads(text)
  except json.JSONDecodeError:
    pass

  match = _CODE_FENCE.search(text)
  if match:
    try:
      return json.loads(match.group(1))
    except json.JSONDecodeError:
      pass

  start, end = text.find("{"), text.rfind("}")
  if 0 <= start < end:
    try:
      return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
      pass

  log.error("Could not parse evaluation response: %s", text[:300])
  return {
      "reasoning": "The evaluator returned a malformed response.",
      "is_correct": False,
      "feedback": (
          "Something went wrong on my side reading that. Try again, or "
          "describe what you changed."
      ),
      "citations": [],
  }


def _normalise_citations(
    raw_citations: Any, feedback: str
) -> list[str]:
  """Collect cited chunk ids from the citations field and the prose.

  Models sometimes put the bracketed id inline in the feedback instead of
  in the structured field, or in both. Taking the union, de-duplicated and
  order-preserving, means a correct citation in the wrong place still
  counts, and a duplicate does not inflate the metric.
  """
  collected: list[str] = []

  if isinstance(raw_citations, (list, tuple)):
    for item in raw_citations:
      value = str(item).strip().strip("[]")
      if value:
        collected.append(value)
  elif isinstance(raw_citations, str):
    collected.extend(_CITATION.findall(raw_citations))

  collected.extend(_CITATION.findall(feedback or ""))

  seen: set[str] = set()
  ordered: list[str] = []
  for value in collected:
    if value not in seen:
      seen.add(value)
      ordered.append(value)
  return ordered


# ---------------------------------------------------------------------------
# Backwards-compatible shim
# ---------------------------------------------------------------------------
def evaluate_student_multi_turn(
    current_code: str,
    chat_history: Sequence[dict],
    attempt_count: int,
    expected_fix: str = "",
    concept: str = "Computer Science Logic",
) -> dict[str, Any]:
  """The old `chains.evaluate_student_multi_turn` signature, as a dict.

  Kept so nothing that imported the old module breaks silently. Note the
  positional-argument trap it came with: `main.py` called this with four
  positional arguments and never passed `concept`, so every evaluation ran
  as "Computer Science Logic" and no concept-scoped retrieval was possible.
  New code should call `evaluate_student`, where `concept` is
  keyword-only and required.
  """
  return evaluate_student(
      current_code,
      chat_history,
      attempt_count,
      concept=concept,
      expected_fix=expected_fix,
  ).to_dict()

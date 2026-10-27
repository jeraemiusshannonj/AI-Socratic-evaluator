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

from . import config, retrieval, verify, vertex
from .retrieval import Passage

log = logging.getLogger(__name__)

# How many passages to put in front of the critic. More is not better: the
# prompt grows, latency grows, and a weakly related passage invites a
# citation that is technically valid and substantively irrelevant.
EVAL_PASSAGE_COUNT = 3

# Bounded at one; see check_grounding.
MAX_REVISIONS = 1

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
  # The interface the generated code must expose. Shown to the student so
  # they know what they are meant to be fixing, and relied on by the test
  # cases, which call the function by name.
  signature: str = ""
  contract: str = ""

  def to_dict(self) -> dict[str, Any]:
    return {
        "concept": self.concept,
        "flawed_code": self.flawed_code,
        "expected_fix": self.expected_fix,
        "taxonomy_match": self.taxonomy_match,
        "signature": self.signature,
        "contract": self.contract,
    }


@dataclass
class Evaluation:
  is_correct: bool
  reasoning: str
  feedback: str
  # What the model concluded on its own, kept separate from `is_correct`.
  # When test cases exist they decide correctness and this records whether
  # the model agreed -- a disagreement rate is a far more interesting
  # number than either judgement alone.
  model_is_correct: bool | None = None
  judged_by: str = "model"
  verification: dict[str, Any] = field(default_factory=dict)
  # Chunk ids the model said it used.
  citations: list[str] = field(default_factory=list)
  # Of those, the ones that were actually in the prompt.
  valid_citations: list[str] = field(default_factory=list)
  # Ids the model produced that were never supplied. A non-empty list here
  # means the model invented a source, which is the thing worth catching.
  invalid_citations: list[str] = field(default_factory=list)
  passages: list[Passage] = field(default_factory=list)
  # Whether the critic judged the retrieved passages to bear on this
  # question at all. False with no citations is a deliberate decline, not
  # an ungrounded answer.
  material_relevant: bool = True
  # How many times the answer was sent back for failing the citation
  # check, and why the last rejection happened.
  revisions: int = 0
  revision_reason: str = ""
  usage: dict[str, Any] = field(default_factory=dict)

  @property
  def grounded(self) -> bool:
    """Did the critic cite at least one passage it was actually given?"""
    return bool(self.valid_citations)

  @property
  def declined(self) -> bool:
    """Did it decline on the grounds that the material does not cover it?"""
    return bool(self.passages) and not self.material_relevant

  @property
  def model_agreed(self) -> bool | None:
    """Did the model's own verdict match what running the code showed?"""
    if self.model_is_correct is None or self.judged_by != "tests":
      return None
    return self.model_is_correct == self.is_correct

  def to_dict(self) -> dict[str, Any]:
    return {
        "is_correct": self.is_correct,
        "model_is_correct": self.model_is_correct,
        "judged_by": self.judged_by,
        "model_agreed": self.model_agreed,
        "verification": self.verification,
        "reasoning": self.reasoning,
        "feedback": self.feedback,
        "citations": self.citations,
        "valid_citations": self.valid_citations,
        "invalid_citations": self.invalid_citations,
        "grounded": self.grounded,
        "material_relevant": self.material_relevant,
        "declined": self.declined,
        "revisions": self.revisions,
        "revision_reason": self.revision_reason,
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
REQUIRED INTERFACE: {signature}
WHAT IT MUST DO: {contract}
BUG YOU MUST INJECT: {misconception}

Rules:
1. Output only the definition, starting with `def` or `class`.
2. Use exactly the interface given above -- the same name and the same
   parameters. Anything else is wrong even if the code is good.
3. Apart from the injected bug, follow the contract exactly.
4. The bug must be introduced naturally, as a plausible mistake a student
   would make, not as something obviously out of place.
5. No driver code, no test data, no prints outside the definition.
6. No comments, and nothing that hints at where the bug is.
7. The code must be syntactically valid and must run. It should produce
   wrong answers or fail to terminate, not raise SyntaxError.

Generate the flawed implementation for {concept}."""


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
      _GENERATOR_PROMPT.format(
          concept=concept,
          misconception=misconception,
          signature=entry.get("signature") or "a single function",
          contract=entry.get("contract") or "Behave correctly apart from "
                                            "the injected bug.",
      ),
      system_instruction=_GENERATOR_SYSTEM,
      temperature=0.3,
      usage=usage,
  )

  return Exercise(
      concept=concept,
      flawed_code=extract_code(raw),
      expected_fix=misconception,
      taxonomy_match=match_kind,
      signature=entry.get("signature", ""),
      contract=entry.get("contract", ""),
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

### Test results
{verification}

The test results above come from actually running the student's code. They
are fact. Where your reading of the code disagrees with them, the test
results are right and your reading is wrong. Use the failing case names to
say something specific about what breaks.

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
written. Cite only passages shown above.

If none of the passages bear on this question, set `material_relevant` to
false, return an empty citations list, and say so. That is a correct
response, not a failure -- do not cite something unrelated to fill the
field."""

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
        "material_relevant": {
            "type": "BOOLEAN",
            "description": (
                "True if the reference passages bear on this question. "
                "False if none of them do -- in which case cite nothing "
                "and say the material does not cover it."
            ),
        },
    },
    "required": ["reasoning", "is_correct", "feedback", "citations",
                 "material_relevant"],
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
    run_tests: bool = True,
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

  # Run the code before asking the model about it. Whether a fix works is a
  # question with an answer, and reading the code to guess at it is the one
  # thing a language model is least dependable at.
  if run_tests:
    try:
      checked = verify.verify(current_code, concept)
    except Exception as exc:  # noqa: BLE001
      log.warning("Verification failed for %r: %s", concept, exc)
      checked = verify.VerificationResult(
          concept=concept, ran=False, error=str(exc))
  else:
    checked = verify.VerificationResult(concept=concept, ran=False,
                                        error="test execution disabled")

  prompt = _EVAL_PROMPT.format(
      concept=concept,
      expected_fix=expected_fix,
      passages=_format_passages(passages),
      code=current_code,
      verification=checked.as_prompt_block(),
      history=_format_history(chat_history),
      scaffolding=_scaffolding_for(attempt_count),
  )

  supplied = {p.chunk_id for p in passages}
  base_prompt = prompt
  revisions = 0
  revision_reason = ""

  # One bounded revision when the citation check fails. Detecting a
  # fabricated source and then using the answer anyway makes the check
  # decorative; a model that fabricates twice will fabricate ten times, so
  # the retry is capped at one.
  while True:
    raw = vertex.generate(
        prompt,
        system_instruction=_EVAL_SYSTEM,
        temperature=0.0,
        response_schema=_EVAL_SCHEMA,
        usage=usage,
    )
    data = _parse_evaluation(raw)
    cited = _normalise_citations(
        data.get("citations"), data.get("feedback", ""))

    material_relevant = bool(data.get("material_relevant", True))
    problem = check_grounding(
        cited, supplied, bool(passages), material_relevant)
    if not problem or revisions >= MAX_REVISIONS:
      if problem:
        log.warning("Evaluation still ungrounded after %d revision(s): %s",
                    revisions, problem)
      break

    revisions += 1
    revision_reason = problem
    log.info("Revising evaluation (attempt %d): %s", revisions + 1, problem)
    prompt = base_prompt + _REVISION_NOTE.format(problem=problem)

  model_verdict = bool(data.get("is_correct", False))
  # Executable cases decide correctness where they exist; the model's view
  # is kept for comparison rather than discarded. Where no cases exist for
  # a concept, the model is all there is, and `judged_by` says so.
  if checked.ran and checked.cases:
    is_correct = checked.all_passed
    judged_by = "tests"
  else:
    is_correct = model_verdict
    judged_by = "model"

  return Evaluation(
      is_correct=is_correct,
      model_is_correct=model_verdict,
      judged_by=judged_by,
      verification=checked.to_dict(),
      reasoning=str(data.get("reasoning", "")),
      feedback=str(data.get("feedback", "")),
      citations=cited,
      valid_citations=[c for c in cited if c in supplied],
      invalid_citations=[c for c in cited if c not in supplied],
      passages=list(passages),
      material_relevant=material_relevant,
      revisions=revisions,
      revision_reason=revision_reason,
      usage=usage.summary() if usage else {},
  )


def check_grounding(
    citations: list[str],
    supplied: set[str],
    had_passages: bool,
    material_relevant: bool = True,
) -> str:
  """Why this answer should be revised, or "" if it should not.

  Citing an id that was never supplied is a fabricated source and always
  earns a retry.

  Citing nothing is the interesting case, because it has two opposite
  meanings. If the passages do bear on the question and the model ignored
  them, the answer came from its own knowledge while still presenting a
  citations field -- exactly what that field exists to catch. If the
  passages do not bear on the question, citing nothing is the correct
  answer and retrying is wrong.

  Text cannot distinguish those reliably, so the model declares which it
  means in `material_relevant`. Before that field existed this check
  revised every refusal: measured over the twelve unanswerable questions
  in eval/, ten triggered a pointless second call and logged a warning
  that read like a failure when the system had just behaved correctly.
  """
  invalid = [c for c in citations if c not in supplied]
  if invalid:
    return (
        f"You cited {invalid}, which were not among the passages you were "
        f"given. Only these ids exist: {sorted(supplied)}."
    )
  if had_passages and not citations and material_relevant:
    return (
        "You cited nothing, although you judged the passages relevant. "
        "Either cite the ones you used, or set material_relevant to false "
        "and say the material does not cover this."
    )
  return ""


_REVISION_NOTE = """

### Your previous answer was rejected
{problem}

The passages above are the only ones available. Answer again using them.
If none support an answer, say so and return an empty citations list --
that is a correct response, not a failure."""


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

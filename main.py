"""FastAPI backend.

Rewired from ChromaDB + Groq + SQLite onto Atlas + Vertex. Four changes
worth knowing about beyond the provider swap:

  * `/api/evaluate` now passes the concept through. The previous version
    called the evaluator with four positional arguments and never supplied
    `concept`, so every evaluation ran under the default
    "Computer Science Logic" -- which also meant no concept-scoped
    retrieval was possible even once a corpus existed.
  * responses carry the sources the critic was given and the ids it cited,
    so the retrieval is inspectable from the client rather than only in
    the server log.
  * interactions are written to Atlas with a student id. The old schema
    had none, which is why the migrated history is all under "legacy".
  * `/api/run` goes through `socratic.execute`, which is better hardened
    than the previous bare `subprocess.run` and still explicitly not a
    security boundary. See that module's docstring before exposing this
    service to anyone.

Run with:  uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from socratic import config, db, execute, history, retrieval, tutor, vertex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("socratic.api")

app = FastAPI(
    title="AI Socratic Evaluator",
    description=(
        "Socratic debugging tutor over two retrieval pipelines: Atlas Vector "
        "Search for CS explanations, structured Atlas queries for the "
        "misconception taxonomy and student history."
    ),
    version="2.0.0",
)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class ConceptRequest(BaseModel):
  concept: str


class RunRequest(BaseModel):
  flawed_code: str
  test_script: str = ""


class EvalRequest(BaseModel):
  concept: str
  flawed_code: str = ""
  original_code: Optional[str] = ""
  current_code: Optional[str] = ""
  chat_history: list[dict] = Field(default_factory=list)
  attempt_count: int = 1
  expected_fix: Optional[str] = ""
  # Supplied by the client so history accumulates per student instead of
  # all landing under one id as the legacy rows did.
  student_id: Optional[str] = None
  session_id: Optional[str] = None


class SearchRequest(BaseModel):
  query: str
  concept: Optional[str] = None
  k: int = config.DEFAULT_K


# ---------------------------------------------------------------------------
# Health and introspection
# ---------------------------------------------------------------------------
@app.get("/api/health")
def api_health() -> dict[str, Any]:
  """Is everything this service depends on actually reachable?

  Checked rather than assumed: an empty `cs_chunks` or a vector index that
  is not queryable both present as retrieval silently returning nothing.
  """
  status: dict[str, Any] = {"ok": True}

  try:
    status["mongo"] = db.ping()
  except Exception as exc:  # noqa: BLE001
    status["ok"] = False
    status["mongo"] = {"ok": False, "error": str(exc)}

  try:
    status["corpus"] = retrieval.corpus_stats()
    if status["corpus"]["chunks"] == 0:
      status["ok"] = False
      status["corpus"]["warning"] = (
          "cs_chunks is empty; run scripts.build_corpus then "
          "scripts.ingest_corpus"
      )
  except Exception as exc:  # noqa: BLE001
    status["ok"] = False
    status["corpus"] = {"error": str(exc)}

  try:
    indexes = [
        {"name": i.get("name"), "queryable": i.get("queryable"),
         "status": i.get("status")}
        for i in db.chunks().list_search_indexes()
    ]
    status["search_indexes"] = indexes
    if not any(i["queryable"] for i in indexes):
      status["ok"] = False
  except Exception as exc:  # noqa: BLE001
    status["search_indexes"] = {"error": str(exc)}

  status["models"] = {
      "generation": config.GEN_MODEL,
      "embedding": f"{config.EMBED_MODEL}@{config.EMBED_DIM}",
      "location": config.GCP_LOCATION,
  }
  return status


@app.get("/api/concepts")
def api_concepts() -> dict[str, Any]:
  """Concepts the taxonomy can actually generate an exercise for.

  The client's topic selector offers 22; the taxonomy has 11. Returning
  both lists lets the UI mark the difference instead of offering topics
  that fail on generation.
  """
  available = sorted(db.misconceptions().distinct("concept"))
  return {
      "available": available,
      "count": len(available),
      "corpus_concepts": sorted(
          c for c in db.chunks().distinct("concept_tags") if c
      ),
  }


# ---------------------------------------------------------------------------
# Exercise generation -- structured path
# ---------------------------------------------------------------------------
@app.post("/api/generate")
def api_generate(req: ConceptRequest) -> dict[str, Any]:
  usage = vertex.Usage()
  try:
    exercise = tutor.generate_flaw(req.concept, usage=usage)
  except tutor.ConceptNotInTaxonomy as exc:
    # 422, not 500: the request named something the taxonomy does not
    # cover, which is a client-correctable problem and worth saying so.
    raise HTTPException(
        status_code=422,
        detail={
            "error": "concept_not_in_taxonomy",
            "concept": exc.concept,
            "available": sorted(exc.available),
            "message": str(exc),
        },
    ) from exc
  except Exception as exc:  # noqa: BLE001
    log.exception("generate failed for %r", req.concept)
    raise HTTPException(status_code=500, detail=str(exc)) from exc

  payload = exercise.to_dict()
  payload["usage"] = usage.summary()
  return payload


# ---------------------------------------------------------------------------
# Code execution
# ---------------------------------------------------------------------------
@app.post("/api/run")
def api_run(req: RunRequest) -> dict[str, Any]:
  result = execute.run_student_code(req.flawed_code, req.test_script)
  return result.to_dict()


# ---------------------------------------------------------------------------
# Evaluation -- both paths
# ---------------------------------------------------------------------------
@app.post("/api/evaluate")
def api_evaluate(req: EvalRequest) -> dict[str, Any]:
  active_code = req.current_code or req.flawed_code
  if not active_code.strip():
    raise HTTPException(
        status_code=422, detail="No code supplied to evaluate.")

  usage = vertex.Usage()
  try:
    evaluation = tutor.evaluate_student(
        active_code,
        req.chat_history,
        req.attempt_count,
        concept=req.concept,
        expected_fix=req.expected_fix or "",
        usage=usage,
    )
  except Exception as exc:  # noqa: BLE001
    log.exception("evaluate failed for %r", req.concept)
    raise HTTPException(status_code=500, detail=str(exc)) from exc

  verification = evaluation.verification or {}
  history.log_interaction(
      concept=req.concept,
      is_correct=evaluation.is_correct,
      attempts=req.attempt_count,
      student_id=req.student_id or "anonymous",
      session_id=req.session_id,
      citations=evaluation.valid_citations,
      invalid_citations=evaluation.invalid_citations,
      grounded=evaluation.grounded,
      passage_count=len(evaluation.passages),
      usage=usage.summary(),
      judged_by=evaluation.judged_by,
      model_is_correct=evaluation.model_is_correct,
      model_agreed=evaluation.model_agreed,
      tests_passed=verification.get("passed"),
      tests_total=verification.get("total"),
  )

  if evaluation.model_agreed is False:
    # The model read the code one way and running it showed the other.
    # Worth a log line: it is the raw material for the agreement rate, and
    # a run of these means the critic is drifting from what the code does.
    log.info(
        "Model verdict (%s) disagreed with tests (%s) on %r: %s/%s cases",
        evaluation.model_is_correct, evaluation.is_correct, req.concept,
        verification.get("passed"), verification.get("total"))

  if evaluation.invalid_citations:
    # The critic named a passage that was never in its prompt. Worth a log
    # line: it is the signal that grounding is slipping, and it is
    # invisible from the client unless something records it.
    log.warning(
        "Evaluation cited %d id(s) that were not supplied: %s",
        len(evaluation.invalid_citations), evaluation.invalid_citations)

  return evaluation.to_dict()


# ---------------------------------------------------------------------------
# Retrieval, exposed directly
# ---------------------------------------------------------------------------
@app.post("/api/search")
def api_search(req: SearchRequest) -> dict[str, Any]:
  """Path A on its own, so the retrieval can be inspected without a tutor turn."""
  try:
    passages = retrieval.semantic_search(
        req.query, k=req.k, concept=req.concept)
  except Exception as exc:  # noqa: BLE001
    log.exception("search failed")
    raise HTTPException(status_code=500, detail=str(exc)) from exc
  return {
      "query": req.query,
      "concept": req.concept,
      "count": len(passages),
      "passages": [p.to_dict() for p in passages],
  }


@app.get("/api/history/{student_id}")
def api_history(student_id: str, concept: Optional[str] = None) -> dict[str, Any]:
  """Path B on its own: aggregation over records, no embeddings involved."""
  try:
    summary = retrieval.student_concept_history(student_id, concept)
    summary["weakest"] = retrieval.weakest_concepts(student_id)
    return summary
  except Exception as exc:  # noqa: BLE001
    log.exception("history failed for %r", student_id)
    raise HTTPException(status_code=500, detail=str(exc)) from exc

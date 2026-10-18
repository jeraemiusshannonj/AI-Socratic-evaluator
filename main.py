from contextlib import contextmanager
import datetime
import sqlite3
import subprocess
import sys
from typing import Optional

from chains import evaluate_student_multi_turn, generate_flaw
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

DB_PATH = "evaluator.db"


@contextmanager
def get_db_cursor():
  conn = sqlite3.connect(DB_PATH)
  cursor = conn.cursor()
  try:
    yield cursor
    conn.commit()
  finally:
    conn.close()


def init_db():
  with get_db_cursor() as cursor:
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                concept TEXT,
                is_correct BOOLEAN,
                attempts INTEGER,
                timestamp TEXT
            )
        """)


init_db()


class ConceptRequest(BaseModel):
  concept: str


class RunRequest(BaseModel):
  flawed_code: str
  test_script: str


class EvalRequest(BaseModel):
  concept: str
  flawed_code: str
  original_code: Optional[str] = ""
  current_code: Optional[str] = ""
  chat_history: list[dict]
  attempt_count: int
  expected_fix: Optional[str] = ""


@app.post("/api/generate")
def api_generate(req: ConceptRequest):
  try:
    data = generate_flaw(req.concept)
    return data
  except Exception as e:
    print(f"❌ Error in /api/generate: {e}")
    raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/run")
def run_code(request: RunRequest):
  combined_code = f"{request.flawed_code}\n\n{request.test_script}"

  try:
    result = subprocess.run(
        [sys.executable, "-c", combined_code],
        capture_output=True,
        text=True,
        timeout=3,
    )

    output = result.stdout
    if result.stderr:
      output += f"\n⚠️ Runtime Error:\n{result.stderr}"

    if not output.strip():
      output = "Execution finished with no stdout output."

    return {"output": output}

  except subprocess.TimeoutExpired:
    return {
        "output": (
            "⏱️ Execution Timed Out! (An infinite loop was detected in your"
            " code)"
        )
    }
  except Exception as e:
    return {"output": f"⚠️ Execution Error: {e}"}


@app.post("/api/evaluate")
def api_evaluate(req: EvalRequest):
  try:
    active_code = req.current_code or req.flawed_code
    print(
        f"\n--- DEBUG: Evaluator received code length:"
        f" {len(active_code)} ---\n{active_code}\n--------------------\n"
    )
    evaluation = evaluate_student_multi_turn(
        active_code,
        req.chat_history,
        req.attempt_count,
        req.expected_fix or "",
    )

    with get_db_cursor() as cursor:
      cursor.execute(
          "INSERT INTO interactions (concept, is_correct, attempts, timestamp)"
          " VALUES (?, ?, ?, ?)",
          (
              req.concept,
              bool(evaluation.get("is_correct", False)),
              req.attempt_count,
              str(datetime.datetime.now()),
          ),
      )

    return evaluation
  except Exception as e:
    print(f"❌ Error in /api/evaluate: {e}")
    raise HTTPException(status_code=500, detail=str(e))
from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3
import datetime
from chains import generate_flaw, evaluate_student

app = FastAPI()

# Setup SQLite Telemetry Database
conn = sqlite3.connect('telemetry.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
               CREATE TABLE IF NOT EXISTS interactions
               (
                   id
                   INTEGER
                   PRIMARY
                   KEY
                   AUTOINCREMENT,
                   concept
                   TEXT,
                   is_correct
                   BOOLEAN,
                   timestamp
                   TEXT
               )
               ''')
conn.commit()


class ConceptRequest(BaseModel):
    concept: str


class EvalRequest(BaseModel):
    concept: str
    flawed_code: str
    critique: str


@app.post("/api/generate")
def api_generate(req: ConceptRequest):
    code = generate_flaw(req.concept)
    return {"flawed_code": code}


@app.post("/api/evaluate")
def api_evaluate(req: EvalRequest):
    evaluation = evaluate_student(req.flawed_code, req.critique)

    # Log telemetry for research analysis
    cursor.execute(
        "INSERT INTO interactions (concept, is_correct, timestamp) VALUES (?, ?, ?)",
        (req.concept, evaluation['is_correct'], str(datetime.datetime.now()))
    )
    conn.commit()

    return evaluation
"""The code execution service, deployed on its own.

This exists to be the blast radius. The main API holds a Mongo connection
string and a Vertex-capable service account; this service holds neither,
reaches no network, writes to no persistent disk, and can be reached only
from inside the project. Running untrusted student code here rather than
in the API process is the difference between a bug and an incident.

It is deliberately the smallest thing that can work:

  * stdlib plus FastAPI -- no pymongo, no google-genai, no dotenv, so a
    compromise finds no credentials to steal because none are mounted
  * one endpoint, which takes code and returns what it printed
  * no logging of the code itself, since it is untrusted input and the
    logs are readable from the console

The guarantees that matter are not in this file. They are on the Cloud Run
service: `--no-allow-unauthenticated` and internal ingress so only the API
can call it, `--vpc-egress` unset with no connector so it has no route out,
a service account with zero roles, a memory cap, and a non-root user in the
image. See deploy/deploy.sh.
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel, Field

from socratic import execute

app = FastAPI(
    title="socratic-sandbox",
    description="Executes untrusted Python. Not reachable from the internet.",
    version="1.0.0",
)

MAX_CODE_CHARS = 100_000
MAX_TIMEOUT = 15.0


class RunRequest(BaseModel):
  code: str = Field(max_length=MAX_CODE_CHARS)
  test_script: str = Field(default="", max_length=MAX_CODE_CHARS)
  timeout: float = execute.TIMEOUT_SECONDS


@app.get("/healthz")
def healthz() -> dict:
  return {"ok": True, "service": "sandbox"}


@app.post("/run")
def run(req: RunRequest) -> dict:
  # Clamp rather than trust: the caller is our own API, but a bug there
  # should not be able to pin a container for an arbitrary duration.
  timeout = max(0.5, min(float(req.timeout), MAX_TIMEOUT))
  result = execute.run_student_code(req.code, req.test_script, timeout=timeout)
  return result.to_dict()


if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

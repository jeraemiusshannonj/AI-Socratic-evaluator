"""The code execution service, deployed on its own.

This exists to be the blast radius. The main API holds a Mongo connection
string and a Vertex-capable service account; this service holds neither,
and can be invoked only by the API's service account. Running untrusted
student code here rather than in the API process is the difference
between a bug and an incident.

It is deliberately the smallest thing that can work:

  * stdlib plus FastAPI -- no pymongo, no google-genai, no dotenv, so a
    compromise finds no credentials to steal because none are mounted
  * one endpoint, which takes code and returns what it printed
  * no logging of the code itself, since it is untrusted input and the
    logs are readable from the console

The guarantees that matter are not in this file. They are on the Cloud Run
service: `--no-allow-unauthenticated` with `roles/run.invoker` granted only
to the API's service account, so Google's front end rejects every other
caller; a service account with zero roles; `--concurrency=1`; a memory cap;
and a non-root user in the image. See deploy/deploy.sh.

What it does not have, stated so nobody assumes otherwise:

  * internal ingress. `--ingress=internal` means "from VPC networks", and
    Cloud Run to Cloud Run calls do not cross a VPC without a paid
    connector, so it would break the API's calls. IAM is the boundary.
  * an egress block. Cloud Run allows outbound traffic by default and
    removing it needs a VPC connector. Student code can reach the
    internet; it finds nothing here worth sending.
  * a read-only filesystem. The container filesystem is in-memory and
    writable; each run gets a fresh temporary directory that is removed
    afterwards, and nothing survives an instance being recycled.
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

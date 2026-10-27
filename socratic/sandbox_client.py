"""Call the sandbox service instead of running untrusted code in-process.

Kept out of `socratic/execute.py` on purpose. That module is stdlib-only,
which is what lets the sandbox image ship without pymongo, google-genai or
dotenv -- a compromise there finds no credentials because none are
installed. Putting an HTTP client and a Google auth dependency into it
would undo that.

Two modes:

  SANDBOX_URL set    -> POST to the sandbox service, authenticated with a
                        Cloud Run identity token. This is production.
  SANDBOX_URL unset  -> run locally through socratic.execute. This is
                        development, and is the behaviour the app had
                        before the split.

The local path is not a security boundary and the module says so loudly
rather than quietly degrading: `require_remote()` lets the API refuse to
start in a deployed environment that forgot to set the URL.
"""

from __future__ import annotations

import logging

from . import config, execute

log = logging.getLogger(__name__)

# Request timeout for the HTTP call itself. Deliberately longer than the
# execution timeout: the sandbox needs room to kill a hung process and
# return its partial output, and a client timeout shorter than that turns
# a clean "your code did not terminate" into an opaque network error.
HTTP_TIMEOUT_MARGIN = 20.0


def _identity_token(audience: str) -> str | None:
  """Fetch a Cloud Run identity token for the sandbox service.

  Returns None when running somewhere without a service account -- a local
  machine on user ADC, typically. The caller then has to decide whether an
  unauthenticated attempt is acceptable, and in a deployed environment it
  is not.
  """
  try:
    import google.auth.transport.requests
    import google.oauth2.id_token

    request = google.auth.transport.requests.Request()
    return google.oauth2.id_token.fetch_id_token(request, audience)
  except Exception as exc:  # noqa: BLE001
    log.warning("No identity token available for %s: %s", audience, exc)
    return None


def run(
    code: str, test_script: str = "", *, timeout: float | None = None
) -> execute.ExecutionResult:
  """Execute code, remotely when configured and locally otherwise."""
  seconds = timeout if timeout is not None else execute.TIMEOUT_SECONDS

  if not config.SANDBOX_URL:
    return execute.run_student_code(code, test_script, timeout=seconds)

  import requests

  url = config.SANDBOX_URL.rstrip("/") + "/run"
  headers = {}
  token = _identity_token(config.SANDBOX_URL.rstrip("/"))
  if token:
    headers["Authorization"] = f"Bearer {token}"

  try:
    response = requests.post(
        url,
        json={"code": code, "test_script": test_script, "timeout": seconds},
        headers=headers,
        timeout=seconds + HTTP_TIMEOUT_MARGIN,
    )
  except Exception as exc:  # noqa: BLE001
    log.exception("Sandbox request failed")
    return execute.ExecutionResult(
        output=f"The code runner could not be reached: {exc}")

  if response.status_code == 403:
    # Almost always a missing run.invoker binding rather than anything
    # the student did, so say which.
    log.error("Sandbox refused the call (403). Does the API service "
              "account have roles/run.invoker on the sandbox service?")
    return execute.ExecutionResult(
        output="The code runner rejected this request (not authorised).")

  if response.status_code != 200:
    log.error("Sandbox returned %s: %s", response.status_code,
              response.text[:300])
    return execute.ExecutionResult(
        output=f"The code runner returned an error ({response.status_code}).")

  payload = response.json()
  return execute.ExecutionResult(
      output=payload.get("output", ""),
      timed_out=bool(payload.get("timed_out")),
      exit_code=payload.get("exit_code"),
      truncated=bool(payload.get("truncated")),
  )


def require_remote() -> None:
  """Refuse to run untrusted code in-process in a deployed environment.

  Falling back to local execution is right on a laptop and wrong on a
  public URL, and the difference is invisible at the call site. The API
  calls this at startup when it detects it is deployed, so the failure is
  a refusal to boot rather than a silently unsafe service.
  """
  if not config.SANDBOX_URL:
    raise RuntimeError(
        "SANDBOX_URL is not set, so untrusted code would run inside this "
        "service alongside the Mongo credentials and the Vertex service "
        "account. Deploy the sandbox service and set SANDBOX_URL, or set "
        "ALLOW_LOCAL_EXECUTION=1 if this really is a local development run."
    )

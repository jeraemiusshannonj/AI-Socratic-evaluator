"""Run the student's code and capture what it printed.

**This is not a security sandbox, and must not be treated as one.** The
code being run is arbitrary Python supplied over HTTP, executed with the
same user and filesystem access as the server process. The hardening below
makes accidents survivable -- an infinite loop, a runaway print, an import
of something in the working directory -- and does nothing against someone
who is actually trying. It would stop neither `open('/etc/passwd').read()`
nor an outbound network call.

Running this service anywhere reachable by someone you do not trust needs
a real boundary: a container with no network and a read-only root, gVisor,
a Firecracker microVM, or a hosted execution service. The previous
implementation ran `subprocess.run([sys.executable, '-c', code])` with a
three-second timeout and nothing else, so what follows is a meaningful
improvement on that and still not sufficient.

What is actually enforced here:

  * `-I` isolated mode, so `PYTHONPATH`, the user site-packages directory
    and the current directory are all off `sys.path`. Without it, a file
    named `json.py` next to the server gets imported by `import json`.
  * a fresh empty working directory per run, removed afterwards
  * a minimal environment, so credentials in the server's environment --
    `MONGO_URI` among them -- are not inherited by student code
  * a wall-clock timeout, with the process killed rather than left running
  * output truncation, so a print loop cannot exhaust server memory
  * on POSIX, soft limits on CPU time, address space and file size
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass

log = logging.getLogger(__name__)

TIMEOUT_SECONDS = 5.0
MAX_OUTPUT_CHARS = 20_000
# POSIX-only soft limits. Deliberately loose enough that ordinary teaching
# code never trips them.
CPU_SECONDS = 5
ADDRESS_SPACE_BYTES = 512 * 1024 * 1024
FILE_SIZE_BYTES = 1024 * 1024


@dataclass
class ExecutionResult:
  output: str
  timed_out: bool = False
  exit_code: int | None = None
  truncated: bool = False

  def to_dict(self) -> dict:
    return {
        "output": self.output,
        "timed_out": self.timed_out,
        "exit_code": self.exit_code,
        "truncated": self.truncated,
    }


def _limits() -> object | None:
  """Build a preexec_fn applying resource limits, on platforms that have them."""
  try:
    import resource
  except ImportError:  # Windows
    return None

  def apply() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
    resource.setrlimit(
        resource.RLIMIT_AS, (ADDRESS_SPACE_BYTES, ADDRESS_SPACE_BYTES))
    resource.setrlimit(
        resource.RLIMIT_FSIZE, (FILE_SIZE_BYTES, FILE_SIZE_BYTES))
    # No core dumps from a crash in student code.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

  return apply


def _child_env() -> dict[str, str]:
  """A minimal environment.

  The server's own environment holds the Mongo connection string and the
  GCP project; student code has no business reading either.
  """
  keep = ("PATH", "SYSTEMROOT", "WINDIR", "TMP", "TEMP", "TMPDIR",
          "COMSPEC", "PATHEXT")
  env = {name: os.environ[name] for name in keep if name in os.environ}
  env["PYTHONDONTWRITEBYTECODE"] = "1"
  env["PYTHONUNBUFFERED"] = "1"
  return env


def _truncate(text: str) -> tuple[str, bool]:
  if len(text) <= MAX_OUTPUT_CHARS:
    return text, False
  return (
      text[:MAX_OUTPUT_CHARS]
      + f"\n\n[output truncated at {MAX_OUTPUT_CHARS:,} characters]",
      True,
  )


def run_student_code(
    code: str, test_script: str = "", *, timeout: float = TIMEOUT_SECONDS
) -> ExecutionResult:
  """Execute `code` followed by `test_script`, returning combined output."""
  combined = code if not test_script else f"{code}\n\n{test_script}"

  workdir = tempfile.mkdtemp(prefix="socratic-run-")
  script_path = os.path.join(workdir, "student_main.py")
  try:
    with open(script_path, "w", encoding="utf-8") as handle:
      handle.write(combined)

    kwargs: dict = {}
    preexec = _limits()
    if preexec is not None:
      kwargs["preexec_fn"] = preexec

    try:
      completed = subprocess.run(
          # -I: isolated. -S: skip site customisation.
          [sys.executable, "-I", "-S", script_path],
          capture_output=True,
          text=True,
          timeout=timeout,
          cwd=workdir,
          env=_child_env(),
          stdin=subprocess.DEVNULL,
          **kwargs,
      )
    except subprocess.TimeoutExpired as expired:
      partial = (expired.stdout or "") + (expired.stderr or "")
      if isinstance(partial, bytes):
        partial = partial.decode("utf-8", "replace")
      body, truncated = _truncate(partial.strip())
      message = (
          f"Execution timed out after {timeout:g}s -- this usually means an "
          f"infinite loop."
      )
      return ExecutionResult(
          output=f"{body}\n\n{message}".strip(),
          timed_out=True,
          truncated=truncated,
      )

    output = completed.stdout
    if completed.stderr.strip():
      output += f"\n--- stderr ---\n{completed.stderr}"
    if not output.strip():
      output = "Ran successfully with no output."

    body, truncated = _truncate(output.strip())
    return ExecutionResult(
        output=body,
        exit_code=completed.returncode,
        truncated=truncated,
    )

  except Exception as exc:  # noqa: BLE001
    log.exception("Failed to run student code")
    return ExecutionResult(output=f"Could not run the code: {exc}")
  finally:
    shutil.rmtree(workdir, ignore_errors=True)

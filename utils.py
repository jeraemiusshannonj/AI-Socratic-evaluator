import io
import multiprocessing
import re
import sys


def clean_and_extract_code(code_str: str) -> str:
  """Extracts raw Python from markdown code fences, strips comments,

  and preserves line breaks/indentation.
  """
  # 1. Extract code inside ```python ... ``` or ``` ... ``` blocks if present
  code_match = re.search(
      r"```(?:python)?\s*(.*?)\s*```", code_str, re.DOTALL | re.IGNORECASE
  )
  if code_match:
    code_str = code_match.group(1)

  # 2. Strip comments line by line
  cleaned_lines = []
  for line in code_str.splitlines():
    # Strip SQL comments
    line = re.sub(r"--.*$", "", line)
    # Strip Python comments while preserving '#' inside strings
    line = re.sub(r"(?<![\'\"\w])#.*$", "", line)

    if line.strip():
      cleaned_lines.append(line.rstrip())

  return "\n".join(cleaned_lines)


def _worker_exec(code: str, test_script: str, queue: multiprocessing.Queue):
  """Subprocess target executing user code and custom test parameters."""
  buffer = io.StringIO()
  sys.stdout = buffer
  sys.stderr = buffer
  try:
    exec_globals = {}
    exec(code, exec_globals)
    exec(test_script, exec_globals)
    queue.put(("SUCCESS", buffer.getvalue()))
  except Exception as e:
    queue.put(("ERROR", f"{type(e).__name__}: {str(e)}"))


def run_code_in_sandbox(
    code: str, test_script: str, timeout_sec: float = 2.0
) -> str:
  """Executes code in an isolated process with a strict timeout to handle

  infinite loops safely.
  """
  queue = multiprocessing.Queue()
  process = multiprocessing.Process(
      target=_worker_exec, args=(code, test_script, queue)
  )
  process.start()
  process.join(timeout_sec)

  if process.is_alive():
    process.terminate()
    process.join()
    return "⚠️ [Execution Limit Reached]: Possible infinite loop detected! Execution timed out after 2 seconds."

  if not queue.empty():
    status, output = queue.get()
    if status == "SUCCESS":
      return (
          output.strip()
          if output.strip()
          else "Code executed cleanly with no print output."
      )
    return f"❌ Runtime Error: {output}"

  return "No output returned."
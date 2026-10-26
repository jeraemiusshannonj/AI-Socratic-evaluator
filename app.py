"""Streamlit front end.

Changed alongside the backend rewiring, in three ways that matter:

  * The topic selector used to offer 22 concepts while the taxonomy held
    11, and picking one of the missing 11 silently produced a
    loop-termination bug regardless of topic. The list is now fetched from
    the API and the unavailable ones are marked and disabled, so the gap
    is visible instead of being papered over.
  * The sources the critic was given, and the ones it cited, are shown.
    Retrieval that cannot be seen cannot be trusted, and a tutor that
    cites a passage the student can open is a different product from one
    that asserts.
  * A per-browser student id is sent, so history accumulates per student.
    The old schema had no identifier at all.
"""

import uuid

import requests
import streamlit as st

st.set_page_config(
    page_title="AI Socratic Evaluator",
    page_icon="brain",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = "http://localhost:8000/api"

# Generation and evaluation both make a model call; evaluation also makes an
# embedding call and a vector search first. The old 15/20s timeouts were
# tight enough to fail on a cold start.
GENERATE_TIMEOUT = 60
EVALUATE_TIMEOUT = 90
RUN_TIMEOUT = 20

ALL_CONCEPTS = [
    "Linear Search",
    "Binary Search (Off-by-One)",
    "Binary Search (Loop Condition)",
    "Binary Search (Prerequisite)",
    "Bubble Sort (Early Exit)",
    "Selection Sort (Min Index Swap)",
    "Insertion Sort (Shifting Bug)",
    "Merge Sort (Slice Overhead)",
    "Merge Sort (Two-Pointer Comparison)",
    "Quick Sort (Partition Off-By-One)",
    "Quick Sort (Lomuto Partition)",
    "Heap Sort (0-Indexed Parent/Child)",
    "Graph Traversal (BFS Visited Tracking)",
    "Graph Traversal (DFS Cycle Detection)",
    "Recursion",
    "Python Lists",
    "Python Default Arguments",
    "Python Object Mutability",
    "Hash Tables",
    "SQL Left Join",
    "SQL Group By",
    "Object-Oriented Programming",
]


def safe_rerun():
  if hasattr(st, "rerun"):
    st.rerun()
  elif hasattr(st, "experimental_rerun"):
    st.experimental_rerun()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "messages": [],
    "code_editor": "",
    "original_flawed_code": "",
    "selected_concept": "Linear Search",
    "expected_fix": "",
    "attempt_count": 1,
    "sandbox_output": "",
    "last_sources": [],
    "last_citations": [],
    "last_grounded": None,
}
for key, default in _DEFAULTS.items():
  if key not in st.session_state:
    st.session_state[key] = default

# One id per browser session. Not authentication -- just enough for
# per-student history to mean something, which the old schema could not do.
if "student_id" not in st.session_state:
  st.session_state.student_id = f"web-{uuid.uuid4().hex[:12]}"
if "session_id" not in st.session_state:
  st.session_state.session_id = uuid.uuid4().hex[:16]


@st.cache_data(ttl=60)
def fetch_available_concepts() -> tuple[list[str], str]:
  """Which concepts the taxonomy can actually generate an exercise for."""
  try:
    response = requests.get(f"{API_BASE_URL}/concepts", timeout=10)
    if response.status_code == 200:
      return response.json().get("available", []), ""
    return [], f"API returned {response.status_code}"
  except Exception as exc:  # noqa: BLE001
    return [], str(exc)


def generate_new_problem(concept_name: str) -> None:
  try:
    with st.spinner(f"Generating a flawed implementation for {concept_name}..."):
      response = requests.post(
          f"{API_BASE_URL}/generate",
          json={"concept": concept_name},
          timeout=GENERATE_TIMEOUT,
      )
  except Exception as exc:  # noqa: BLE001
    st.error(
        f"Could not reach the API at {API_BASE_URL}. Start it with "
        f"`uvicorn main:app --port 8000`. Details: {exc}"
    )
    return

  if response.status_code == 422:
    # The concept has no taxonomy entry. Say so plainly rather than
    # generating something unrelated.
    detail = response.json().get("detail", {})
    st.error(
        f"**{concept_name}** has no entry in the misconception taxonomy, so "
        f"there is no specific bug to inject. Add a row to "
        f"`misconceptions.csv` and re-run `python -m scripts.migrate_sqlite`."
    )
    if isinstance(detail, dict) and detail.get("available"):
      st.caption(f"{len(detail['available'])} concepts are available.")
    return

  if response.status_code != 200:
    st.error(f"Generation failed with status {response.status_code}.")
    return

  data = response.json()
  flawed_code = data.get("flawed_code", "")
  st.session_state.code_editor = flawed_code
  st.session_state.original_flawed_code = flawed_code
  st.session_state.expected_fix = data.get("expected_fix", "")
  st.session_state.selected_concept = concept_name
  st.session_state.attempt_count = 1
  st.session_state.sandbox_output = ""
  st.session_state.last_sources = []
  st.session_state.last_citations = []
  st.session_state.last_grounded = None
  st.session_state.session_id = uuid.uuid4().hex[:16]

  note = ""
  if data.get("taxonomy_match") == "alias":
    note = (
        " (matched a related variant in the taxonomy, so the bug is for that "
        "variant)"
    )

  st.session_state.messages = [{
      "role": "assistant",
      "content": (
          f"I've written an implementation of **{concept_name}** with one "
          f"deliberate logic bug{note}. Read it, run it in the sandbox, or "
          f"tell me what you think is wrong."
      ),
  }]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
available, concepts_error = fetch_available_concepts()

with st.sidebar:
  st.title("Control panel")

  if concepts_error:
    st.warning(
        f"Could not read the taxonomy ({concepts_error}). Showing all topics; "
        f"some may fail to generate."
    )
    options = ALL_CONCEPTS
  else:
    # Unavailable topics stay visible but marked. Hiding them would hide the
    # gap; offering them unmarked is what the old version did.
    options = sorted(available) + [
        c for c in ALL_CONCEPTS if c not in available
    ]

  def _label(concept: str) -> str:
    if concepts_error or concept in available:
      return concept
    return f"{concept}  --  no taxonomy entry"

  current = st.session_state.selected_concept
  chosen_concept = st.selectbox(
      "Topic",
      options=options,
      format_func=_label,
      index=options.index(current) if current in options else 0,
  )

  unavailable = bool(available) and chosen_concept not in available
  if unavailable:
    st.caption(
        "This topic has no misconception defined, so no exercise can be "
        "generated for it yet."
    )

  if st.button("Generate problem", disabled=unavailable):
    generate_new_problem(chosen_concept)

  if st.button("Reset conversation"):
    st.session_state.messages = []
    st.session_state.attempt_count = 1
    st.session_state.sandbox_output = ""
    st.session_state.last_sources = []
    safe_rerun()

  st.divider()
  st.caption(f"Active concept: {st.session_state.selected_concept}")
  st.caption(f"Attempt: {st.session_state.attempt_count}")
  if available:
    st.caption(f"Taxonomy coverage: {len(available)} of {len(ALL_CONCEPTS)}")
  st.caption(f"Student: `{st.session_state.student_id}`")


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
st.title("AI Socratic Evaluator")
st.markdown(
    "Find the logic bug, test your fix, and argue your reasoning with a tutor "
    "that cites its sources."
)

col_code, col_chat = st.columns([1, 1], gap="medium")

with col_code:
  st.subheader("Code under review")

  edited_code = st.text_area(
      label="Code editor",
      value=st.session_state.code_editor,
      key="code_editor_input",
      height=260,
  )
  st.session_state.code_editor = edited_code

  if st.button("Reset code to original"):
    st.session_state.code_editor = st.session_state.original_flawed_code
    safe_rerun()

  st.subheader("Sandbox")
  st.caption(
      "Runs in a separate process with a timeout. Not a security sandbox -- "
      "see socratic/execute.py."
  )

  test_script = st.text_area(
      label="Test script",
      value="# Call your function here, for example:\n# print(my_function([10, 20, 30], 20))\n",
      height=110,
      key="sandbox_script",
  )

  if st.button("Run"):
    try:
      response = requests.post(
          f"{API_BASE_URL}/run",
          json={
              "flawed_code": st.session_state.code_editor,
              "test_script": test_script,
          },
          timeout=RUN_TIMEOUT,
      )
      if response.status_code == 200:
        payload = response.json()
        st.session_state.sandbox_output = payload.get("output", "")
        if payload.get("timed_out"):
          st.session_state.sandbox_output += (
              "\n\n(The process was killed. An infinite loop is the usual "
              "cause, and for several of these bugs it is the symptom.)"
          )
      else:
        st.session_state.sandbox_output = (
            f"Execution failed with status {response.status_code}."
        )
    except Exception as exc:  # noqa: BLE001
      st.session_state.sandbox_output = f"Could not reach the API: {exc}"

  if st.session_state.sandbox_output:
    st.code(st.session_state.sandbox_output, language="text")

with col_chat:
  st.subheader("Socratic dialogue")

  for message in st.session_state.messages:
    speaker = "You" if message["role"] == "user" else "Tutor"
    with st.chat_message(
        "user" if message["role"] == "user" else "assistant"
    ):
      st.markdown(f"**{speaker}:** {message['content']}")

  with st.form(key="chat_form", clear_on_submit=True):
    user_prompt = st.text_input(
        "Explain the flaw, or answer the tutor", key="user_chat_input"
    )
    submit_chat = st.form_submit_button("Send")

  if submit_chat and user_prompt.strip():
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    payload = {
        "concept": st.session_state.selected_concept,
        "flawed_code": st.session_state.original_flawed_code,
        "current_code": st.session_state.code_editor,
        "chat_history": st.session_state.messages,
        "attempt_count": st.session_state.attempt_count,
        "expected_fix": st.session_state.expected_fix,
        "student_id": st.session_state.student_id,
        "session_id": st.session_state.session_id,
    }

    try:
      with st.spinner("Retrieving sources and checking your code..."):
        response = requests.post(
            f"{API_BASE_URL}/evaluate", json=payload, timeout=EVALUATE_TIMEOUT
        )

      if response.status_code == 200:
        data = response.json()
        st.session_state.messages.append(
            {"role": "assistant",
             "content": data.get("feedback", "No feedback returned.")}
        )
        st.session_state.attempt_count += 1
        st.session_state.last_sources = data.get("sources", [])
        st.session_state.last_citations = data.get("valid_citations", [])
        st.session_state.last_grounded = data.get("grounded")

        if data.get("is_correct"):
          st.balloons()
          st.success("That's the fix. Nicely done.")
      else:
        st.error(f"Evaluation failed with status {response.status_code}.")

    except Exception as exc:  # noqa: BLE001
      st.error(f"Could not reach /evaluate: {exc}")

    safe_rerun()

  # ------------------------------------------------------------------
  # What the tutor actually read
  # ------------------------------------------------------------------
  if st.session_state.last_sources:
    cited = set(st.session_state.last_citations or [])
    label = (
        f"Sources retrieved ({len(st.session_state.last_sources)}, "
        f"{len(cited)} cited)"
    )
    with st.expander(label, expanded=False):
      if st.session_state.last_grounded is False:
        st.caption(
            "The tutor did not cite any of these, so its last answer was "
            "not grounded in the corpus."
        )
      for source in st.session_state.last_sources:
        marker = "cited" if source["chunk_id"] in cited else "retrieved"
        st.markdown(
            f"**{source.get('title', 'unknown')}** &nbsp; "
            f"`{marker}` &nbsp; score {source.get('score', 0):.3f}"
        )
        if source.get("section"):
          st.caption(source["section"])
        if source.get("url"):
          st.caption(f"[{source['chunk_id']}]({source['url']})")
        st.divider()

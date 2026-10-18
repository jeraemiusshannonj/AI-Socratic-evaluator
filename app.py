import io
import sys
import requests
import streamlit as st

# =====================================================================
# 1. Page Configuration & Version Fallbacks
# =====================================================================
st.set_page_config(
    page_title="AI Socratic Evaluator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = "http://localhost:8000/api"


def safe_rerun():
  """Rerun helper compatible with older and newer Streamlit versions."""
  if hasattr(st, "rerun"):
    st.rerun()
  elif hasattr(st, "experimental_rerun"):
    st.experimental_rerun()


# =====================================================================
# 2. Session State Initialization
# =====================================================================
if "messages" not in st.session_state:
  st.session_state.messages = []

if "code_editor" not in st.session_state:
  st.session_state.code_editor = ""

if "original_flawed_code" not in st.session_state:
  st.session_state.original_flawed_code = ""

if "selected_concept" not in st.session_state:
  st.session_state.selected_concept = "Linear Search"

if "expected_fix" not in st.session_state:
  st.session_state.expected_fix = ""

if "attempt_count" not in st.session_state:
  st.session_state.attempt_count = 1

if "sandbox_output" not in st.session_state:
  st.session_state.sandbox_output = ""


# =====================================================================
# 3. Helper Functions
# =====================================================================
def generate_new_problem(concept_name: str):
  """Calls the FastAPI backend to fetch a flawed code snippet for the selected concept."""
  try:
    with st.spinner(f"Generating flawed code for '{concept_name}'..."):
      response = requests.post(
          f"{API_BASE_URL}/generate", json={"concept": concept_name}, timeout=15
      )

    if response.status_code == 200:
      data = response.json()
      flawed_code = data.get("flawed_code", "")
      st.session_state.code_editor = flawed_code
      st.session_state.original_flawed_code = flawed_code
      st.session_state.expected_fix = data.get("expected_fix", "")
      st.session_state.selected_concept = concept_name
      st.session_state.attempt_count = 1
      st.session_state.sandbox_output = ""

      st.session_state.messages = [{
          "role": "assistant",
          "content": (
              f"I've generated a code snippet for **{concept_name}** with a"
              " subtle logic bug. Inspect the code on the left, test it in the"
              " sandbox, or explain the flaw to me!"
          ),
      }]
    else:
      st.error(f"Failed to generate problem. API Status: {response.status_code}")
  except Exception as e:
    st.error(
        f"Could not connect to FastAPI server at `{API_BASE_URL}`. Ensure it is"
        f" running! Details: {e}"
    )


# =====================================================================
# 4. Sidebar Controls
# =====================================================================
with st.sidebar:
  st.title("🧠 Control Panel")
  st.write("Select a topic to generate a debugging exercise:")

  concepts_list = [
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

  chosen_concept = st.selectbox(
      "Topic:",
      options=concepts_list,
      index=concepts_list.index(st.session_state.selected_concept)
      if st.session_state.selected_concept in concepts_list
      else 0,
  )

  if st.button("🚀 Generate Problem"):
    generate_new_problem(chosen_concept)

  st.write("---")

  if st.button("🔄 Reset Conversation"):
    st.session_state.messages = []
    st.session_state.attempt_count = 1
    st.session_state.sandbox_output = ""
    safe_rerun()

  st.markdown("---")
  st.caption(f"**Active Concept:** {st.session_state.selected_concept}")
  st.caption(f"**Attempts:** {st.session_state.attempt_count}")


# =====================================================================
# 5. Main UI Layout (2 Columns)
# =====================================================================
st.title("🧠 AI Socratic Evaluator")
st.markdown(
    "Analyze code, find hidden logic bugs, test edge cases, and refine your"
    " reasoning with Socratic dialogue."
)

col_code, col_chat = st.columns([1, 1], gap="medium")

# ---------------------------------------------------------------------
# Left Column: Code Review & Interactive Sandbox
# ---------------------------------------------------------------------
with col_code:
  st.subheader("🔍 Code Snippet under Review")
  st.caption("Edits made here are saved automatically upon clicking outside:")

  # Separate text widget key from session variable to eliminate warning banner
  edited_code = st.text_area(
      label="Code Editor",
      value=st.session_state.code_editor,
      key="code_editor_input",
      height=260,
  )
  st.session_state.code_editor = edited_code

  if st.button("↺ Reset Code to Original"):
    st.session_state.code_editor = st.session_state.original_flawed_code
    safe_rerun()

  st.subheader("🧪 Interactive Sandbox")
  st.caption("Add test drivers below to run your function:")

  default_test = (
      "# Write driver code to test your fix:\nprint(linear_search([10, 20,"
      " 30], 20))\n"
  )
  test_script = st.text_area(
      label="Test Script",
      value=default_test,
      height=110,
      key="sandbox_script",
  )

  if st.button("▶️ Run Sandbox Test"):
    try:
      run_payload = {
          "flawed_code": st.session_state.code_editor,
          "test_script": test_script,
      }
      res = requests.post(f"{API_BASE_URL}/run", json=run_payload, timeout=10)
      if res.status_code == 200:
        st.session_state.sandbox_output = res.json().get(
            "output", "No output returned."
        )
      else:
        st.session_state.sandbox_output = f"Execution Error: {res.status_code}"
    except Exception as err:
      st.session_state.sandbox_output = f"Could not connect to server: {err}"

  if st.session_state.sandbox_output:
    st.text_area(
        "Execution Output:",
        value=st.session_state.sandbox_output,
        height=100,
        disabled=True,
    )

# ---------------------------------------------------------------------
# Right Column: Socratic Dialogue Chat
# ---------------------------------------------------------------------
with col_chat:
  st.subheader("💬 Socratic Dialogue")

  chat_container = st.container()

  with chat_container:
    for msg in st.session_state.messages:
      if msg["role"] == "user":
        st.markdown(f"**🧑‍💻 You:** {msg['content']}")
      else:
        st.markdown(f"**🤖 Tutor:** {msg['content']}")
      st.markdown("---")

  with st.form(key="chat_form", clear_on_submit=True):
    user_prompt = st.text_input(
        "Explain the flaw or respond to the hint:", key="user_chat_input"
    )
    submit_chat = st.form_submit_button("Send Response 💬")

  if submit_chat and user_prompt.strip():
    # 1. Append user prompt to UI chat history
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # 2. Build payload aligning with EvalRequest schema in main.py
    payload = {
        "concept": st.session_state.selected_concept,
        "flawed_code": st.session_state.original_flawed_code,
        "current_code": st.session_state.code_editor,
        "chat_history": st.session_state.messages,
        "attempt_count": st.session_state.attempt_count,
        "expected_fix": st.session_state.expected_fix,
    }

    # 3. Post to FastAPI Evaluator
    try:
      with st.spinner("Tutor is inspecting your code and answer..."):
        res = requests.post(
            f"{API_BASE_URL}/evaluate", json=payload, timeout=20
        )

      if res.status_code == 200:
        eval_data = res.json()
        feedback = eval_data.get("feedback", "No feedback provided.")
        is_correct = eval_data.get("is_correct", False)

        st.session_state.messages.append(
            {"role": "assistant", "content": feedback}
        )
        st.session_state.attempt_count += 1

        if is_correct:
          st.balloons()
          st.success("🎉 Fantastic! You solved the problem correctly.")

      else:
        st.error(f"Error from API server: {res.status_code}")

    except Exception as e:
      st.error(f"Failed to connect to `/evaluate` endpoint: {e}")

    safe_rerun()
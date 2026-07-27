import requests
import streamlit as st

st.set_page_config(page_title="AI Socratic Evaluator", layout="centered")

st.title("🧠 AI Socratic Evaluator")
st.write(
    "Analyze the AI's code, find the hidden logic flaw, and refine your"
    " critique iteratively."
)

if "concept" not in st.session_state:
  st.session_state["concept"] = ""
if "flawed_code" not in st.session_state:
  st.session_state["flawed_code"] = None
if "evaluated" not in st.session_state:
  st.session_state["evaluated"] = False
if "feedback" not in st.session_state:
  st.session_state["feedback"] = ""
if "is_correct" not in st.session_state:
  st.session_state["is_correct"] = False

if not st.session_state["flawed_code"]:
  concept = st.text_input(
      "Enter a CS Concept (e.g., Binary Search, SQL Left Join):"
  )

  if st.button("Generate Practice Problem"):
    if not concept:
      st.warning("Please enter a concept first.")
    else:
      with st.spinner("Generating flawed code via RAG..."):
        try:
          res = requests.post(
              "http://localhost:8000/api/generate", json={"concept": concept}
          )
          if res.status_code == 200:
            st.session_state["flawed_code"] = res.json()["flawed_code"]
            st.session_state["concept"] = concept
            st.session_state["evaluated"] = False
            st.session_state["feedback"] = ""
            st.session_state["is_correct"] = False
          else:
            st.error(f"Backend error: {res.text}")
        except requests.exceptions.ConnectionError:
          st.error(
              "Could not connect to FastAPI backend. Make sure uvicorn is"
              " running!"
          )

if st.session_state["flawed_code"]:
  st.markdown("### 🔍 Review this code snippet:")
  with st.expander("📄 View Tutorial & Code Snippet", expanded=True):
    st.markdown(st.session_state["flawed_code"])

  if st.session_state["evaluated"]:
    if st.session_state["is_correct"]:
      st.success(f"**Correct!** {st.session_state['feedback']}")
    else:
      st.warning(f"**Not quite.** {st.session_state['feedback']}")
    st.markdown("---")

  critique = st.text_area(
      "Identify and explain the hidden error:", key="critique_input"
  )

  col1, col2 = st.columns(2)

  with col1:
    button_label = (
        "Re-submit Critique"
        if st.session_state["evaluated"]
        else "Submit Critique"
    )
    if st.button(button_label):
      if not critique.strip():
        st.warning("Please enter a critique before submitting.")
      else:
        with st.spinner("Evaluating your critique..."):
          try:
            res = requests.post(
                "http://localhost:8000/api/evaluate",
                json={
                    "concept": st.session_state["concept"],
                    "flawed_code": st.session_state["flawed_code"],
                    "critique": critique,
                },
            )
            if res.status_code == 200:
              data = res.json()
              st.session_state["evaluated"] = True
              st.session_state["is_correct"] = data.get("is_correct", False)
              st.session_state["feedback"] = data.get("feedback", "")
            else:
              st.error(f"Backend error: {res.text}")
          except requests.exceptions.ConnectionError:
            st.error("Could not connect to FastAPI backend.")

  with col2:
    if st.button("🔄 Try Another Problem"):
      for key in [
          "flawed_code",
          "evaluated",
          "concept",
          "feedback",
          "is_correct",
          "critique_input",
      ]:
        if key in st.session_state:
          del st.session_state[key]
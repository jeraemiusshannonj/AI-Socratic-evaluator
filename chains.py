from contextlib import contextmanager
import os
import pandas as pd
from chromadb import PersistentClient
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from utils import clean_and_extract_code

load_dotenv()

# =====================================================================
# 1. ChromaDB Initialization & CSV Re-indexing
# =====================================================================
chroma_client = PersistentClient(path="./chroma_db")

# Delete old collection to force re-indexing whenever the server restarts
try:
  chroma_client.delete_collection("misconceptions")
except Exception:
  pass

collection = chroma_client.create_collection(name="misconceptions")

# Populate vector store from fresh CSV with robust quoting support
if os.path.exists("misconceptions.csv"):
  df = pd.read_csv("misconceptions.csv", quotechar='"', skipinitialspace=True)
  for index, row in df.iterrows():
    collection.add(
        documents=[str(row["Misconception"])],
        metadatas=[{"concept": str(row["Concept"])}],
        ids=[str(index)],
    )
  print(
      f"✅ Successfully indexed {collection.count()} misconceptions into"
      " ChromaDB."
  )

# =====================================================================
# 2. LLM Configuration
# =====================================================================
generator_llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b",
    temperature=0.3,
)

eval_llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b",
    temperature=0.0,
)


# =====================================================================
# 3. Flawed Code Generation Function
# =====================================================================
def generate_flaw(concept: str) -> dict:
  """Queries ChromaDB for the EXACT misconception matching the selected concept."""
  # Filter strictly by metadata 'concept' key
  results = collection.get(where={"concept": concept})

  if results and results.get("documents") and len(results["documents"]) > 0:
    retrieved_error = results["documents"][0]
  else:
    # Fallback if concept is not found in CSV
    retrieved_error = (
        "Introduce a subtle logic error related to loop termination."
    )

  prompt = PromptTemplate.from_template(
      "You are a code generator creating flawed Python function definitions for"
      " debugging exercises.\n\n"
      "TOPIC: {concept}\n"
      "REQUIRED INTENTIONAL BUG: {error}\n\n"
      "STRICT RULES:\n"
      "1. Output ONLY the Python function definition (`def ...`).\n"
      "2. You MUST corrupt/break the function logic by explicitly injecting the"
      " REQUIRED INTENTIONAL BUG.\n"
      "3. ABSOLUTELY NO driver code, test inputs, arrays, or print statements"
      " outside the function (NO `arr = ...`, NO `target = ...`, NO"
      " `print(...)`).\n"
      "4. DO NOT include comments explaining or pointing out the bug.\n\n"
      "Now, generate ONLY the broken function definition for:\n"
      "Topic: {concept}\n"
      "Bug to inject: {error}\n"
  )

  chain = prompt | generator_llm
  raw_response = chain.invoke(
      {"concept": concept, "error": retrieved_error}
  ).content
  flawed_code = clean_and_extract_code(raw_response)

  return {"flawed_code": flawed_code, "expected_fix": retrieved_error}

# =====================================================================
# 4. Multi-turn Socratic Evaluation Function
# =====================================================================
def evaluate_student_multi_turn(
    current_code: str,
    chat_history: list,
    attempt_count: int,
    expected_fix: str = "",
    concept: str = "Computer Science Logic",
) -> dict:
  """Generically evaluates student dialogue and code against expected fix across ANY topic."""
  parser = JsonOutputParser()

  if attempt_count <= 1:
    scaffolding_guidance = (
        "Ask a subtle, open-ended question that encourages the student to"
        " re-check the flawed logic on their own."
    )
  elif attempt_count == 2:
    scaffolding_guidance = (
        "Focus the student's attention directly on the specific line or operator"
        " where the logic error occurs, without giving away the complete"
        " solution."
    )
  else:
    scaffolding_guidance = (
        "Provide a clear, direct hint explaining what needs to be updated to"
        " fix the bug."
    )

  # Sanitize and format chat history to prevent completion/echoing loops
  formatted_history = []
  for msg in chat_history[-6:]:  # Keep last 6 messages max to prevent token bloat
    role_label = "Student" if msg["role"] == "user" else "Tutor"
    formatted_history.append(f"[{role_label}]: {msg['content']}")
  history_str = (
      "\n".join(formatted_history) if formatted_history else "No chat history"
      " yet."
  )

  template_str = (
      "You are an expert Socratic CS Tutor evaluating a student's code and"
      " reasoning.\n\n"
      "### TASK CONTEXT:\n"
      "- **Topic:** {concept}\n"
      "- **Target Bug to Fix:** {expected_fix}\n\n"
      "### CODE SNIPPET TO EVALUATE (CURRENT VERSION):\n"
      "```python\n"
      "{code}\n"
      "```\n\n"
      "### RECENT CHAT HISTORY:\n"
      "{history}\n\n"
      "### STRICT EVALUATION RUBRIC:\n"
      "You must analyze the **CODE SNIPPET TO EVALUATE (CURRENT VERSION)**"
      " structurally:\n"
      "1. Identify the core bug described in '{expected_fix}'.\n"
      "2. Examine the code lines carefully: Check the indentation, loop structure,"
      " and placement of `return` statements.\n"
      "3. **If the code has successfully fixed the bug** (e.g., placing the"
      " failure return statement outside the loop, correcting loop boundaries,"
      " or adjusting index logic), you **MUST** set `\"is_correct\": true`"
      " immediately, regardless of what the student wrote in the chat.\n"
      "4. **If the bug is still present**, set `\"is_correct\": false` and"
      " provide a helpful, non-repetitive hint based on: {scaffolding_guidance}\n\n"
      "Respond strictly in valid JSON format using this exact schema:\n"
      "{{\n"
      '  "reasoning": "Detailed explanation of why the code is correct or'
      ' incorrect based on line structure",\n'
      '  "is_correct": true_or_false_boolean,\n'
      '  "feedback": "Your response to the student"\n'
      "}}\n"
  )

  prompt = PromptTemplate.from_template(template_str)
  chain = prompt | eval_llm | parser

  return chain.invoke({
      "concept": concept,
      "code": current_code,
      "expected_fix": expected_fix or "Correct the logic bug.",
      "history": history_str,
      "scaffolding_guidance": scaffolding_guidance,
  })
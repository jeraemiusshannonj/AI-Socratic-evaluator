# 🧠 AI Socratic Evaluator

An interactive, RAG-powered AI tutor designed to teach computer science concepts through Socratic evaluation. Instead of just giving away answers, the app presents students with code snippets containing subtle logic errors, evaluates their written critiques, and provides guided Socratic feedback to help them discover the bug themselves.

---

## 🚀 Features

* **Retrieval-Augmented Generation (RAG):** Dynamically retrieves targeted conceptual tutorials and flawed code examples from a custom dataset using **ChromaDB**.
* **FastAPI Backend:** A lightweight, high-performance asynchronous REST API managing vector searches and LLM communication.
* **Streamlit Frontend:** A clean, responsive UI supporting stateful navigation, expandable tutorial boxes, and iterative critique re-submission.
* **Groq LLM Integration:** Powered by high-speed inference to evaluate user explanations and deliver intelligent Socratic hints.

---

## 🛠️ Tech Stack

* **Backend:** FastAPI, Uvicorn, Pydantic
* **AI / Vector Store:** ChromaDB, LangChain / Groq SDK
* **Frontend:** Streamlit
* **Data Source:** Custom CSV (`misconceptions.csv`)

---

## 📂 Project Structure

```text
AI-socratic-evaluator/
│
├── .env                  # Environment variables (Groq API Key)
├── main.py               # FastAPI backend server
├── app.py                # Streamlit frontend UI
├── misconceptions.csv    # Knowledge base of concepts and flawed code
└── README.md             # Project documentation
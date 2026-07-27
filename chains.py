import os
import pandas as pd
import chromadb
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="misconceptions")

if collection.count() == 0:
    df = pd.read_csv("misconceptions.csv")
    for index, row in df.iterrows():
        collection.add(
            documents=[row['Misconception']],
            metadatas=[{"concept": row['Concept']}],
            ids=[str(index)]
        )


llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.2
)



def generate_flaw(concept: str) -> str:
    results = collection.query(query_texts=[concept], n_results=1)
    retrieved_error = results['documents'][0][0] if results['documents'][0] else "Make a subtle logic error."

    prompt = PromptTemplate.from_template(
        """You are a CS tutor. The student is learning: {concept}.
        Generate a brief explanation and a Python/SQL code snippet.
        You MUST intentionally include this specific logic error in your code: {error}
        DO NOT explain or announce the error. Make it look like a normal tutorial.
        """
    )
    chain = prompt | llm
    response = chain.invoke({"concept": concept, "error": retrieved_error})
    return response.content


def evaluate_student(flawed_code: str, critique: str) -> dict:
    parser = JsonOutputParser()
    prompt = PromptTemplate(
        template="""You are a CS professor. Review the flawed code and the student's critique.
        Flawed Code: {code}
        Student Critique: {critique}

        Did the student correctly identify the hidden logic error?
        Respond strictly in JSON with two keys:
        - "is_correct": boolean (true/false)
        - "feedback": string (A Socratic hint if wrong, or congratulatory explanation if right. Do not give the direct answer if wrong).

        {format_instructions}""",
        input_variables=["code", "critique"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    json_llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0).bind(response_format={"type": "json_object"})
    chain = prompt | json_llm | parser

    return chain.invoke({"code": flawed_code, "critique": critique})
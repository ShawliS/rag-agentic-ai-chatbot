import os
from typing import TypedDict
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from openai import OpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

INDEX_NAME = "agentic-ai-ebook"
TOP_K = 4
SCORE_THRESHOLD = 0.3
GROQ_MODEL = "openai/gpt-oss-20b"

# --- Shared clients (loaded once, reused for every question) ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index(INDEX_NAME)
groq_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)


class RAGState(TypedDict):
    question: str
    chunks: list[str]
    top_score: float
    answer: str


def retrieve_node(state: RAGState) -> RAGState:
    question = state["question"]
    query_vector = embedding_model.encode(question).tolist()

    results = pinecone_index.query(
        vector=query_vector,
        top_k=TOP_K,
        include_metadata=True,
    )

    matches = results["matches"]
    chunks = [match["metadata"]["text"] for match in matches]
    top_score = matches[0]["score"] if matches else 0.0

    return {**state, "chunks": chunks, "top_score": top_score}


def generate_node(state: RAGState) -> RAGState:
    if state["top_score"] < SCORE_THRESHOLD:
        return {
            **state,
            "answer": "I don't have information about that in the provided document.",
        }

    context = "\n\n".join(state["chunks"])
    system_prompt = (
        "You are a helpful assistant that answers questions using ONLY the "
        "provided context from a document. Do not use any outside knowledge. "
        "If the context does not contain enough information to answer the "
        "question, respond exactly with: "
        "'I don't have information about that in the provided document.'"
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {state['question']}"

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content
    return {**state, "answer": answer}


def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


rag_app = build_graph()


def ask_question(question: str) -> dict:
    result = rag_app.invoke({"question": question})
    return {
        "answer": result["answer"],
        "chunks": result["chunks"],
        "confidence": round(result["top_score"], 3),
    }


if __name__ == "__main__":
    test_questions = [
        "What is agentic AI?",
        "What is the capital of France?",
    ]
    for q in test_questions:
        print(f"\nQ: {q}")
        result = ask_question(q)
        print(f"A: {result['answer']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Chunks used: {len(result['chunks'])}")
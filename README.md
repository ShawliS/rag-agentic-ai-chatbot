# Agentic AI eBook RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions strictly based on the ["Agentic AI: An Executive's Guide"](https://konverge.ai/pdf/Ebook-Agentic-AI.pdf) eBook by Konverge AI. Built with LangGraph, Pinecone, and free local embeddings.
- Live Demo:https://rag-agentic-ai-chatbot-apqejbheg9q4xjjttgyjn7.streamlit.app/

## Features

- Ingests and chunks the source PDF automatically
- Stores semantic embeddings in a Pinecone vector database
- Retrieves the most relevant chunks for any user question
- Generates answers strictly grounded in the retrieved content using an LLM
- Explicitly refuses to answer when the document doesn't contain relevant information
- Returns the final answer, the retrieved source chunks, and a confidence score
- Interactive Streamlit chat interface

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.14 |
| PDF Extraction | pypdf |
| Text Chunking | LangChain Text Splitters |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`, local, free) |
| Vector Database | Pinecone (serverless) |
| Orchestration | LangGraph |
| LLM | Groq (`openai/gpt-oss-20b`) |
| UI | Streamlit |

## Architecture

**Ingestion (run once):**
```
PDF → Extract Text → Chunk Text → Generate Embeddings → Store in Pinecone
```

**Query time (every user question):**
```
User Question
   → Embed Question (same model as ingestion)
   → Pinecone Similarity Search (top 4 chunks)
   → LangGraph:
       [Retrieve Node] → [Generate Node]
   → Score Gate: if best similarity < 0.3, refuse immediately
   → Otherwise: LLM generates answer strictly from retrieved chunks
   → Return: answer + retrieved chunks + confidence score
   → Displayed in Streamlit chat UI
```

## Grounding Strategy

The chatbot uses two layers to ensure answers are strictly grounded in the source document:

1. **Similarity threshold gate:** If the top retrieved chunk's similarity score falls below 0.3, the system returns a refusal without calling the LLM at all.
2. **Prompt-level instruction:** The LLM is explicitly instructed to answer only from the provided context and to say so if the context doesn't contain the answer — this catches cases where retrieved chunks are topically related but don't actually answer the specific question.

## Setup Instructions

1. Clone this repository:
```bash
   git clone <your-repo-url>
   cd rag-agentic-ai-chatbot
```

2. Create and activate a virtual environment:
```bash
   python -m venv venv
   venv\Scripts\activate    # Windows
   source venv/bin/activate # Mac/Linux
```

3. Install dependencies:
```bash
   pip install -r requirements.txt
```

4. Create a `.env` file in the project root with:
```
PINECONE_API_KEY=your_pinecone_key
GROQ_API_KEY=your_groq_key
```

5. Run ingestion (downloads the PDF, chunks it, embeds it, and stores it in Pinecone):
```bash
   python ingest.py
```

6. Launch the chatbot:
```bash
   streamlit run app.py
```

## Sample Queries

| Question | Result |
|---|---|
| What is agentic AI? | Grounded answer, confidence 0.796 |
| What is the difference between traditional AI tools and agentic AI systems? | Grounded answer, confidence 0.682 |
| What challenges are involved in implementing agentic AI, like regulatory compliance? | Grounded answer, confidence 0.665 |
| What is Emergence AI's contribution mentioned in the book? | Grounded answer, confidence 0.566 |
| What industries or use cases does the book mention for agentic AI? | Correctly declined — chunks were topically related but didn't contain a direct answer |
| What is the capital of France? | Correctly declined — out of scope, confidence 0.093 |

## Limitations

- Retrieval is limited to the top 4 chunks per question; some valid answers spread across more distant parts of the document may not surface (see the "industries" example above).
- Uses a free local embedding model (384 dimensions), which is slightly lower quality than commercial alternatives like OpenAI's embeddings, though sufficient for this single-document use case.
- The similarity threshold (0.3) was set based on limited manual testing and could benefit from more systematic tuning.

## Future Improvements

- Add a re-ranking step after retrieval for improved relevance
- Support multiple source documents
- Add automated evaluation tests for grounding accuracy
- Deploy publicly (e.g., Streamlit Community Cloud)

## Note on Development

This project avoids no-code/low-code AI platforms; the RAG pipeline, retrieval logic, and grounding strategy were implemented directly in Python using the specified frameworks (LangGraph, Pinecone).
**Demo video**
https://github.com/user-attachments/assets/21fc5e52-42b8-4a66-81df-3227eab52d89


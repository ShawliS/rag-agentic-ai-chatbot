import requests
from io import BytesIO
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()

PDF_URL = "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"
INDEX_NAME = "agentic-ai-ebook"
EMBED_DIM = 384  # matches all-MiniLM-L6-v2

def download_pdf_text(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()
    pdf_file = BytesIO(response.content)
    reader = PdfReader(pdf_file)

    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    return splitter.split_text(text)

def get_embedding_model():
    print("Loading local embedding model (first run downloads ~80MB)...")
    return SentenceTransformer("all-MiniLM-L6-v2")

def setup_pinecone_index():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    else:
        print(f"Index '{INDEX_NAME}' already exists, reusing it.")

    return pc.Index(INDEX_NAME)

def embed_and_upload(chunks: list[str], model, index):
    print(f"Embedding {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True)

    vectors_to_upsert = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors_to_upsert.append({
            "id": f"chunk-{i}",
            "values": embedding.tolist(),
            "metadata": {"text": chunk},
        })

    print("Uploading to Pinecone...")
    index.upsert(vectors=vectors_to_upsert)
    print("Done!")

if __name__ == "__main__":
    print("Downloading and extracting PDF text...")
    raw_text = download_pdf_text(PDF_URL)
    print(f"Extracted {len(raw_text)} characters total.\n")

    chunks = chunk_text(raw_text)
    print(f"Split into {len(chunks)} chunks.\n")

    model = get_embedding_model()
    index = setup_pinecone_index()
    embed_and_upload(chunks, model, index)

    stats = index.describe_index_stats()
    print("\nPinecone index stats:", stats)
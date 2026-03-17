import os
import sys
import tiktoken
from openai import AzureOpenAI
from dotenv import load_dotenv

# Allow running from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.chroma import collection

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    enc = tiktoken.get_encoding(model)
    return len(enc.encode(text))


def chunk_text(text: str, max_tokens: int = 400, overlap_tokens: int = 50) -> list[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    start = 0

    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk = enc.decode(chunk_tokens)
        chunks.append(chunk)
        if end >= len(tokens):
            break
        start += max_tokens - overlap_tokens

    return chunks


def embed(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    return [item.embedding for item in response.data]


def ingest(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print(f"📄 Loaded: {filepath} ({count_tokens(raw_text)} tokens)")

    chunks = chunk_text(raw_text)
    print(f"   Chunks: {len(chunks)}")

    print("🔢 Embedding...")
    embeddings = embed(chunks)

    print("💾 Storing in ChromaDB (persistent)...")
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"source": filepath, "chunk_index": i} for i in range(len(chunks))],
    )

    print(f"✅ Done. {len(chunks)} chunks stored.")


if __name__ == "__main__":
    ingest("data/songwriting_guide.md")
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from openai import AzureOpenAI
from dotenv import load_dotenv
from services.chroma import collection

load_dotenv()

router = APIRouter()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
CHAT_MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")


# ── Prompt templates ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are SongMind, an expert songwriting assistant.

Answer the user's question using ONLY the context provided below.
If the context does not contain enough information to answer confidently,
say "I don't have enough information on that — try rephrasing or asking 
something about rhyme schemes, song structure, or lyric techniques."

Do not invent information. Do not reference external sources.
Keep answers concise, practical, and useful to a songwriter.

Context:
{context}
"""

# ── Models ─────────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    n_results: int = 4  # top N chunks to retrieve


class AskResponse(BaseModel):
    answer: str
    chunks_used: int
    sources: list[str]  # the actual retrieved chunks, useful for debugging


# ── Helpers ────────────────────────────────────────────────────────────────────

def embed_query(text: str) -> list[float]:
    response = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return response.data[0].embedding


def retrieve(query_embedding: list[float], n_results: int) -> list[str]:
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )
    return results["documents"][0]  # list of chunk strings


# ── Endpoint ───────────────────────────────────────────────────────────────────

@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        # Step 1 — embed the question
        query_embedding = embed_query(request.question)

        # Step 2 — retrieve relevant chunks from Chroma
        chunks = retrieve(query_embedding, request.n_results)

        # Step 3 — build context string from chunks
        context = "\n\n---\n\n".join(chunks)

        # Step 4 — inject into system prompt and call the LLM
        filled_prompt = SYSTEM_PROMPT.format(context=context)

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": filled_prompt},
                {"role": "user", "content": request.question},
            ],
            max_tokens=600,
            temperature=0.4,  # lower temp = more grounded, less creative
        )

        return AskResponse(
            answer=response.choices[0].message.content,
            chunks_used=len(chunks),
            sources=chunks,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
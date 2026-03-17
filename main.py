from fastapi import FastAPI
from routers import chat, rag

app = FastAPI(
    title="SongAssist API",
    description="AI-powered song-writing assistant",
    version="0.1.0"
)

app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(rag.router, prefix="/api/v1", tags=["rag"])

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "song-assist-api"}

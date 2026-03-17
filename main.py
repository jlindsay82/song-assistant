from fastapi import FastAPI
from routers import chat

app = FastAPI(
    title="SongMind API",
    description="AI-powered lyric assistant",
    version="0.1.0"
)

app.include_router(chat.router, prefix="/api/v1", tags=["chat"])

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "songmind-api"}

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from openai import AzureOpenAI
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from the project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

router = APIRouter()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")


class ChatRequest(BaseModel):
    message: str
    system_prompt: str = "You are a helpful lyric writing assistant."


class ChatResponse(BaseModel):
    reply: str
    model: str
    usage: dict


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.message},
            ],
            max_tokens=500,
            temperature=0.7,
        )

        return ChatResponse(
            reply=response.choices[0].message.content,
            model=response.model,
            usage=response.usage.model_dump(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
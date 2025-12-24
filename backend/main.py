import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from rag import chat
from indexer import index_articles

app = FastAPI(title="RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


class Citation(BaseModel):
    title: str
    url: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = chat(request.question)
        return ChatResponse(
            answer=result["answer"],
            citations=[Citation(**c) for c in result["citations"]]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index")
async def index_endpoint():
    try:
        index_articles()
        return {"status": "success", "message": "Articles indexed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

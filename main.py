
from __future__ import annotations

from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


try:
    import agent  
except Exception as e:
    agent = None
    _agent_err = e

try:
    import rag_app  
except Exception as e:
    rag_app = None
    _rag_err = e

try:
    import vectorStore  
except Exception as e:
    vectorStore = None
    _vs_err = e

app = FastAPI(title="Agent Backend", version="1.0.1", description="Strict wiring to agent.py, vectorStore.py, rag_app.py")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_chatbot = None
_rag = None
_vector = None

def get_chatbot():
    global _chatbot
    if _chatbot is None:
        if agent is None:
            raise HTTPException(500, f"Failed to import agent.py: {_agent_err!s}")
        try:
            _chatbot = agent.ChatBot()
        except Exception as e:
            raise HTTPException(500, f"Could not init ChatBot: {e!s}")
    return _chatbot

def get_rag():
    global _rag
    if _rag is None:
        if rag_app is None:
            raise HTTPException(500, f"Failed to import rag_app.py: {_rag_err!s}")
        try:
            _rag = rag_app.RAGApp()
            _rag.setup()
        except Exception as e:
            raise HTTPException(500, f"Could not init RAGApp: {e!s}")
    return _rag

def get_vector():
    global _vector
    if _vector is None:
        if vectorStore is None:
            raise HTTPException(500, f"Failed to import vectorStore.py: {_vs_err!s}")
        try:
            _vector = vectorStore.QdrantVector()
            # QdrantVector is expected to connect inside its methods or __init__.
            # If it needs explicit connect, uncomment next line:
            # _vector.connect_client()
        except Exception as e:
            raise HTTPException(500, f"Could not init QdrantVector: {e!s}")
    return _vector

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

class AskRequest(BaseModel):
    question: str
    k: int = 3

class AskResponse(BaseModel):
    answer: str

class SearchRequest(BaseModel):
    query: str
    k: int = 3

class SearchResponse(BaseModel):
    results: list

@app.get("/health")
def health():
    return {"status": "ok"}

# Exact: agent.ChatBot.chat(message: str) -> str
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    bot = get_chatbot()
    try:
        reply = bot.chat(req.message)
    except Exception as e:
        raise HTTPException(500, f"Chat error: {e!s}")
    return ChatResponse(reply=reply)

# Upload + add via *existing* RAGApp.add_document(file_path)
@app.post("/documents/upload")
async def upload_documents(files: List[UploadFile] = File(...)) -> dict:
    rag = get_rag()
    saved = []
    added = 0
    for f in files:
        data = await f.read()
        path = f"/tmp/{f.filename}"
        with open(path, "wb") as out:
            out.write(data)
        saved.append(path)
        try:
            ok = rag.add_document(path)  # uses your existing method
            if ok is not False:  # treat truthy/None as success
                added += 1
        except Exception as e:
            raise HTTPException(500, f"add_document failed for {f.filename}: {e!s}")
    return {"uploaded_files": saved, "added_count": added}

# Ask via RAGApp.ask(question: str, k: int = 3)
@app.post("/rag/ask", response_model=AskResponse)
def rag_ask(req: AskRequest):
    rag = get_rag()
    try:
        ans = rag.ask(req.question, k=req.k)
    except Exception as e:
        raise HTTPException(500, f"RAG ask failed: {e!s}")
    return AskResponse(answer=ans)

# Search via QdrantVector.find_similar_texts(query: str, k: int = 3)
@app.post("/documents/search", response_model=SearchResponse)
def search(req: SearchRequest):
    vs = get_vector()
    try:
        res = vs.find_similar_texts(req.query, k=req.k)
    except Exception as e:
        raise HTTPException(500, f"Vector search failed: {e!s}")

    # Normalize to plain list for JSON: if Documents, convert
    out = []
    for r in res or []:
        try:
            # common for LangChain Document
            if hasattr(r, "page_content"):
                out.append({"text": r.page_content, "metadata": getattr(r, "metadata", {})})
            else:
                out.append(r)
        except Exception:
            out.append(str(r))
    return SearchResponse(results=out)

@app.get("/")
def root():
    return {"routes": ["/chat", "/documents/upload", "/documents/search", "/rag/ask", "/health"]}

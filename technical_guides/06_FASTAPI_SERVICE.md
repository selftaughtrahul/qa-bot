# Technical Guide 06: FastAPI Service

## 📋 Overview
This guide covers building the **FastAPI REST API** that exposes the QA Bot's document ingestion and question-answering capabilities as HTTP endpoints — enabling integration with any frontend or external system.

---

## Step 1: API Entry Point (`api.py`)

```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import uuid

from src.ingestion.pipeline import ingest_document
from src.retrieval.embedder import DocumentEmbedder
from src.retrieval.vector_store import ChromaVectorStore
from src.generation.conversational_rag_chain import ConversationalRAGChain

# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="QA Bot API",
    description="Ask questions on your documents using RAG + LLM.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Global State ───────────────────────────────────────────────────────────────
UPLOAD_DIR = "data/raw"
VECTOR_STORE_DIR = "data/vector_store"
os.makedirs(UPLOAD_DIR, exist_ok=True)

embedder = DocumentEmbedder()
vector_store = ChromaVectorStore()

# Session-based RAG chains (keyed by session_id)
sessions: dict = {}


def get_or_create_chain(session_id: str) -> ConversationalRAGChain:
    if session_id not in sessions:
        sessions[session_id] = ConversationalRAGChain(
            vector_store=vector_store,
            embedder=embedder,
            llm_provider="openai"
        )
    return sessions[session_id]


# ── Request/Response Models ────────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default"


class AnswerResponse(BaseModel):
    answer: str
    sources: list
    session_id: str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "vector_store_size": vector_store.collection.count() if vector_store.collection else 0,
        "active_sessions": len(sessions)
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and ingest a document into the vector store."""
    allowed_types = [".pdf", ".docx", ".txt"]
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # Save uploaded file
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Ingest: parse → chunk → embed → store
    chunks = ingest_document(save_path)
    vector_store.add_chunks(chunks, embedder)
    vector_store.save(VECTOR_STORE_DIR)

    return {
        "status": "success",
        "document": file.filename,
        "chunks_created": len(chunks),
        "total_indexed": vector_store.collection.count()
    }


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """Ask a question against the ingested knowledge base."""
    if vector_store.collection.count() == 0:
        raise HTTPException(status_code=400, detail="No documents ingested yet. Please upload documents first.")

    chain = get_or_create_chain(request.session_id)
    answer, sources = chain.answer(request.question)

    formatted_sources = [
        {
            "document": s["source"],
            "page": s["page"],
            "excerpt": s["content"][:200] + "...",
            "score": round(s.get("similarity_score", 0), 3)
        }
        for s in sources
    ]

    return AnswerResponse(
        answer=answer,
        sources=formatted_sources,
        session_id=request.session_id
    )


@app.delete("/clear/{session_id}")
def clear_session(session_id: str):
    """Clear conversation history for a session."""
    if session_id in sessions:
        sessions[session_id].clear_memory()
        del sessions[session_id]
    return {"status": "cleared", "session_id": session_id}


@app.delete("/reset")
def reset_vector_store():
    """Clear all ingested documents and reset the vector store."""
    global vector_store
    vector_store = ChromaVectorStore()
    sessions.clear()
    return {"status": "vector store reset", "message": "All documents and sessions cleared."}
```

---

## Step 2: Run the API

```bash
# Start the FastAPI server
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Visit the interactive docs at: **http://localhost:8000/docs**

---

## Step 3: Test with cURL / Python

**Upload a document:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@data/raw/policy.pdf"
```

**Ask a question:**
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the leave policy?", "session_id": "user_001"}'
```

**Python client:**
```python
import requests

BASE_URL = "http://localhost:8000"

# Upload
with open("data/raw/policy.pdf", "rb") as f:
    r = requests.post(f"{BASE_URL}/upload", files={"file": f})
print(r.json())

# Ask
r = requests.post(f"{BASE_URL}/ask", json={
    "question": "How many days of leave do employees get?",
    "session_id": "test_session"
})
data = r.json()
print("Answer:", data["answer"])
print("Sources:", data["sources"])
```

---

## Step 4: API Response Examples

**`POST /upload` Response:**
```json
{
  "status": "success",
  "document": "policy.pdf",
  "chunks_created": 87,
  "total_indexed": 87
}
```

**`POST /ask` Response:**
```json
{
  "answer": "Employees are entitled to 20 days of paid annual leave per calendar year.",
  "sources": [
    {
      "document": "policy.pdf",
      "page": 5,
      "excerpt": "Annual leave entitlement is 20 days per calendar year for full-time...",
      "score": 0.912
    }
  ],
  "session_id": "user_001"
}
```

---

## ✅ Checklist
- [ ] FastAPI app runs without errors on port 8000.
- [ ] `/upload` endpoint ingests documents and updates vector store.
- [ ] `/ask` endpoint returns grounded answers with source citations.
- [ ] `/clear/{session_id}` resets conversation memory.
- [ ] Swagger UI accessible at `http://localhost:8000/docs`.
- [ ] Tested with at least 2 documents and 10 questions.

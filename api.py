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


def get_or_create_chain(session_id: str, top_k: int = 4, max_history_turns: int = 5) -> ConversationalRAGChain:
    if session_id not in sessions:
        sessions[session_id] = ConversationalRAGChain(
            vector_store=vector_store,
            embedder=embedder,
            llm_provider="groq",
            top_k=top_k,
            max_history_turns=max_history_turns
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
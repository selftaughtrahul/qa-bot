# Functional Requirements Document (FRD)
## Question Answering System (QA Bot)

---

### 1. Introduction

This document describes the detailed functional behavior of the QA Bot system — covering all use cases, data flows, API contracts, and system interactions.

---

### 2. System Overview

The QA Bot is a RAG-based system composed of three layers:
1. **Ingestion Layer** — Parses, chunks, embeds, and stores documents.
2. **Retrieval Layer** — Performs semantic search over the vector store.
3. **Generation Layer** — Uses an LLM to generate answers from retrieved context.

---

### 3. Use Cases

#### UC-01: Upload Document
- **Actor:** User
- **Input:** One or more files (PDF / DOCX / TXT) via Streamlit file uploader.
- **Action:** System parses text, splits into chunks, generates embeddings, stores in vector DB.
- **Output:** Confirmation message — "✅ 3 document(s) processed. Ready to answer questions."

#### UC-02: Ask a Question
- **Actor:** User
- **Input:** Natural-language question typed in the chat input box.
- **Action:** System embeds the question, retrieves top-k relevant chunks, sends to LLM.
- **Output:** Fluent answer displayed in chat, with source citations below.

#### UC-03: Multi-Turn Conversation
- **Actor:** User
- **Input:** Follow-up question referencing previous context (e.g., "What about the second point?").
- **Action:** System uses conversation memory to resolve references and retrieve correctly.
- **Output:** Contextually accurate answer.

#### UC-04: View Source Citations
- **Actor:** User
- **Action:** Expands "📄 Sources" section below each answer.
- **Output:** List of source chunks with document name, page number, and excerpt.

#### UC-05: Clear Session
- **Actor:** User
- **Action:** Clicks "🗑️ Clear Chat" button.
- **Output:** Chat history and vector store cleared. Ready for new documents.

---

### 4. Data Flow

```
[User Uploads Document]
        ↓
[Document Parser] → Extract raw text (PyMuPDF / python-docx)
        ↓
[Text Splitter] → Chunk text (RecursiveCharacterTextSplitter, chunk_size=500, overlap=50)
        ↓
[Embedding Model] → Generate dense vectors (sentence-transformers / OpenAI)
        ↓
[Vector Store] → Index & persist vectors (ChromaDB)
        ↓
[User Asks Question]
        ↓
[Query Embedder] → Embed question into dense vector
        ↓
[Retriever] → Similarity search → Top-k chunks (k=4)
        ↓
[LLM Chain] → Prompt = System Prompt + Retrieved Context + Question
        ↓
[Answer + Citations] → Displayed in Streamlit Chat UI
```

---

### 5. API Endpoints (FastAPI)

#### `POST /upload`
- **Description:** Upload and ingest a document.
- **Payload:** `multipart/form-data` with file field.
- **Response:** `{"status": "success", "chunks_created": 42, "document": "policy.pdf"}`

#### `POST /ask`
- **Description:** Ask a question against the ingested knowledge base.
- **Payload:** `{"question": "What is the leave policy?", "session_id": "abc123"}`
- **Response:**
```json
{
  "answer": "Employees are entitled to 20 days of annual leave...",
  "sources": [
    {"document": "policy.pdf", "page": 5, "excerpt": "Annual leave entitlement is 20 days..."}
  ]
}
```

#### `DELETE /clear`
- **Description:** Clear the vector store and conversation history for a session.
- **Response:** `{"status": "cleared"}`

#### `GET /health`
- **Response:** `{"status": "ok", "vector_store": "loaded", "documents": 3}`

---

### 6. Prompt Template

```
System: You are a helpful assistant. Answer the user's question ONLY based on the
provided context. If the answer is not in the context, say "I don't know based on
the provided documents."

Context:
{retrieved_chunks}

Conversation History:
{chat_history}

Question: {question}
Answer:
```

---

### 7. Evaluation Metrics (RAGAS Framework)

| Metric | Description |
|--------|-------------|
| **Faithfulness** | Is the answer grounded in the retrieved context? |
| **Answer Relevance** | Does the answer address the question? |
| **Context Recall** | Were the right chunks retrieved? |
| **Context Precision** | Are retrieved chunks relevant (no noise)? |

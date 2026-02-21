# Statement of Work (SOW)
## Question Answering System (QA Bot)

---

### 1. Project Overview

**Project Name:** Question Answering System (QA Bot)

**Project Duration:** 6-8 weeks

**Project Type:** GenAI + NLP — Retrieval-Augmented Generation (RAG) System

**Objective:** Build an intelligent QA Bot that allows users to ask natural-language questions over PDFs, Word documents, or a company knowledge base, and receive accurate, context-grounded answers using embeddings, a vector database, and a large language model (LLM).

---

### 2. Project Background

Enterprises and individuals deal with large volumes of unstructured text — policy documents, research papers, product manuals, and internal wikis. Finding specific answers manually is time-consuming and error-prone. This project leverages the RAG (Retrieval-Augmented Generation) paradigm to combine semantic search with LLM-powered answer generation, enabling fast and accurate Q&A over any document corpus.

---

### 3. Project Scope

#### 3.1 In-Scope

- **Document Ingestion Pipeline**
  - Support PDF, DOCX, and plain-text file formats.
  - Text extraction, chunking, and metadata tagging.

- **Embedding & Vector Store**
  - Generate dense embeddings using `sentence-transformers` or OpenAI Embeddings.
  - Store and index embeddings in a vector database (FAISS / ChromaDB).

- **Retrieval-Augmented Generation (RAG)**
  - Semantic similarity search to retrieve top-k relevant chunks.
  - Feed retrieved context to an LLM (OpenAI GPT / HuggingFace model) for answer generation.

- **Conversational Memory**
  - Maintain multi-turn conversation history using LangChain `ConversationBufferMemory`.

- **API Development**
  - REST API using FastAPI for document upload and Q&A endpoints.

- **User Interface**
  - Streamlit chat interface with file upload, chat history, and source citation display.

#### 3.2 Out-of-Scope

- Real-time web scraping or live internet search.
- Speech-to-text input (future phase).
- Fine-tuning the base LLM on custom data.

---

### 4. Deliverables

| # | Deliverable | Description |
|---|-------------|-------------|
| 1 | Ingestion Pipeline | Script to parse, chunk, embed, and store documents. |
| 2 | Vector Store Index | FAISS/ChromaDB index files for fast retrieval. |
| 3 | RAG Chain | LangChain-based retrieval + generation pipeline. |
| 4 | FastAPI Service | API for document upload and question answering. |
| 5 | Streamlit App | Interactive chat UI with source citations. |
| 6 | Documentation | SOW, BRD, FRD, and module-wise Technical Guides. |

---

### 5. Technical Stack

- **Language:** Python 3.10+
- **Embeddings:** `sentence-transformers` (all-MiniLM-L6-v2), OpenAI `text-embedding-ada-002`
- **Vector DB:** FAISS (local) / ChromaDB (persistent)
- **LLM:** OpenAI GPT-3.5/4 or HuggingFace `mistralai/Mistral-7B-Instruct`
- **Orchestration:** LangChain
- **Document Parsing:** PyMuPDF (`fitz`), `python-docx`, `pypdf`
- **Web:** Streamlit, FastAPI
- **Tools:** Jupyter Notebook, Git, dotenv

---

### 6. Timeline

- **Week 1:** Project setup, environment, document ingestion pipeline
- **Week 2:** Embedding generation & vector store setup
- **Week 3:** RAG chain implementation with LangChain
- **Week 4:** Conversational memory & multi-turn Q&A
- **Week 5:** FastAPI development & integration
- **Week 6:** Streamlit UI development
- **Week 7:** Testing, evaluation, and optimization
- **Week 8:** Documentation & final review

---

### 7. Approval

**Prepared by:** Antigravity AI
**Date:** February 18, 2026

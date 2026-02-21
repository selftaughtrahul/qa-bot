# 📚 QA Bot — Ask Your Documents

## 📖 Project Overview

**QA Bot** is an intelligent Question Answering System built using a Retrieval-Augmented Generation (RAG) paradigm. It allows users to ask natural-language questions over unstructured text documents (PDFs, Word documents, plain text) and receive accurate, context-grounded answers. The system leverages dense embeddings, a vector database, and modern Large Language Models (LLMs) to provide fast and accurate Q&A capabilities over uploaded document corpora.

## ✨ Key Features

- **Document Ingestion Pipeline:** Supports uploading and processing multiple file formats including PDF, DOCX, and TXT files. Extracts text, chunks it, and tags it with metadata.
- **Embedding & Vector Store:** Generates dense embeddings for text chunks using `sentence-transformers` and efficiently stores/indexes them using **FAISS** for fast similarity search.
- **Retrieval-Augmented Generation (RAG):** Uses LangChain to perform semantic search, retrieving top relevant document chunks and feeding them as context to an LLM to generate precise answers.
- **Multiple LLM Providers:** Supports flexible LLM backends including **Groq**, **HuggingFace**, and **Google Gemini***.
- **Conversational Memory:** Preserves multi-turn conversation history for context-aware follow-up questions.
- **Interactive UI:** A highly intuitive **Streamlit** chat interface with file upload capabilities, chat history, and visual source citation display.
- **REST API:** Provides a **FastAPI** backend for document upload and Q&A endpoints.

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Frameworks:** Streamlit, FastAPI, LangChain
- **Embeddings:** `sentence-transformers`
- **Vector Database:** FAISS (Local)
- **Document Parsing:** PyMuPDF (`fitz`), `python-docx`
- **LLM Providers:** Groq, Google GenAI, HuggingFace

## 🗂️ Project Structure

```text
qa_bot/
├── app.py                     # Streamlit web application frontend
├── api.py                     # FastAPI backend application
├── requirements.txt           # Project Python dependencies
├── .env                       # Environment variables (API keys)
├── src/                       # Source code directory
│   ├── ingestion/             # Document parsing and chunking logic
│   ├── retrieval/             # Embedding generation and FAISS vector store management
│   └── generation/            # RAG chain and prompt templates
├── data/                      # Data directory for uploaded files and persistent stores
├── docs/                      # General project documentation
├── notebooks/                 # Jupyter notebooks for testing and experimentation
├── reports/                   # Generated reports and metrics
├── scripts/                   # Utility scripts
├── technical_guides/          # Module-wise technical guides
├── tests/                     # Unit and integration tests
├── SOW.md                     # Statement of Work
├── BRD.md                     # Business Requirements Document
└── FRD.md                     # Functional Requirements Document
```

## 🚀 Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd qa_bot
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   Create a `.env` file in the root directory and add your necessary API keys (e.g., Groq, Gemini, HuggingFace, or OpenAI):
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   # Add any other required keys
   ```

## 🎮 Usage

### Running the Streamlit App

To launch the interactive user interface:

```bash
streamlit run app.py
```

1. Open the provided localserver URL in your browser.
2. Use the **Sidebar** to upload your PDF, DOCX, or TXT documents.
3. Click **⚡ Process Documents** to chunk and embed the files.
4. Start asking questions in the main chat area!

### Running the FastAPI Backend

If you wish to use the REST API rather than the web GUI:

```bash
uvicorn api:app --reload
```

Then, navigate to `http://127.0.0.1:8000/docs` to interact with the Swagger UI for testing the API endpoints.

---
*Prepared by Antigravity AI*

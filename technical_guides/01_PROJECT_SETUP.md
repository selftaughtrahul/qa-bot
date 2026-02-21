# Technical Guide 01: Project Setup & Environment

## 📋 Overview
This guide covers the initial setup of the **Question Answering System (QA Bot)** project — directory structure, virtual environment, and dependency installation.

---

## Step 1: Create Project Directory Structure

Run these commands in your project root `d:\Codebasics\NLP_Projects\qa_bot`:

```bash
# Create subdirectories
mkdir data\raw data\processed data\vector_store
mkdir src\ingestion src\retrieval src\generation src\utils
mkdir notebooks reports docs scripts tests

# Create empty __init__.py files
type nul > src\__init__.py
type nul > src\ingestion\__init__.py
type nul > src\retrieval\__init__.py
type nul > src\generation\__init__.py
type nul > src\utils\__init__.py
```

**Final Structure:**
```
qa_bot/
├── data/
│   ├── raw/              # Uploaded source documents (PDF, DOCX, TXT)
│   ├── processed/        # Extracted & chunked text (JSON)
│   └── vector_store/     # ChromaDB index files
├── src/
│   ├── ingestion/        # Document parsing & chunking
│   ├── retrieval/        # Embedding & vector search
│   ├── generation/       # LLM chain & prompt templates
│   └── utils/            # Config, helpers, logging
├── notebooks/            # EDA & experimentation
├── reports/              # Evaluation results (RAGAS)
├── docs/                 # Project documentation
├── scripts/              # CLI utility scripts
├── tests/                # Unit tests
├── app.py                # Streamlit UI
├── api.py                # FastAPI service
├── requirements.txt
└── .env                  # API keys (never commit!)
```

---

## Step 2: Virtual Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip
```

---

## Step 3: Install Dependencies

Create `requirements.txt`:

```
# Core
langchain==0.1.16
langchain-community==0.0.36
langchain-openai==0.1.3

# Document Parsing
pymupdf==1.24.1
python-docx==1.1.0
pypdf==4.2.0

# Embeddings & Vector Store
sentence-transformers==2.7.0
chromadb
chromadb==0.5.0

# LLM (local option)
transformers==4.40.0
torch==2.2.2

# API & UI
fastapi==0.111.0
uvicorn==0.29.0
streamlit==1.34.0
python-multipart==0.0.9

# Evaluation
ragas==0.1.9

# Utilities
python-dotenv==1.0.1
pandas==2.2.2
numpy==1.26.4
tqdm==4.66.4
```

```bash
pip install -r requirements.txt
```

---

## Step 4: Configure Environment Variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-your-openai-api-key-here
EMBEDDING_MODEL=all-MiniLM-L6-v2
VECTOR_STORE_PATH=data/vector_store
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_RESULTS=4
```

> ⚠️ **Never commit `.env` to Git!** Add it to `.gitignore`.

---

## ✅ Checklist
- [ ] Project directory structure created.
- [ ] Virtual environment set up and activated.
- [ ] All dependencies installed successfully.
- [ ] `.env` file created with API keys.
- [ ] `.gitignore` includes `venv/`, `.env`, `data/vector_store/`.

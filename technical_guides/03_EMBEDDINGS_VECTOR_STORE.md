# Technical Guide 03: Embeddings & Vector Store

## 📋 Overview
This guide covers generating dense vector embeddings from document chunks and storing/indexing them in a ChromaDB vector store for fast semantic retrieval.

---

## Concept: Why Embeddings?

Traditional keyword search (BM25, TF-IDF) matches exact words. Embeddings capture **semantic meaning** — so "annual leave" and "vacation days" are understood as similar concepts.

```
Text Chunk → Embedding Model → Dense Vector [0.12, -0.45, 0.78, ...]
                                                    ↓
                                          Stored in ChromaDB
```

---

## Step 1: Embedding Model Setup (`src/retrieval/embedder.py`)

```python
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import os


class DocumentEmbedder:
    """Generates dense embeddings using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"🔄 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        print("✅ Embedding model loaded.")

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed a list of text strings. Returns a 2D numpy array."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        return self.model.encode([query], normalize_embeddings=True)[0]
```

> 💡 **Model Options:**
> - `all-MiniLM-L6-v2` — Fast, lightweight (384 dims). Best for local use.
> - `all-mpnet-base-v2` — Higher quality (768 dims). Slower.
> - `text-embedding-ada-002` — OpenAI API (1536 dims). Best quality, requires API key.

---

## Step 2: ChromaDB Vector Store (`src/retrieval/vector_store.py`)

```python
import chromadb
import uuid
import numpy as np
import os
from typing import List, Dict
from src.retrieval.embedder import DocumentEmbedder

class ChromaVectorStore:
    """ChromaDB-based vector store for document chunk retrieval."""

    def __init__(self, persist_directory: str = "data/chroma_db", collection_name: str = "qa_docs"):
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Dict], embedder: DocumentEmbedder):
        """Embed chunks and add them to the ChromaDB index."""
        if not chunks:
            return

        texts = [chunk["content"] for chunk in chunks]
        print(f"🔢 Generating embeddings for {len(texts)} chunks...")
        embeddings = embedder.embed_texts(texts)

        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = []
        for chunk in chunks:
            metadata = {k: v for k, v in chunk.items() if k != "content"}
            metadatas.append(metadata)

        # Add to Chroma
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        print(f"✅ Added {len(chunks)} chunks to vector store. Total: {self.collection.count()}")

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> List[Dict]:
        """Retrieve top-k most similar chunks for a query embedding."""
        query_vec = query_embedding.reshape(-1).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k
        )

        chunks_res = []
        if not results['documents'] or not results['documents'][0]:
            return chunks_res

        docs = results['documents'][0]
        metas = results['metadatas'][0] if results['metadatas'] else [{}] * len(docs)
        distances = results['distances'][0] if results['distances'] else [0.0] * len(docs)

        for doc, meta, dist in zip(docs, metas, distances):
            chunk = meta.copy()
            chunk["content"] = doc
            chunk["similarity_score"] = float(1.0 - dist)
            chunks_res.append(chunk)

        return chunks_res

    def save(self, save_dir: str = None):
        """ChromaDB automatically persists data to disk."""
        print("💾 Vector store implicitly saved (ChromaDB is persistent).")

    def load(self, save_dir: str = None):
        """Data is automatically loaded by PersistentClient."""
        print(f"✅ Loaded vector store. Total: {self.collection.count()} vectors.")
```

---

## Step 3: Build the Vector Store (`scripts/build_vector_store.py`)

```python
import json
import glob
from src.retrieval.embedder import DocumentEmbedder
from src.retrieval.vector_store import ChromaVectorStore

# Load all processed chunks
all_chunks = []
for chunk_file in glob.glob("data/processed/*.json"):
    with open(chunk_file, "r") as f:
        all_chunks.extend(json.load(f))

print(f"📦 Total chunks to index: {len(all_chunks)}")

# Initialize embedder and vector store
embedder = DocumentEmbedder(model_name="all-MiniLM-L6-v2")
store = ChromaVectorStore(persist_directory="data/chroma_db", collection_name="qa_docs")

# Build index
store.add_chunks(all_chunks, embedder)

# Save to disk
store.save("data/chroma_db")
```

```bash
python scripts/build_vector_store.py
```

---

## Step 4: Test Retrieval

```python
# Quick retrieval test
from src.retrieval.embedder import DocumentEmbedder
from src.retrieval.vector_store import ChromaVectorStore

embedder = DocumentEmbedder()
store = ChromaVectorStore()
store.load("data/chroma_db")

query = "What is the annual leave policy?"
query_vec = embedder.embed_query(query)
results = store.search(query_vec, top_k=3)

for r in results:
    print(f"[{r['source']} | Page {r['page']} | Score: {r['similarity_score']:.3f}]")
    print(r['content'][:200])
    print("---")
```

---

## ✅ Checklist
- [ ] `DocumentEmbedder` generates normalized embeddings.
- [ ] `ChromaVectorStore` indexes and retrieves chunks correctly.
- [ ] Vector store implicitly saved to `data/chroma_db/`.
- [ ] Retrieval test returns relevant chunks for sample queries.

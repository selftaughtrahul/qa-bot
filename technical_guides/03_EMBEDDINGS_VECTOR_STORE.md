# Technical Guide 03: Embeddings & Vector Store

## 📋 Overview
This guide covers generating dense vector embeddings from document chunks and storing/indexing them in a FAISS vector store for fast semantic retrieval.

---

## Concept: Why Embeddings?

Traditional keyword search (BM25, TF-IDF) matches exact words. Embeddings capture **semantic meaning** — so "annual leave" and "vacation days" are understood as similar concepts.

```
Text Chunk → Embedding Model → Dense Vector [0.12, -0.45, 0.78, ...]
                                                    ↓
                                          Stored in FAISS Index
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

## Step 2: FAISS Vector Store (`src/retrieval/vector_store.py`)

```python
import faiss
import numpy as np
import json
import os
from typing import List, Dict, Tuple
from embedder import DocumentEmbedder


class FAISSVectorStore:
    """FAISS-based vector store for document chunk retrieval."""

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        # Inner Product index (works with normalized vectors = cosine similarity)
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunks: List[Dict] = []  # Stores metadata alongside vectors

    def add_chunks(self, chunks: List[Dict], embedder: DocumentEmbedder):
        """Embed chunks and add them to the FAISS index."""
        texts = [chunk["content"] for chunk in chunks]
        print(f"🔢 Generating embeddings for {len(texts)} chunks...")
        embeddings = embedder.embed_texts(texts)

        # Add to FAISS index
        self.index.add(embeddings.astype(np.float32))
        self.chunks.extend(chunks)
        print(f"✅ Added {len(chunks)} chunks to vector store. Total: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> List[Dict]:
        """Retrieve top-k most similar chunks for a query embedding."""
        query_vec = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                chunk = self.chunks[idx].copy()
                chunk["similarity_score"] = float(score)
                results.append(chunk)
        return results

    def save(self, save_dir: str = "data/vector_store"):
        """Persist the FAISS index and chunk metadata to disk."""
        os.makedirs(save_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(save_dir, "index.faiss"))
        with open(os.path.join(save_dir, "chunks.json"), "w") as f:
            json.dump(self.chunks, f, indent=2)
        print(f"💾 Vector store saved to: {save_dir}")

    def load(self, save_dir: str = "data/vector_store"):
        """Load a persisted FAISS index and chunk metadata."""
        self.index = faiss.read_index(os.path.join(save_dir, "index.faiss"))
        with open(os.path.join(save_dir, "chunks.json"), "r") as f:
            self.chunks = json.load(f)
        print(f"✅ Loaded vector store: {self.index.ntotal} vectors.")
```

---

## Step 3: Build the Vector Store (`scripts/build_vector_store.py`)

```python
import json
import glob
from src.retrieval.embedder import DocumentEmbedder
from src.retrieval.vector_store import FAISSVectorStore

# Load all processed chunks
all_chunks = []
for chunk_file in glob.glob("data/processed/*.json"):
    with open(chunk_file, "r") as f:
        all_chunks.extend(json.load(f))

print(f"📦 Total chunks to index: {len(all_chunks)}")

# Initialize embedder and vector store
embedder = DocumentEmbedder(model_name="all-MiniLM-L6-v2")
store = FAISSVectorStore(embedding_dim=384)

# Build index
store.add_chunks(all_chunks, embedder)

# Save to disk
store.save("data/vector_store")
```

```bash
python scripts/build_vector_store.py
```

---

## Step 4: Test Retrieval

```python
# Quick retrieval test
embedder = DocumentEmbedder()
store = FAISSVectorStore()
store.load("data/vector_store")

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
- [ ] `FAISSVectorStore` indexes and retrieves chunks correctly.
- [ ] Vector store saved to `data/vector_store/`.
- [ ] Retrieval test returns relevant chunks for sample queries.

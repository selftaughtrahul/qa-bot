import faiss
import numpy as np
import json
import os
from typing import List, Dict, Tuple
from src.retrieval.embedder import DocumentEmbedder



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
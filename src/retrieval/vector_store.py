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
            # ChromaDB metadatas can only be strings, ints, floats or bools
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
            # Cosine similarity is 1 - distance for hnsw:space cosine
            chunk["similarity_score"] = float(1.0 - dist)
            chunks_res.append(chunk)

        return chunks_res

    def save(self, save_dir: str = None):
        """ChromaDB automatically persists data to disk."""
        print("💾 Vector store implicitly saved (ChromaDB is persistent).")

    def load(self, save_dir: str = None):
        """Data is automatically loaded by PersistentClient."""
        print(f"✅ Loaded vector store. Total: {self.collection.count()} vectors.")
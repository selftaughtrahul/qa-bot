import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import glob
from src.retrieval.embedder import DocumentEmbedder
from src.retrieval.vector_store import FAISSVectorStore

# Load all processed chunks
all_chunks = []
for chunk_file in glob.glob("data/processed/*.json"):
    with open(chunk_file, "r", encoding="utf-8") as f:
        all_chunks.extend(json.load(f))

print(f"📦 Total chunks to index: {len(all_chunks)}")

# Initialize embedder and vector store
embedder = DocumentEmbedder(model_name="all-MiniLM-L6-v2")
store = FAISSVectorStore(embedding_dim=384)

# Build index
store.add_chunks(all_chunks, embedder)

# Save to disk
store.save("data/vector_store")
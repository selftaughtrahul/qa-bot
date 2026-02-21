import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.embedder import DocumentEmbedder
from src.retrieval.vector_store import FAISSVectorStore
from src.generation.rag_chain import RAGChain

# Load components
embedder = DocumentEmbedder()
store = FAISSVectorStore()
store.load("data/vector_store")

# Build RAG chain
rag = RAGChain(vector_store=store, embedder=embedder, llm_provider="groq")

# Ask a question
question = "What is the annual leave policy?"
answer, sources = rag.answer(question)

print(f"Q: {question}")
print(f"A: {answer}")
print("\n📄 Sources:")
for s in sources:
    print(f"  - {s['source']} | Page {s['page']} | Score: {s['similarity_score']:.3f}")
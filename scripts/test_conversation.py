import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.embedder import DocumentEmbedder
from src.retrieval.vector_store import FAISSVectorStore
from src.generation.conversational_rag_chain import ConversationalRAGChain

embedder = DocumentEmbedder()
store = FAISSVectorStore()
store.load("data/vector_store")

chain = ConversationalRAGChain(store, embedder, llm_provider="groq")

# Simulate a multi-turn conversation
questions = [
    "What is the annual leave policy?",
    "How many days are allowed?",           # Follow-up — "days" refers to leave
    "How do I apply for it?",               # Follow-up — "it" refers to leave
    "What happens if I don't use them all?" # Follow-up — "them" refers to leave days
]

for q in questions:
    print(f"\n👤 User: {q}")
    answer, sources = chain.answer(q)
    print(f"🤖 Bot: {answer}")
    print(f"   📄 Sources: {[s['source'] for s in sources]}")
from typing import List, Dict, Tuple
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.retrieval.embedder import DocumentEmbedder
from src.retrieval.vector_store import ChromaVectorStore
from src.generation.prompt_template import QA_PROMPT
from src.generation.llm_setup import get_llm


class RAGChain:
    """
    Full RAG pipeline: embed query → retrieve chunks → generate answer.
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedder: DocumentEmbedder,
        llm_provider: str = "groq",
        top_k: int = 4
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm = get_llm(provider=llm_provider)
        self.top_k = top_k
        self.output_parser = StrOutputParser()

    def retrieve(self, question: str) -> List[Dict]:
        """Retrieve top-k relevant chunks for the question."""
        query_vec = self.embedder.embed_query(question)
        return self.vector_store.search(query_vec, top_k=self.top_k)

    def format_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks into a single context string."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[Source {i}: {chunk['source']}, Page {chunk['page']}]\n{chunk['content']}"
            )
        return "\n\n".join(parts)

    def answer(self, question: str) -> Tuple[str, List[Dict]]:
        """
        Full RAG pipeline.
        Returns: (answer_text, source_chunks)
        """
        # Step 1: Retrieve relevant chunks
        chunks = self.retrieve(question)

        # Step 2: Format context
        context = self.format_context(chunks)

        # Step 3: Build prompt and invoke LLM
        prompt = QA_PROMPT.format(context=context, question=question)
        response = self.llm.invoke(prompt)

        # Extract text from response
        answer_text = response.content if hasattr(response, "content") else str(response)

        return answer_text.strip(), chunks
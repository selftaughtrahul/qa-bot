from typing import Tuple, List, Dict
from src.retrieval.embedder import DocumentEmbedder
from src.retrieval.vector_store import ChromaVectorStore
from src.generation.llm_setup import get_llm
from src.generation.prompt_template import CONVERSATIONAL_QA_PROMPT
from src.generation.memory import ConversationMemory


class ConversationalRAGChain:
    """RAG chain with multi-turn conversational memory."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedder: DocumentEmbedder,
        llm_provider: str = "groq",
        top_k: int = 4,
        max_history_turns: int = 5
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm = get_llm(provider=llm_provider)
        self.top_k = top_k
        self.memory = ConversationMemory(max_turns=max_history_turns)

    def answer(self, question: str) -> Tuple[str, List[Dict]]:
        """
        Answer a question with conversational context.
        Returns: (answer_text, source_chunks)
        """
        # Step 1: Retrieve relevant chunks
        query_vec = self.embedder.embed_query(question)
        chunks = self.vector_store.search(query_vec, top_k=self.top_k)

        # Step 2: Format context and history
        context = self._format_context(chunks)
        chat_history = self.memory.format_history()

        # Step 3: Build prompt and invoke LLM
        prompt = CONVERSATIONAL_QA_PROMPT.format(
            context=context,
            chat_history=chat_history,
            question=question
        )
        response = self.llm.invoke(prompt)
        answer_text = response.content if hasattr(response, "content") else str(response)
        answer_text = answer_text.strip()

        # Step 4: Update memory
        self.memory.add_user_message(question)
        self.memory.add_assistant_message(answer_text)

        return answer_text, chunks

    def _format_context(self, chunks: List[Dict]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[Source {i}: {chunk['source']}, Page {chunk['page']}]\n{chunk['content']}"
            )
        return "\n\n".join(parts)

    def clear_memory(self):
        """Reset conversation history."""
        self.memory.clear()
        print("🗑️ Conversation memory cleared.")
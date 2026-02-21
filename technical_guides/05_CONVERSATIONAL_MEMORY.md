# Technical Guide 05: Conversational Memory & Multi-Turn Q&A

## 📋 Overview
This guide covers adding **conversational memory** to the QA Bot so it can handle follow-up questions that reference previous turns — making it feel like a real conversation rather than isolated Q&A.

---

## Concept: Why Memory Matters

Without memory:
```
User: "What is the leave policy?"
Bot:  "Employees get 20 days of annual leave."
User: "How do I apply for it?"
Bot:  ❌ "I don't know what 'it' refers to." (loses context)
```

With memory:
```
User: "What is the leave policy?"
Bot:  "Employees get 20 days of annual leave."
User: "How do I apply for it?"
Bot:  ✅ "To apply for annual leave, submit a request via the HR portal..." (resolves "it")
```

---

## Step 1: Conversation History Manager (`src/generation/memory.py`)

```python
from typing import List, Dict
from dataclasses import dataclass, field


@dataclass
class ConversationTurn:
    role: str    # "user" or "assistant"
    content: str


class ConversationMemory:
    """Manages multi-turn conversation history."""

    def __init__(self, max_turns: int = 10):
        self.history: List[ConversationTurn] = []
        self.max_turns = max_turns  # Limit history to avoid token overflow

    def add_user_message(self, message: str):
        self.history.append(ConversationTurn(role="user", content=message))
        self._trim()

    def add_assistant_message(self, message: str):
        self.history.append(ConversationTurn(role="assistant", content=message))
        self._trim()

    def _trim(self):
        """Keep only the last N turns to stay within LLM context limits."""
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]

    def format_history(self) -> str:
        """Format history as a readable string for the prompt."""
        if not self.history:
            return "No previous conversation."
        lines = []
        for turn in self.history:
            prefix = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{prefix}: {turn.content}")
        return "\n".join(lines)

    def clear(self):
        """Reset conversation history."""
        self.history = []

    def __len__(self):
        return len(self.history)
```

---

## Step 2: Conversational Prompt Template (`src/generation/prompt_template.py`)

Update the prompt to include conversation history:

```python
from langchain.prompts import PromptTemplate

CONVERSATIONAL_QA_PROMPT_TEMPLATE = """You are a helpful and precise assistant.
Answer the user's question ONLY based on the provided context.
Use the conversation history to understand follow-up questions and resolve references like "it", "that", "the above", etc.
If the answer cannot be found in the context, say: "I don't know based on the provided documents."

Context (from documents):
{context}

Conversation History:
{chat_history}

Current Question: {question}

Answer:"""

CONVERSATIONAL_QA_PROMPT = PromptTemplate(
    template=CONVERSATIONAL_QA_PROMPT_TEMPLATE,
    input_variables=["context", "chat_history", "question"]
)
```

---

## Step 3: Update RAG Chain with Memory (`src/generation/conversational_rag_chain.py`)

```python
from typing import Tuple, List, Dict
from src.retrieval.embedder import DocumentEmbedder
from src.retrieval.vector_store import FAISSVectorStore
from src.generation.llm_setup import get_llm
from src.generation.prompt_template import CONVERSATIONAL_QA_PROMPT
from src.generation.memory import ConversationMemory


class ConversationalRAGChain:
    """RAG chain with multi-turn conversational memory."""

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedder: DocumentEmbedder,
        llm_provider: str = "openai",
        top_k: int = 4,
        max_history_turns: int = 10
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
```

---

## Step 4: Test Multi-Turn Conversation

```python
# scripts/test_conversation.py
from src.retrieval.embedder import DocumentEmbedder
from src.retrieval.vector_store import FAISSVectorStore
from src.generation.conversational_rag_chain import ConversationalRAGChain

embedder = DocumentEmbedder()
store = FAISSVectorStore()
store.load("data/vector_store")

chain = ConversationalRAGChain(store, embedder, llm_provider="openai")

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
```

---

## ✅ Checklist
- [ ] `ConversationMemory` stores and formats history correctly.
- [ ] Prompt includes `{chat_history}` variable.
- [ ] `ConversationalRAGChain` updates memory after each turn.
- [ ] Follow-up questions resolve correctly (test with "it", "that", "above").
- [ ] `clear_memory()` resets history for new sessions.

# Technical Guide 04: RAG Chain with LangChain

## 📋 Overview
This guide covers building the **Retrieval-Augmented Generation (RAG) chain** using LangChain — the core brain of the QA Bot that combines retrieved context with an LLM to generate grounded answers.

---

## Concept: How RAG Works

```
User Question
     ↓
[Embed Question] → Dense Vector
     ↓
[Vector Store Search] → Top-K Relevant Chunks
     ↓
[Prompt Builder] → System Prompt + Context + Question
     ↓
[LLM] → Fluent, Grounded Answer
     ↓
[Return Answer + Source Citations]
```

---

## Step 1: Prompt Template (`src/generation/prompt_template.py`)

```python
from langchain.prompts import PromptTemplate

QA_PROMPT_TEMPLATE = """You are a helpful and precise assistant. Answer the user's question
ONLY based on the context provided below. If the answer cannot be found in the context,
respond with: "I don't know based on the provided documents."

Do NOT make up information. Be concise and factual.

Context:
{context}

Question: {question}

Answer:"""

QA_PROMPT = PromptTemplate(
    template=QA_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)
```

---

## Step 2: LLM Setup (`src/generation/llm_setup.py`)

```python
import os
from dotenv import load_dotenv

load_dotenv()


def get_llm(provider: str = "openai"):
    """
    Returns the configured LLM.
    Supports: 'openai', 'huggingface'
    """
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,           # 0 = deterministic, factual answers
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

    elif provider == "huggingface":
        from langchain_community.llms import HuggingFacePipeline
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
        import torch

        model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
        return HuggingFacePipeline(pipeline=pipe)

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
```

> 💡 **Tip:** Use `provider="openai"` for best quality. Use `provider="huggingface"` for fully local, offline operation.

---

## Step 3: RAG Chain (`src/generation/rag_chain.py`)

```python
from typing import List, Dict, Tuple
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

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
        llm_provider: str = "openai",
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
```

---

## Step 4: Test the RAG Chain

```python
# scripts/test_rag.py
from src.retrieval.embedder import DocumentEmbedder
from src.retrieval.vector_store import ChromaVectorStore
from src.generation.rag_chain import RAGChain

# Load components
embedder = DocumentEmbedder()
store = ChromaVectorStore()
store.load("data/vector_store")

# Build RAG chain
rag = RAGChain(vector_store=store, embedder=embedder, llm_provider="openai")

# Ask a question
question = "What is the annual leave policy?"
answer, sources = rag.answer(question)

print(f"Q: {question}")
print(f"A: {answer}")
print("\n📄 Sources:")
for s in sources:
    print(f"  - {s['source']} | Page {s['page']} | Score: {s['similarity_score']:.3f}")
```

```bash
python scripts/test_rag.py
```

**Expected Output:**
```
Q: What is the annual leave policy?
A: Employees are entitled to 20 days of paid annual leave per year...

📄 Sources:
  - policy.pdf | Page 5 | Score: 0.912
  - policy.pdf | Page 6 | Score: 0.887
```

---

## ✅ Checklist
- [ ] Prompt template created with clear instructions.
- [ ] LLM configured (OpenAI or HuggingFace).
- [ ] `RAGChain.answer()` returns answer + source citations.
- [ ] Tested with at least 5 sample questions.
- [ ] Answers are grounded in context (no hallucinations).

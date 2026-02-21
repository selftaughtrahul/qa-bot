# Technical Guide 08: Evaluation with RAGAS

## 📋 Overview
This guide covers **evaluating the QA Bot's quality** using the **RAGAS** (Retrieval-Augmented Generation Assessment) framework — measuring faithfulness, answer relevance, context recall, and context precision.

---

## Concept: Why Evaluate RAG Systems?

RAG systems can fail in multiple ways:
- **Hallucination** — LLM generates facts not in the retrieved context.
- **Poor Retrieval** — Wrong chunks retrieved, so the answer is off-topic.
- **Incomplete Answers** — Relevant chunks retrieved but answer is too vague.
- **Irrelevant Retrieval** — Retrieved chunks are noisy and unrelated.

RAGAS provides automated metrics to catch all of these.

---

## RAGAS Metrics Explained

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **Faithfulness** | Is the answer grounded in the retrieved context? (No hallucinations) | ≥ 0.90 |
| **Answer Relevance** | Does the answer actually address the question? | ≥ 0.80 |
| **Context Recall** | Were all relevant facts retrieved? (Requires ground truth) | ≥ 0.85 |
| **Context Precision** | Are retrieved chunks relevant? (No noisy chunks) | ≥ 0.80 |

---

## Step 1: Create Evaluation Dataset (`data/eval_dataset.json`)

```json
[
  {
    "question": "What is the annual leave entitlement?",
    "ground_truth": "Employees are entitled to 20 days of paid annual leave per year.",
    "document": "policy.pdf"
  },
  {
    "question": "How many sick days are allowed per year?",
    "ground_truth": "Employees can take up to 10 sick days per year with a medical certificate.",
    "document": "policy.pdf"
  },
  {
    "question": "What is the notice period for resignation?",
    "ground_truth": "The notice period is 30 days for employees with less than 2 years of service.",
    "document": "policy.pdf"
  },
  {
    "question": "Can unused leave be carried forward?",
    "ground_truth": "Up to 5 unused leave days can be carried forward to the next calendar year.",
    "document": "policy.pdf"
  },
  {
    "question": "What is the maternity leave policy?",
    "ground_truth": "Female employees are entitled to 26 weeks of paid maternity leave.",
    "document": "policy.pdf"
  }
]
```

---

## Step 2: Run Evaluation (`scripts/evaluate.py`)

```python
import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)

from src.retrieval.embedder import DocumentEmbedder
from src.retrieval.vector_store import ChromaVectorStore
from src.generation.rag_chain import RAGChain


def run_evaluation(eval_dataset_path: str = "data/eval_dataset.json"):
    """Run RAGAS evaluation on the QA Bot."""

    # Load eval dataset
    with open(eval_dataset_path, "r") as f:
        eval_data = json.load(f)

    # Load RAG components
    embedder = DocumentEmbedder()
    store = ChromaVectorStore()
    store.load("data/chroma_db")
    rag = RAGChain(vector_store=store, embedder=embedder, llm_provider="openai")

    # Collect predictions
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print("🔄 Running inference on evaluation set...")
    for item in eval_data:
        q = item["question"]
        gt = item["ground_truth"]

        answer, chunks = rag.answer(q)
        context_texts = [c["content"] for c in chunks]

        questions.append(q)
        answers.append(answer)
        contexts.append(context_texts)
        ground_truths.append(gt)

        print(f"  ✅ Q: {q[:60]}...")

    # Build RAGAS dataset
    ragas_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    # Run evaluation
    print("\n📊 Running RAGAS evaluation...")
    results = evaluate(
        ragas_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision
        ]
    )

    # Display results
    print("\n" + "="*50)
    print("📈 RAGAS Evaluation Results")
    print("="*50)
    df = results.to_pandas()
    print(df[["question", "faithfulness", "answer_relevancy",
              "context_recall", "context_precision"]].to_string(index=False))

    # Summary scores
    print("\n📊 Average Scores:")
    print(f"  Faithfulness:      {df['faithfulness'].mean():.3f}")
    print(f"  Answer Relevancy:  {df['answer_relevancy'].mean():.3f}")
    print(f"  Context Recall:    {df['context_recall'].mean():.3f}")
    print(f"  Context Precision: {df['context_precision'].mean():.3f}")

    # Save report
    df.to_csv("reports/ragas_evaluation.csv", index=False)
    print("\n💾 Report saved to: reports/ragas_evaluation.csv")

    return results


if __name__ == "__main__":
    run_evaluation()
```

---

## Step 3: Run Evaluation

```bash
python scripts/evaluate.py
```

**Expected Output:**
```
🔄 Running inference on evaluation set...
  ✅ Q: What is the annual leave entitlement?...
  ✅ Q: How many sick days are allowed per year?...
  ...

📊 Running RAGAS evaluation...

==================================================
📈 RAGAS Evaluation Results
==================================================
question                              faithfulness  answer_relevancy  context_recall  context_precision
What is the annual leave entitlement?    0.95          0.92              0.88            0.91
How many sick days are allowed?          0.91          0.89              0.85            0.87
...

📊 Average Scores:
  Faithfulness:      0.93
  Answer Relevancy:  0.90
  Context Recall:    0.86
  Context Precision: 0.89

💾 Report saved to: reports/ragas_evaluation.csv
```

---

## Step 4: Interpret & Improve Results

| Score | Interpretation | Action |
|-------|---------------|--------|
| Faithfulness < 0.80 | LLM is hallucinating | Tighten prompt: "ONLY use context" |
| Answer Relevancy < 0.75 | Answers are off-topic | Check prompt clarity |
| Context Recall < 0.80 | Wrong chunks retrieved | Increase `top_k`, improve chunking |
| Context Precision < 0.75 | Too much noise retrieved | Decrease `top_k`, improve embeddings |

---

## Step 5: Optimization Strategies

```python
# 1. Increase chunk overlap for better boundary handling
chunk_documents(pages, chunk_size=500, chunk_overlap=100)  # was 50

# 2. Use a better embedding model
embedder = DocumentEmbedder(model_name="all-mpnet-base-v2")  # was MiniLM

# 3. Increase top_k for better recall
rag = RAGChain(..., top_k=6)  # was 4

# 4. Use GPT-4 for higher faithfulness
llm = get_llm(provider="openai")  # use gpt-4 in llm_setup.py
```

---

## ✅ Checklist
- [ ] Evaluation dataset created with at least 10 Q&A pairs.
- [ ] `evaluate.py` runs without errors.
- [ ] All 4 RAGAS metrics computed and logged.
- [ ] Results saved to `reports/ragas_evaluation.csv`.
- [ ] Faithfulness ≥ 0.90 achieved.
- [ ] Context Recall ≥ 0.85 achieved.
- [ ] Optimization strategies applied if scores are below target.

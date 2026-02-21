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
    rag = RAGChain(vector_store=store, embedder=embedder, llm_provider="groq")

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
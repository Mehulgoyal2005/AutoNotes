"""RAG evaluation harness.

Runs every question in eval/testset.json through the LIVE pipeline
(VectorStore retrieval -> RAGService generation), then scores the results
with RAGAS using Groq as the judge LLM and the project's own local
sentence-transformers model for embeddings.

Usage (from the project root):
    .venv/bin/python eval/run_eval.py                          # -> eval/results_baseline.csv
    .venv/bin/python eval/run_eval.py --out results_hybrid_rerank.csv
"""

import argparse
import json
import os
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

# Run with backend/ as cwd so relative paths in .env (uploads, vector_db, ...)
# resolve exactly as they do for the live uvicorn server.
os.chdir(BACKEND_DIR)
sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings  # noqa: E402
from app.services.vector_store import VectorStore  # noqa: E402
from app.services.rag_service import RAGService  # noqa: E402


def run_pipeline(testset: list, top_k: int) -> list:
    """Push each test question through the existing retrieval + generation pipeline."""
    vector_store = VectorStore(
        embedding_model=settings.embedding_model,
        vector_db_dir=settings.vector_db_dir,
    )
    rag_service = RAGService(
        vector_store=vector_store,
        groq_api_key=settings.groq_api_key,
    )

    rows = []
    for i, case in enumerate(testset, 1):
        doc_id = case["source_doc"]
        if not vector_store.document_exists(doc_id):
            print(f"⚠️  [{i}/{len(testset)}] source_doc {doc_id} not found in vector_db — skipping")
            continue

        print(f"▶️  [{i}/{len(testset)}] {case['question'][:70]}")
        answer, contexts = rag_service.generate_answer(
            document_id=doc_id,
            question=case["question"],
            top_k=top_k,
        )
        rows.append({
            "user_input": case["question"],
            "retrieved_contexts": contexts,
            "response": answer,
            "reference": case["ground_truth"],
        })
    return rows


def score_with_ragas(rows: list):
    """Score captured pipeline outputs with RAGAS, using Groq as judge."""
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    )
    from ragas.run_config import RunConfig

    judge_llm = LangchainLLMWrapper(ChatGroq(
        groq_api_key=settings.groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0,
    ))
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=settings.embedding_model)
    )

    dataset = EvaluationDataset.from_list(rows)
    # strictness=1: Groq's API rejects n>1 completions per call, which
    # AnswerRelevancy's default (strictness=3) relies on
    metrics = [Faithfulness(), AnswerRelevancy(strictness=1), ContextPrecision(), ContextRecall()]

    # max_workers=1 keeps us inside Groq free-tier rate limits
    return evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=1, max_retries=10, timeout=180),
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline with RAGAS")
    parser.add_argument("--testset", default=str(EVAL_DIR / "testset.json"))
    parser.add_argument("--out", default="results_baseline.csv",
                        help="Output CSV filename (written into eval/)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Chunks to retrieve per question (matches the live chat route)")
    args = parser.parse_args()

    with open(args.testset) as f:
        testset = json.load(f)

    print(f"📋 Loaded {len(testset)} test cases\n--- Running pipeline ---")
    rows = run_pipeline(testset, top_k=args.top_k)
    if not rows:
        print("❌ No test cases could be run (missing documents?). Nothing to score.")
        sys.exit(1)

    print("\n--- Scoring with RAGAS (Groq judge) ---")
    result = score_with_ragas(rows)

    df = result.to_pandas()
    score_cols = [c for c in df.columns
                  if c not in ("user_input", "retrieved_contexts", "response", "reference")]

    out_path = EVAL_DIR / args.out
    df.to_csv(out_path, index=False)

    print("\n================ RESULTS ================")
    for _, row in df.iterrows():
        print(f"\nQ: {row['user_input'][:70]}")
        for c in score_cols:
            print(f"   {c}: {row[c]:.4f}")
    print("\n---------------- AVERAGES ----------------")
    for c in score_cols:
        print(f"   {c}: {df[c].mean():.4f}")
    print(f"\n💾 Saved per-question results to {out_path}")


if __name__ == "__main__":
    main()

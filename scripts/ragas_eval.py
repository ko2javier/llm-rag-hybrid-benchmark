"""
RAGAS quality evaluation — reads a CSV produced by evaluator.py or hyde.py
(needs question, generated_answer, ideal_answer, retrieved_context columns)
and scores it with the standard RAGAS metrics: faithfulness, answer_relevancy,
context_precision, context_recall.

Judge LLM must be a THIRD model family — neither Gemma nor Qwen (the two
candidates being compared) — to avoid self-preference bias and keep scores
comparable across both. No paid API is used: the judge and the embeddings
both point at local self-hosted endpoints (vLLM + embed_server_batching.py),
both of which expose an OpenAI-compatible route.

Run AFTER both candidate models have already been evaluated and their CSVs
saved, and after loading the judge model in vLLM (see README_VAST.md step 12):

    python scripts/ragas_eval.py --input results/gemma3_27b_selfhosted.csv \
        --output results/gemma3_27b_ragas.csv \
        --judge-model mistralai/Mistral-7B-Instruct-v0.3

    python scripts/ragas_eval.py --input results/qwen32b_selfhosted.csv \
        --output results/qwen32b_ragas.csv \
        --judge-model mistralai/Mistral-7B-Instruct-v0.3

Note: meta-llama/Llama-3.1-8B-Instruct was the originally planned judge (FASE2_LEY.md
S7) but it's gated on HF and needs manual license acceptance tied to the account, so
Mistral-7B-Instruct-v0.3 (ungated, third model family, not Gemma/Qwen) was substituted.
"""

import argparse
import csv
import json

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

JUDGE_URL       = "http://localhost:8081/v1"  # vLLM serving the judge model
EMBED_URL       = "http://localhost:8083/v1"  # embed_server_batching.py OpenAI-compatible route
EMBEDDING_MODEL = "BAAI/bge-m3"


def load_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_dataset(rows: list[dict]) -> Dataset:
    return Dataset.from_dict({
        "question":     [r["question"] for r in rows],
        "answer":       [r["generated_answer"] for r in rows],
        "contexts":     [json.loads(r["retrieved_context"]) if r.get("retrieved_context") else [] for r in rows],
        "ground_truth": [r["ideal_answer"] for r in rows],
    })


def parse_args():
    parser = argparse.ArgumentParser(description="Evalua un CSV de evaluator.py/hyde.py con metricas RAGAS")
    parser.add_argument("--input", required=True, help="CSV de entrada (salida de evaluator.py o hyde.py)")
    parser.add_argument("--output", required=True, help="CSV de salida con columnas de score RAGAS por fila")
    parser.add_argument("--judge-model", required=True,
                         help="Nombre/ID del modelo juez servido por vLLM (tercera familia, ni Gemma ni Qwen)")
    parser.add_argument("--judge-url", default=JUDGE_URL, help="Base URL OpenAI-compatible del vLLM del juez")
    parser.add_argument("--embed-url", default=EMBED_URL, help="Base URL OpenAI-compatible del embedding server")
    return parser.parse_args()


def main():
    args = parse_args()

    rows = load_rows(args.input)
    dataset = build_dataset(rows)

    judge_llm = ChatOpenAI(
        model=args.judge_model,
        base_url=args.judge_url,
        api_key="not-needed",  # local vLLM, no real key required
        temperature=0.0,
    )
    judge_embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=args.embed_url,
        api_key="not-needed",  # local embed server, no real key required
        tiktoken_enabled=False,  # embed_server_batching.py takes raw strings, not pre-tokenized IDs
    )

    print(f"Scoring {len(rows)} rows from {args.input} with judge={args.judge_model} …")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    scored_df = result.to_pandas()
    scored_df.to_csv(args.output, index=False)

    print(f"\nResults saved → {args.output}")
    print(result)


if __name__ == "__main__":
    main()

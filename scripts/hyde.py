"""
HyDE (Hypothetical Document Embeddings) evaluation benchmark — single model.

Differs from evaluator.py in retrieval:
  Standard:  embed(question)       → cosine similarity → top-k chunks
  HyDE:      LLM(question) → hypothetical answer → embed(hyp_answer)
             → cosine similarity → top-k chunks → LLM(final answer)

Uses ONE model for both steps (hypothetical answer + final answer) — this
phase runs one self-hosted model at a time on a single ~20GB GPU, so no
second model is assumed to be running concurrently. No cross-judging here;
quality evaluation is handled separately (RAGAS), not by LLM-as-judge.

Run from the project root:
    python scripts/hyde.py --model Qwen/Qwen2.5-32B-Instruct-AWQ \
        --output results/qwen_hyde.csv --gpu-hourly-rate 0.35
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import requests

# ── Configuration (defaults, overridable via CLI) ────────────────────────────

MODEL_URL   = "http://localhost:8081/v1"
MODEL_NAME  = "google/gemma-3-27b-it"

EMBEDDING_URL   = "http://localhost:8083/embed"  # TEI endpoint
EMBEDDING_MODEL = "BAAI/bge-m3"

CHUNKS_FILE    = "output/chunks.json"
GOLDEN_DATASET = "dataset/golden_dataset.json"
OUTPUT_FILE    = "output/hyde_results.csv"

TOP_K        = 5
HTTP_TIMEOUT = 60  # seconds

GPU_HOURLY_RATE_USD    = 0.35
API_INPUT_COST_PER_M   = 0.0
API_OUTPUT_COST_PER_M  = 0.0

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def embed(text: str) -> np.ndarray | None:
    """Return a unit-normalised embedding vector, or None on error."""
    try:
        resp = requests.post(
            EMBEDDING_URL,
            json={"inputs": text},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        vec = data[0] if isinstance(data[0], list) else data
        arr = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr
    except Exception as exc:
        print(f"  [embed error] {exc}")
        return None


def retrieve(query_vec: np.ndarray, chunk_vecs: list,
             chunks: list[dict]) -> list[dict]:
    """Return TOP_K chunks sorted by cosine similarity (descending)."""
    scored = []
    for i, cvec in enumerate(chunk_vecs):
        if cvec is None:
            continue
        score = float(np.dot(query_vec, cvec))
        scored.append((score, i))
    scored.sort(reverse=True)
    results = []
    for score, idx in scored[:TOP_K]:
        results.append({
            "chunk_id":    chunks[idx]["chunk_id"],
            "source_file": chunks[idx]["source_file"],
            "chunk_text":  chunks[idx]["chunk_text"],
            "score":       round(score, 4),
        })
    return results


def build_rag_prompt(question: str, retrieved: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(retrieved, 1):
        context_parts.append(
            f"[{i}] (source: {chunk['source_file']})\n{chunk['chunk_text']}"
        )
    context = "\n\n".join(context_parts) if context_parts else "(no context retrieved)"
    return (
        "You are a helpful assistant. Use only the context below to answer.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def chat_completion(model_url: str, model_name: str, prompt: str) -> tuple[str, float, int, int]:
    """
    Call /v1/chat/completions.
    Returns (response_text, latency_s, prompt_tokens, completion_tokens).
    On error returns ("", elapsed_s, 0, 0).
    """
    payload = {
        "model":       model_name,
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens":  512,
    }
    t0 = time.monotonic()
    try:
        resp = requests.post(
            f"{model_url}/chat/completions",
            json=payload,
            timeout=HTTP_TIMEOUT,
        )
        elapsed = time.monotonic() - t0
        resp.raise_for_status()
        body = resp.json()
        text = body["choices"][0]["message"]["content"].strip()
        usage = body.get("usage", {})
        return text, round(elapsed, 3), usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    except Exception as exc:
        elapsed = time.monotonic() - t0
        print(f"  [chat error] {exc}")
        return "", round(elapsed, 3), 0, 0


def generate_hypothetical_answer(model_url: str, model_name: str, question: str) -> tuple[str, float, int, int]:
    """Ask the model to produce a short hypothetical answer to the question."""
    prompt = (
        "Generate a short hypothetical answer to this question as if you were "
        "a technical documentation expert. Be concise, 2-3 sentences maximum.\n"
        f"Question: {question}"
    )
    return chat_completion(model_url, model_name, prompt)


def compute_costs(latency_s: float, prompt_tokens: int, completion_tokens: int,
                   gpu_hourly_rate: float, api_input_cost_per_m: float,
                   api_output_cost_per_m: float) -> tuple[float, float]:
    cost_per_query_usd = round((latency_s / 3600.0) * gpu_hourly_rate, 8)
    equivalent_api_cost_usd = round(
        (prompt_tokens * api_input_cost_per_m / 1_000_000)
        + (completion_tokens * api_output_cost_per_m / 1_000_000),
        8,
    )
    return cost_per_query_usd, equivalent_api_cost_usd


def embed_all_chunks(chunks: list[dict]) -> list:
    print(f"Embedding {len(chunks)} chunks via TEI …")
    vecs = []
    for i, chunk in enumerate(chunks):
        vecs.append(embed(chunk["chunk_text"]))
        if (i + 1) % 20 == 0 or (i + 1) == len(chunks):
            print(f"  {i + 1}/{len(chunks)} chunks embedded")
    return vecs


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="HyDE evaluator (single model) con coste")
    parser.add_argument("--model", default=MODEL_NAME, help="Nombre/ID del modelo servido por vLLM")
    parser.add_argument("--model-url", default=MODEL_URL, help="Base URL OpenAI-compatible de vLLM")
    parser.add_argument("--output", default=OUTPUT_FILE, help="Ruta del CSV de salida")
    parser.add_argument("--gpu-hourly-rate", type=float, default=GPU_HOURLY_RATE_USD)
    parser.add_argument("--api-input-cost-per-m", type=float, default=API_INPUT_COST_PER_M)
    parser.add_argument("--api-output-cost-per-m", type=float, default=API_OUTPUT_COST_PER_M)
    parser.add_argument("--only-type", choices=["all", "deterministic", "semantic"], default="all",
                         help="Filtra el dataset a un solo tipo de pregunta antes de correr "
                              "(para re-correr solo un subconjunto sin gastar GPU en el resto).")
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading data …")
    chunks  = load_json(CHUNKS_FILE)
    dataset = load_json(GOLDEN_DATASET)
    if args.only_type != "all":
        dataset = [item for item in dataset if item.get("type") == args.only_type]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    chunk_vecs = embed_all_chunks(chunks)

    total = len(dataset)
    print(f"\nRunning HyDE pipeline on {total} questions with model={args.model} …\n")

    CSV_COLUMNS = [
        "id", "type", "source", "question", "ideal_answer",
        "hypothetical_answer", "generated_answer", "retrieved_context",
        "hyde_latency_s", "answer_latency_s", "total_latency_s",
        "prompt_tokens", "completion_tokens", "tokens_used",
        "embedding_model", "llm_model",
        "cost_per_query_usd", "gpu_hourly_rate_usd", "equivalent_api_cost_usd",
    ]

    with open(args.output, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for idx, item in enumerate(dataset, 1):
            qid      = item["id"]
            question = item["question"]
            expected = item.get("expected_answer", "")
            qtype    = item.get("type", "unknown")

            print(f"[{idx}/{total}] {qid}: {question[:80]}")

            # Step 1: hypothetical answer (same model)
            hyp_answer, hyde_latency_s, hyp_prompt_tok, hyp_completion_tok = \
                generate_hypothetical_answer(args.model_url, args.model, question)

            # Step 2: embed the hypothetical answer (fallback to question on failure)
            hyp_vec = embed(hyp_answer) if hyp_answer else embed(question)

            # Step 3: retrieve using the hypothetical-answer vector
            retrieved = retrieve(hyp_vec, chunk_vecs, chunks) if hyp_vec is not None else []
            sources = "; ".join({c["source_file"] for c in retrieved})
            retrieved_context = json.dumps([c["chunk_text"] for c in retrieved], ensure_ascii=False)

            # Step 4: final answer with retrieved context
            prompt = build_rag_prompt(question, retrieved)
            answer, answer_latency_s, ans_prompt_tok, ans_completion_tok = \
                chat_completion(args.model_url, args.model, prompt)

            total_latency_s   = round(hyde_latency_s + answer_latency_s, 3)
            prompt_tokens     = hyp_prompt_tok + ans_prompt_tok
            completion_tokens = hyp_completion_tok + ans_completion_tok
            tokens_used       = prompt_tokens + completion_tokens

            cost_per_query_usd, equivalent_api_cost_usd = compute_costs(
                total_latency_s, prompt_tokens, completion_tokens,
                args.gpu_hourly_rate, args.api_input_cost_per_m, args.api_output_cost_per_m,
            )

            print(f"  HyDE {hyde_latency_s:.2f}s + answer {answer_latency_s:.2f}s "
                  f"= {total_latency_s:.2f}s  ${cost_per_query_usd:.6f}/query — {answer[:60]!r}")

            writer.writerow({
                "id":                      qid,
                "type":                    qtype,
                "source":                  sources,
                "question":                question,
                "ideal_answer":            expected,
                "hypothetical_answer":     hyp_answer,
                "generated_answer":        answer,
                "retrieved_context":       retrieved_context,
                "hyde_latency_s":          hyde_latency_s,
                "answer_latency_s":        answer_latency_s,
                "total_latency_s":         total_latency_s,
                "prompt_tokens":           prompt_tokens,
                "completion_tokens":       completion_tokens,
                "tokens_used":             tokens_used,
                "embedding_model":         EMBEDDING_MODEL,
                "llm_model":               args.model,
                "cost_per_query_usd":      cost_per_query_usd,
                "gpu_hourly_rate_usd":     args.gpu_hourly_rate,
                "equivalent_api_cost_usd": equivalent_api_cost_usd,
            })

    print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()

"""
RAG evaluation benchmark — runs all questions in golden_dataset.json against
a single vLLM instance with cosine-similarity retrieval over chunks.json.
Saves results to CSV (no judge), including self-hosted vs API cost columns.

Optionally routes deterministic questions to an exact lookup in the
`api_facts` Postgres table (via router.classify()) instead of semantic
retrieval — see --use-router (Exp D).

Run from the project root:
    python scripts/evaluator.py --model Qwen/Qwen2.5-32B-Instruct-AWQ \
        --output results/qwen_selfhosted.csv --gpu-hourly-rate 0.35

    # Exp D — router-backed lookup, deterministic questions only:
    python scripts/evaluator.py --model Qwen/Qwen2.5-32B-Instruct-AWQ \
        --output results/qwen_router.csv --use-router --only-type deterministic
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(__file__))
from router import classify as router_classify

# ── Configuration (defaults, overridable via CLI) ────────────────────────────

MODEL_URL   = "http://localhost:8081/v1"
MODEL_NAME  = "google/gemma-3-27b-it"

EMBEDDING_URL   = "http://localhost:8083/embed"  # TEI endpoint
EMBEDDING_MODEL = "BAAI/bge-m3"

CHUNKS_FILE    = "output/chunks.json"
GOLDEN_DATASET = "dataset/golden_dataset.json"
OUTPUT_FILE    = "output/eval_results.csv"

TOP_K           = 5    # chunks to retrieve per question
MIN_CHUNK_SCORE = 0.0  # cosine similarity threshold

HTTP_TIMEOUT = 60  # seconds for all outbound HTTP calls

GPU_HOURLY_RATE_USD    = 0.35  # $/hora, Vast.ai RTX 4000 Ada de referencia
API_INPUT_COST_PER_M   = 0.0   # $/millon tokens input del modelo API de referencia
API_OUTPUT_COST_PER_M  = 0.0   # $/millon tokens output del modelo API de referencia

# ── api_facts DB config (must match ingest_facts.py) ─────────────────────────

DB_HOST     = "localhost"
DB_PORT     = 5432
DB_NAME     = "nexuspay_rag"
DB_USER     = "postgres"
DB_PASSWORD = "postgres"

# ── Helpers ──────────────────────────────────────────────────────────────────

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
        # TEI returns [[float, ...]] (batch) or [float, ...]
        vec = data[0] if isinstance(data[0], list) else data
        arr = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr
    except Exception as exc:
        print(f"  [embed error] {exc}")
        return None


def retrieve(question_vec: np.ndarray, chunk_vecs: list,
             chunks: list[dict]) -> list[dict]:
    """Return TOP_K chunks above MIN_CHUNK_SCORE, sorted by similarity desc."""
    scored = []
    for i, cvec in enumerate(chunk_vecs):
        if cvec is None:
            continue
        score = float(np.dot(question_vec, cvec))
        if score >= MIN_CHUNK_SCORE:
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


def build_fact_prompt(question: str, fact_value: dict, source_file: str | None) -> str:
    context = f"(source: {source_file or 'api_facts'})\n{json.dumps(fact_value, ensure_ascii=False)}"
    return (
        "You are a helpful assistant. Use only the exact fact below to answer precisely.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def lookup_fact(conn, fact_type: str, keywords: list[str]) -> tuple[dict | None, str | None]:
    """
    Return (value, source_file) for the api_facts row of the given fact_type
    with the largest keyword overlap against `keywords`. (None, None) if no
    row exists for fact_type.
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT value, source_file, keywords FROM api_facts WHERE fact_type = %s",
        (fact_type,),
    )
    rows = cur.fetchall()
    cur.close()
    if not rows:
        return None, None
    kw_set = {k.lower() for k in keywords}
    best_value, best_source, best_overlap = None, None, 0
    for value, source_file, row_keywords in rows:
        overlap = len(kw_set & {k.lower() for k in row_keywords})
        if overlap > best_overlap:
            best_value, best_source, best_overlap = value, source_file, overlap
    if best_overlap == 0:
        # No real keyword match — don't guess a row at random, signal a miss.
        return None, None
    return best_value, best_source


def chat_completion(model_url: str, model_name: str, prompt: str) -> tuple[str, float, int, int]:
    """
    Call the /v1/chat/completions endpoint.
    Returns (response_text, latency_s, prompt_tokens, completion_tokens).
    On error returns ("", elapsed_s, 0, 0).
    """
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 512,
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
        prompt_tokens     = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        return text, round(elapsed, 3), prompt_tokens, completion_tokens
    except Exception as exc:
        elapsed = time.monotonic() - t0
        print(f"  [chat error] {exc}")
        return "", round(elapsed, 3), 0, 0


def compute_costs(latency_s: float, prompt_tokens: int, completion_tokens: int,
                   gpu_hourly_rate: float, api_input_cost_per_m: float,
                   api_output_cost_per_m: float) -> tuple[float, float]:
    """
    Returns (cost_per_query_usd, equivalent_api_cost_usd).
    cost_per_query_usd: coste real de GPU self-hosted por esta query (latencia x $/hora).
    equivalent_api_cost_usd: coste que tendria la misma query en un modelo API de referencia
                             (0.0 si no se pasaron precios de referencia).
    """
    cost_per_query_usd = round((latency_s / 3600.0) * gpu_hourly_rate, 8)
    equivalent_api_cost_usd = round(
        (prompt_tokens * api_input_cost_per_m / 1_000_000)
        + (completion_tokens * api_output_cost_per_m / 1_000_000),
        8,
    )
    return cost_per_query_usd, equivalent_api_cost_usd


# ── Embedding cache ───────────────────────────────────────────────────────────

def embed_all_chunks(chunks: list[dict]) -> list:
    """Embed every chunk, printing progress."""
    print(f"Embedding {len(chunks)} chunks via TEI …")
    vecs = []
    for i, chunk in enumerate(chunks):
        vec = embed(chunk["chunk_text"])
        vecs.append(vec)
        if (i + 1) % 20 == 0 or (i + 1) == len(chunks):
            print(f"  {i + 1}/{len(chunks)} chunks embedded")
    return vecs


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="RAG evaluator con coste self-hosted vs API")
    parser.add_argument("--model", default=MODEL_NAME, help="Nombre/ID del modelo servido por vLLM")
    parser.add_argument("--model-url", default=MODEL_URL, help="Base URL OpenAI-compatible de vLLM")
    parser.add_argument("--output", default=OUTPUT_FILE, help="Ruta del CSV de salida")
    parser.add_argument("--gpu-hourly-rate", type=float, default=GPU_HOURLY_RATE_USD,
                         help="$/hora de la GPU self-hosted (Vast.ai)")
    parser.add_argument("--api-input-cost-per-m", type=float, default=API_INPUT_COST_PER_M,
                         help="$/millon tokens input del modelo API de referencia (0 = no comparar)")
    parser.add_argument("--api-output-cost-per-m", type=float, default=API_OUTPUT_COST_PER_M,
                         help="$/millon tokens output del modelo API de referencia (0 = no comparar)")
    parser.add_argument("--use-router", action="store_true",
                         help="Exp D: enruta preguntas deterministas a lookup exacto en api_facts "
                              "via router.classify() en vez de RAG semantico. Las 'rag'/semanticas "
                              "siguen usando retrieval semantico normal.")
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

    db_conn = None
    if args.use_router:
        import psycopg2
        db_conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
        )

    total = len(dataset)
    print(f"\nRunning inference on {total} questions with model={args.model} "
          f"(router={'on' if args.use_router else 'off'}, only_type={args.only_type}) …\n")

    CSV_COLUMNS = [
        "id", "type", "source", "question", "ideal_answer",
        "generated_answer", "retrieved_context", "retrieval_method",
        "latency_s", "prompt_tokens", "completion_tokens",
        "tokens_used", "embedding_model", "llm_model",
        "cost_per_query_usd", "gpu_hourly_rate_usd", "equivalent_api_cost_usd",
    ]

    try:
        with open(args.output, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
            writer.writeheader()

            for idx, item in enumerate(dataset, 1):
                qid      = item["id"]
                question = item["question"]
                expected = item.get("expected_answer", "")
                qtype    = item.get("type", "unknown")

                print(f"[{idx}/{total}] {qid}: {question[:80]}")

                retrieval_method = "semantic"
                fact_value = None
                fact_source = None

                if args.use_router:
                    route = router_classify(question)
                    if route["type"] == "deterministic":
                        fact_value, fact_source = lookup_fact(
                            db_conn, route["fact_type"], route["keywords"]
                        )
                        if fact_value is not None:
                            retrieval_method = "router_lookup"
                        else:
                            retrieval_method = "router_miss_fallback_semantic"

                if fact_value is not None:
                    sources = fact_source or "api_facts"
                    # RAGAS expects contexts as a list of strings, so the fact dict is
                    # itself JSON-stringified before going into the outer list.
                    retrieved_context = json.dumps(
                        [json.dumps(fact_value, ensure_ascii=False)], ensure_ascii=False
                    )
                    prompt = build_fact_prompt(question, fact_value, fact_source)
                else:
                    q_vec = embed(question)
                    retrieved = retrieve(q_vec, chunk_vecs, chunks) if q_vec is not None else []
                    sources = "; ".join({c["source_file"] for c in retrieved})
                    retrieved_context = json.dumps([c["chunk_text"] for c in retrieved], ensure_ascii=False)
                    prompt = build_rag_prompt(question, retrieved)

                answer, latency_s, prompt_tokens, completion_tokens = chat_completion(
                    args.model_url, args.model, prompt
                )
                tokens_used = prompt_tokens + completion_tokens

                cost_per_query_usd, equivalent_api_cost_usd = compute_costs(
                    latency_s, prompt_tokens, completion_tokens,
                    args.gpu_hourly_rate, args.api_input_cost_per_m, args.api_output_cost_per_m,
                )

                print(f"  [{retrieval_method}] {latency_s:.2f}s  {tokens_used} tok  "
                      f"${cost_per_query_usd:.6f}/query — {answer[:60]!r}")

                writer.writerow({
                    "id":                     qid,
                    "type":                   qtype,
                    "source":                 sources,
                    "question":               question,
                    "ideal_answer":           expected,
                    "generated_answer":       answer,
                    "retrieved_context":      retrieved_context,
                    "retrieval_method":       retrieval_method,
                    "latency_s":              latency_s,
                    "prompt_tokens":          prompt_tokens,
                    "completion_tokens":      completion_tokens,
                    "tokens_used":            tokens_used,
                    "embedding_model":        EMBEDDING_MODEL,
                    "llm_model":              args.model,
                    "cost_per_query_usd":     cost_per_query_usd,
                    "gpu_hourly_rate_usd":    args.gpu_hourly_rate,
                    "equivalent_api_cost_usd": equivalent_api_cost_usd,
                })
    finally:
        if db_conn is not None:
            db_conn.close()

    print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()

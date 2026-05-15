"""
RAG evaluation benchmark — runs all questions in golden_dataset.json against
a single vLLM instance with cosine-similarity retrieval over chunks.json.
Saves results to CSV (no judge).

Run from the project root:
    python scripts/evaluator.py
"""

import csv
import json
import os
import time

import numpy as np
import requests

# ── Configuration ────────────────────────────────────────────────────────────

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


def chat_completion(prompt: str) -> tuple[str, float, int]:
    """
    Call the /v1/chat/completions endpoint.
    Returns (response_text, latency_s, tokens_used). On error returns ("", elapsed_s, 0).
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 512,
    }
    t0 = time.monotonic()
    try:
        resp = requests.post(
            f"{MODEL_URL}/chat/completions",
            json=payload,
            timeout=HTTP_TIMEOUT,
        )
        elapsed = time.monotonic() - t0
        resp.raise_for_status()
        body = resp.json()
        text = body["choices"][0]["message"]["content"].strip()
        tokens = body.get("usage", {}).get("total_tokens", 0)
        return text, round(elapsed, 3), tokens
    except Exception as exc:
        elapsed = time.monotonic() - t0
        print(f"  [chat error] {exc}")
        return "", round(elapsed, 3), 0


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

def main():
    print("Loading data …")
    chunks  = load_json(CHUNKS_FILE)
    dataset = load_json(GOLDEN_DATASET)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    chunk_vecs = embed_all_chunks(chunks)

    total = len(dataset)
    print(f"\nRunning inference on {total} questions …\n")

    CSV_COLUMNS = [
        "id", "type", "source", "question", "ideal_answer",
        "generated_answer", "latency_s", "tokens_used",
        "embedding_model", "llm_model",
    ]

    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for idx, item in enumerate(dataset, 1):
            qid      = item["id"]
            question = item["question"]
            expected = item.get("expected_answer", "")
            qtype    = item.get("type", "unknown")

            print(f"[{idx}/{total}] {qid}: {question[:80]}")

            q_vec = embed(question)
            retrieved = retrieve(q_vec, chunk_vecs, chunks) if q_vec is not None else []

            sources = "; ".join({c["source_file"] for c in retrieved})
            prompt  = build_rag_prompt(question, retrieved)

            answer, latency_s, tokens_used = chat_completion(prompt)
            print(f"  {latency_s:.2f}s  {tokens_used} tok — {answer[:60]!r}")

            writer.writerow({
                "id":               qid,
                "type":             qtype,
                "source":           sources,
                "question":         question,
                "ideal_answer":     expected,
                "generated_answer": answer,
                "latency_s":        latency_s,
                "tokens_used":      tokens_used,
                "embedding_model":  EMBEDDING_MODEL,
                "llm_model":        MODEL_NAME,
            })

    print(f"\nResults saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

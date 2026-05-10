"""
RAG evaluation benchmark — runs all questions in golden_dataset.json against
two vLLM instances (Gemma and Qwen) with cosine-similarity retrieval over
chunks.json, then cross-judges each model's answers.

Run from the project root:
    python scripts/evaluator.py
"""

import json
import os
import time

import numpy as np
import requests

# ── Configuration ────────────────────────────────────────────────────────────

GEMMA_URL   = "http://localhost:8081/v1"
GEMMA_MODEL = "google/gemma-3-27b-it"

QWEN_URL    = "http://localhost:8082/v1"
QWEN_MODEL  = "Qwen/Qwen2.5-32B-Instruct-AWQ"

CHUNKS_FILE    = "output/chunks.json"
GOLDEN_DATASET = "dataset/golden_dataset.json"
OUTPUT_DIR     = "output/"

TOP_K           = 5    # chunks to retrieve per question
MIN_CHUNK_SCORE = 0.0  # cosine similarity threshold

EMBEDDING_URL = "http://localhost:8083/embed"  # TEI endpoint

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


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # Both vectors are already unit-normalised; dot product == cosine sim.
    return float(np.dot(a, b))


def retrieve(question_vec: np.ndarray, chunk_vecs: list[np.ndarray],
             chunks: list[dict]) -> list[dict]:
    """Return TOP_K chunks above MIN_CHUNK_SCORE, sorted by similarity desc."""
    scored = []
    for i, cvec in enumerate(chunk_vecs):
        if cvec is None:
            continue
        score = cosine_similarity(question_vec, cvec)
        if score >= MIN_CHUNK_SCORE:
            scored.append((score, i))
    scored.sort(reverse=True)
    results = []
    for score, idx in scored[:TOP_K]:
        results.append({
            "chunk_id":   chunks[idx]["chunk_id"],
            "source_file": chunks[idx]["source_file"],
            "chunk_text": chunks[idx]["chunk_text"],
            "score":      round(score, 4),
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


def chat_completion(base_url: str, model: str, prompt: str) -> tuple[str, float]:
    """
    Call the /v1/chat/completions endpoint.
    Returns (response_text, latency_ms). On error returns ("", elapsed_ms).
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 512,
    }
    t0 = time.monotonic()
    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            json=payload,
            timeout=HTTP_TIMEOUT,
        )
        elapsed = (time.monotonic() - t0) * 1000
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        return text, round(elapsed, 1)
    except Exception as exc:
        elapsed = (time.monotonic() - t0) * 1000
        print(f"  [chat error {base_url}] {exc}")
        return "", round(elapsed, 1)


def judge_answer(judge_url: str, judge_model: str,
                 question: str, expected: str, response: str) -> int:
    """
    Ask the judge model to score 'response' from 0-10.
    Returns an int, or 0 on parse/request failure.
    """
    prompt = (
        "Score this answer from 0 to 10 based on correctness and completeness.\n"
        f"Question: {question}\n"
        f"Expected answer: {expected}\n"
        f"Given answer: {response}\n"
        "Reply with only a number from 0 to 10."
    )
    text, _ = chat_completion(judge_url, judge_model, prompt)
    # Extract the first integer found in the reply
    for token in text.split():
        token = token.strip(".,")
        if token.isdigit():
            return min(10, max(0, int(token)))
    return 0


# ── Embedding cache ───────────────────────────────────────────────────────────

def embed_all_chunks(chunks: list[dict]) -> list[np.ndarray | None]:
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
    # ── Load data ────────────────────────────────────────────────────────────
    print("Loading data …")
    chunks  = load_json(CHUNKS_FILE)
    dataset = load_json(GOLDEN_DATASET)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Pre-embed all chunks ─────────────────────────────────────────────────
    chunk_vecs = embed_all_chunks(chunks)

    # ── Per-question inference ────────────────────────────────────────────────
    results = []
    total = len(dataset)
    print(f"\nRunning inference on {total} questions …\n")

    for idx, item in enumerate(dataset, 1):
        qid      = item["id"]
        question = item["question"]
        expected = item.get("expected_answer", "")
        qtype    = item.get("type", "unknown")

        print(f"[{idx}/{total}] {qid}: {question[:80]}")

        # Embed question
        q_vec = embed(question)

        # Retrieve context
        if q_vec is not None:
            retrieved = retrieve(q_vec, chunk_vecs, chunks)
        else:
            retrieved = []

        # Build prompt
        prompt = build_rag_prompt(question, retrieved)

        # Gemma
        gemma_resp, gemma_latency = chat_completion(GEMMA_URL, GEMMA_MODEL, prompt)
        print(f"  Gemma  {gemma_latency:.0f}ms — {gemma_resp[:60]!r}")

        # Qwen
        qwen_resp, qwen_latency = chat_completion(QWEN_URL, QWEN_MODEL, prompt)
        print(f"  Qwen   {qwen_latency:.0f}ms — {qwen_resp[:60]!r}")

        results.append({
            "question_id":       qid,
            "question":          question,
            "expected_answer":   expected,
            "type":              qtype,
            "retrieved_chunks":  retrieved,
            "gemma_response":    gemma_resp,
            "qwen_response":     qwen_resp,
            "gemma_latency_ms":  gemma_latency,
            "qwen_latency_ms":   qwen_latency,
            "gemma_score_by_qwen": 0,   # filled in judging phase
            "qwen_score_by_gemma": 0,
        })

    # ── Cross-judge ───────────────────────────────────────────────────────────
    print(f"\nCross-judging {len(results)} answers …\n")

    for idx, r in enumerate(results, 1):
        qid = r["question_id"]
        print(f"[{idx}/{len(results)}] Judging {qid} …")

        # Qwen judges Gemma
        r["gemma_score_by_qwen"] = judge_answer(
            QWEN_URL, QWEN_MODEL,
            r["question"], r["expected_answer"], r["gemma_response"],
        )
        # Gemma judges Qwen
        r["qwen_score_by_gemma"] = judge_answer(
            GEMMA_URL, GEMMA_MODEL,
            r["question"], r["expected_answer"], r["qwen_response"],
        )
        print(f"  gemma_score_by_qwen={r['gemma_score_by_qwen']}  "
              f"qwen_score_by_gemma={r['qwen_score_by_gemma']}")

    # ── Save results ──────────────────────────────────────────────────────────
    output_path = os.path.join(OUTPUT_DIR, "eval_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {output_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    def stats(records, score_key, latency_key):
        scores   = [r[score_key]   for r in records]
        latencies = [r[latency_key] for r in records]
        n = len(scores)
        return {
            "n":           n,
            "avg_score":   round(sum(scores) / n, 2)    if n else 0,
            "avg_latency": round(sum(latencies) / n, 1) if n else 0,
            "pct_zeros":   round(scores.count(0) / n * 100, 1) if n else 0,
        }

    semantic     = [r for r in results if r["type"] == "semantic"]
    deterministic = [r for r in results if r["type"] == "deterministic"]

    print("\n" + "=" * 62)
    print("EVALUATION SUMMARY")
    print("=" * 62)

    for label, subset in [
        ("ALL",           results),
        ("Deterministic", deterministic),
        ("Semantic",      semantic),
    ]:
        gs = stats(subset, "gemma_score_by_qwen", "gemma_latency_ms")
        qs = stats(subset, "qwen_score_by_gemma", "qwen_latency_ms")
        print(f"\n── {label} ({gs['n']} questions) ──")
        print(f"  {'Model':<8}  {'Avg score':>9}  {'Avg lat(ms)':>11}  {'% zeros':>7}")
        print(f"  {'Gemma':<8}  {gs['avg_score']:>9}  {gs['avg_latency']:>11}  {gs['pct_zeros']:>6}%")
        print(f"  {'Qwen':<8}  {qs['avg_score']:>9}  {qs['avg_latency']:>11}  {qs['pct_zeros']:>6}%")

    print("\n" + "=" * 62)


if __name__ == "__main__":
    main()

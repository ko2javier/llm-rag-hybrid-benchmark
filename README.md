# llm-rag-hybrid-benchmark

RAG hybrid benchmark comparing advanced retrieval techniques and self-hosted inference cost on synthetic API documentation.  
**Gemma 3 27B AWQ** vs **Qwen 2.5 32B AWQ** vs **Gemma 4 31B AWQ** — bge-m3 embeddings, HyDE, re-ranking, deterministic router, cost-per-query on real GPU pricing.

## What this measures

Building on [llm-rag-benchmark](https://github.com/ko2javier/llm-rag-benchmark), this benchmark addresses the root cause identified in Phase 1-2: the embedding model was the bottleneck.

Tested variables:
- Embedding: `all-MiniLM-L6-v2` (baseline) vs `BAAI/bge-m3` (multilingual, technical)
- Retrieval: standard vs HyDE (Hypothetical Document Embeddings)
- Re-ranking: none vs `cross-encoder/ms-marco-MiniLM-L-6-v2` vs `BAAI/bge-reranker-v2-m3`
- Routing: rules-based router vs RAG-only for exact fact queries

## Dataset

Synthetic documentation for a fictional payment API (NexusPay):
- 25 markdown files — reference docs + narrative guides
- 572 chunks (chunk_paragraph strategy)
- 50 golden questions: 25 semantic + 25 deterministic (exact facts)

## Experiments

| Experiment | What it tests | Status |
|---|---|---|
| A | Self-hosted RAG (bge-m3, semantic retrieval) across 3 candidate LLMs — quality (RAGAS) + real GPU cost/query | ✅ Done (31 Jul 2026) — see [`results/`](results/INFORME_FASE2_RESULTADOS.md) |
| B | HyDE vs standard retrieval — cost vs quality tradeoff | Script ready (`hyde.py`), not run yet |
| C | bge-reranker-v2-m3 vs ms-marco vs no re-ranker | Not started |
| D | Deterministic router vs RAG-only on exact fact questions | Router validated standalone (25/25) — not yet wired into the evaluation pipeline |

## Results (Exp A — self-hosted cost & quality, 31 Jul 2026)

Full report with methodology, incidents, and per-question-type breakdown: [`results/INFORME_FASE2_RESULTADOS.md`](results/INFORME_FASE2_RESULTADOS.md). Raw CSVs (latency/cost + RAGAS scores per model) in [`results/`](results/).

**Setup:** 1× RTX 6000 Ada (48GB) on Vast.ai, $0.6966/h real instance price, vLLM + AWQ quantization, RAGAS-judged by Mistral 7B Instruct (third model family, to avoid self-preference bias).

| Model | Latency | Cost/query | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---|---|---|---|---|---|
| Gemma 3 27B AWQ | 1.30s | $0.000252 | 0.874 | 0.856 | 0.959 | 0.924 |
| Qwen 2.5 32B AWQ | 1.37s | $0.000266 | **0.936** | 0.836 | **0.968** | 0.883 |
| Gemma 4 31B AWQ | **1.01s** | **$0.000196** | 0.888 | **0.879** | 0.961 | 0.895 |

**Headline findings:**
- **No single winner.** Gemma 4 31B is fastest/cheapest and best on deterministic questions, but its faithfulness drops the most on semantic (narrative) questions. Qwen 32B is the most balanced but slowest/most expensive. The right pick depends on the real production question mix.
- **Self-hosting doesn't pay off at demo volume.** Break-even vs the reference APIs (GLM-5.2, Kimi K3) sits between ~250K and ~2.2M queries/month assuming a GPU running 24/7 — far above portfolio/demo traffic. The value of this exercise is the deployment/measurement rigor, not immediate savings (see report §8 for the nuance on burst/spot pricing changing this math).
- **The deterministic router is not yet wired into the eval pipeline** — all 50 questions (including the 25 deterministic ones) went through pure semantic RAG. `router.py` passes 25/25 standalone, but Exp D (router integrated) is still open — see Next steps.

## Next steps

From the report's own recommendations, plus a few additions:

- **Exp D (router integration)** — the most direct next step: wire `router.classify()` into `evaluator.py` so deterministic questions hit `api_facts` lookup instead of semantic RAG, then re-score just that subset against today's baseline.
- **Exp B (HyDE)** and **Exp C (re-ranking)** — scripts exist, not run yet.
- **Quality-per-dollar view** — today's report treats cost and RAGAS quality as separate tables; worth combining into a single score (e.g. faithfulness / cost-per-query) to make the "which model is actually the best value" call explicit instead of implicit.
- **Model the break-even under realistic traffic** (bursty + scale-to-zero / spot pricing) instead of GPU-24/7 — the report flags this as the biggest unmeasured nuance in the cost conclusion (§8.2).
- **Pin the dependency versions that were debugged into working** (`ragas==0.4.3` + the `langchain-community` compatibility patch, `tiktoken_enabled=False` on `OpenAIEmbeddings`) directly in `requirements.txt`, so the next run doesn't rediscover the same two bugs.
- Populate `equivalent_api_cost_usd` directly from `evaluator.py`/`hyde.py` CLI flags instead of computing it post-hoc, as noted in the report.

## Repository structure
docs/reference/    # Rate limits, endpoints, error codes
docs/guides/       # Narrative guides (18 files)
dataset/           # Golden dataset — 50 questions
scripts/           # chunker.py, ingest.py, evaluator.py, hyde.py, router.py, ragas_eval.py
sql/               # PostgreSQL schema and seed data
results/           # Exp A results: report + per-model CSVs (latency, cost, RAGAS scores)

## Author

K. Jabier O'Reilly — [cv.ko2-oreilly.com](https://cv.ko2-oreilly.com) — [@ko2javier](https://github.com/ko2javier)

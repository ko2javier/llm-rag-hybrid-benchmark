# llm-rag-hybrid-benchmark

RAG hybrid benchmark comparing advanced retrieval techniques on synthetic API documentation.  
**Qwen 2.5 32B AWQ** vs **Gemma 3 27B AWQ-INT4** — bge-m3 embeddings, HyDE, re-ranking, deterministic router.

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
- 481 chunks (chunk_paragraph strategy)
- 50 golden questions: 25 semantic + 25 deterministic (exact facts)

## Experiments

| Experiment | What it tests |
|---|---|
| A | MiniLM vs bge-m3 — closes the argument from Phase 1-2 |
| B | HyDE vs standard retrieval — cost vs quality tradeoff |
| C | bge-reranker-v2-m3 vs ms-marco vs no re-ranker |
| D | Deterministic router vs RAG-only on exact fact questions |

## Status

🔄 Scripts and dataset ready — Vast.ai experiments pending.

## Repository structure
docs/reference/    # Rate limits, endpoints, error codes
docs/guides/       # Narrative guides (18 files)
dataset/           # Golden dataset — 50 questions
scripts/           # chunker.py, ingest.py, evaluator.py, hyde.py, router.py
sql/               # PostgreSQL schema and seed data

## Author

K. Jabier O'Reilly — [cv.ko2-oreilly.com](https://cv.ko2-oreilly.com) — [@ko2javier](https://github.com/ko2javier)

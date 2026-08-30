# llm-rag-hybrid-benchmark

RAG hybrid benchmark comparing advanced retrieval techniques and self-hosted inference cost on synthetic API documentation.  
**Gemma 3 27B AWQ** vs **Qwen 2.5 32B AWQ** vs **Gemma 4 31B AWQ** — bge-m3 embeddings, HyDE, re-ranking, deterministic router, cost-per-query on real GPU pricing.

Extended in Aug 2026 with **[FinanceBench](#external-validation-on-a-public-dataset--financebench-2230-aug-2026)** — the same pipeline on 84 real SEC filings, with human-labelled ground truth and a four-judge meta-evaluation of whether an LLM judge can be trusted at all.

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
| B | HyDE vs standard retrieval, semantic questions only — cost vs quality tradeoff | ✅ Done (01 Aug 2026) — see §13 of the report |
| C | bge-reranker-v2-m3 cross-encoder re-ranking over a 100-candidate pool | ✅ Done (21–22 Aug 2026) — **best single change**, see [V2.0 results](results/REPORT_JUDGE_SIZE_AND_V2_MATRIX.md) |
| D | Deterministic router + `api_facts` lookup vs RAG-only, deterministic questions only | ✅ Done (01 Aug 2026), prompt fixed 21 Aug — see §12 of the report |
| C+D | Re-ranker for semantic questions, router for deterministic ones | ✅ Done (21–22 Aug 2026) — **recommended configuration** |

## Results (Exp A — self-hosted cost & quality, 31 Jul 2026)

Full report with methodology, incidents, and per-question-type breakdown: [`results/REPORT_FASE2_RESULTS.md`](results/REPORT_FASE2_RESULTS.md) (English) / [`results/INFORME_FASE2_RESULTADOS.md`](results/INFORME_FASE2_RESULTADOS.md) (Spanish original). Raw CSVs (latency/cost + RAGAS scores per model) in [`results/`](results/).

**Setup:** 1× RTX 6000 Ada (48GB) on Vast.ai, $0.6966/h real instance price, vLLM + AWQ quantization, RAGAS-judged by Mistral 7B Instruct (third model family, to avoid self-preference bias).

| Model | Latency | Cost/query | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---|---|---|---|---|---|
| Gemma 3 27B AWQ | 1.30s | $0.000252 | 0.874 | 0.856 | 0.959 | 0.924 |
| Qwen 2.5 32B AWQ | 1.37s | $0.000266 | **0.936** | 0.836 | **0.968** | 0.883 |
| Gemma 4 31B AWQ | **1.01s** | **$0.000196** | 0.888 | **0.879** | 0.961 | 0.895 |

**Headline findings:**
- **No single winner.** Gemma 4 31B is fastest/cheapest and best on deterministic questions, but its faithfulness drops the most on semantic (narrative) questions. Qwen 32B is the most balanced but slowest/most expensive. The right pick depends on the real production question mix.
- **Self-hosting doesn't pay off at demo volume.** Break-even vs the reference APIs (GLM-5.2, Kimi K3) sits between ~250K and ~2.2M queries/month assuming a GPU running 24/7 — far above portfolio/demo traffic. The value of this exercise is the deployment/measurement rigor, not immediate savings (see report §8 for the nuance on burst/spot pricing changing this math).

## Results (Exp D — router-backed lookup, Exp B — HyDE, 01 Aug 2026)

Full breakdown in [`results/REPORT_FASE2_RESULTS.md`](results/REPORT_FASE2_RESULTS.md) §12-13 (English) / [`results/INFORME_FASE2_RESULTADOS.md`](results/INFORME_FASE2_RESULTADOS.md) §12-13 (Spanish). Two bugs were found and fixed *before* spending any GPU time on this, by validating `router.classify()` offline against the golden dataset: `normalize()` was dropping "IP-level" down to `"iplevel"` (losing the `"ip"` token), and the `constraint` rule's keyword map collapsed two different refund questions onto the same fact. Both fixed, re-validated 25/25 offline, then run for real.

**Exp D (router + `api_facts` lookup, deterministic questions only):** all 25/25 questions resolved via exact lookup in all three models, at ~5-6x lower latency/cost than semantic RAG. `context_precision`/`context_recall` hit **1.000** in all three models, confirming the hypothesis from Exp A. But faithfulness/answer_relevancy did **not** improve uniformly — they dropped in 2 of 3 models. Two distinct causes, verified row by row: (1) answer_relevancy drops everywhere because exact-fact answers are short ("1000", "v2") and RAGAS's relevancy metric penalizes terse answers regardless of correctness; (2) faithfulness genuinely drops for Gemma 4 31B (-0.12) and Qwen 32B (-0.075) because those models sometimes respond "the provided text does not contain..." even though the fact is right there in the prompt, in compact JSON instead of prose — Gemma 3 27B didn't have this problem (25/25 correct). The router's retrieval-side promise holds; the fact-lookup prompt template needs work for non-Gemma-3 models.

**Exp B (HyDE, semantic questions only):** confirms the hypothesis cleanly — `context_recall` improves in all three models, and the model that gains the most is exactly the one that needed it most: Gemma 4 31B, which had the worst semantic faithfulness in Exp A (0.833), jumps to 0.934 (+0.101) and recall +0.127. Costs ~2.5-3x more latency/query than plain semantic RAG (two LLM calls instead of one) — a reasonable trade if semantic/narrative questions are a meaningful share of expected production traffic.

## Methodology check — does the RAGAS judge model matter? (14–15 Aug 2026)

All RAGAS scores above were computed with a local **Mistral 7B Instruct** judge, chosen for cost/risk reasons (no paid API key exposed on a third-party rented GPU). To check whether that choice affects the reported numbers, traces from Exp A were ingested into a self-hosted [Langfuse](https://langfuse.com) instance and re-scored with two independent frontier judges — **gpt-4o** (OpenAI) and **DeepSeek-v4-pro** (DeepSeek) — across the **full 150-row set** (all 3 models × 50 questions), not just a sample.
**Final table, average of the 3 models, 150/150 rows, no missing data:**

| Metric | Mistral (local) | gpt-4o | DeepSeek | Read |
|---|---|---|---|---|
| `context_precision` | 0.963 | 0.767 (−0.196) | 0.776 (−0.187) | **Judge-general effect** — near-identical magnitude across 2 unrelated vendors |
| `context_recall` | 0.901 | 0.807 (−0.094) | 0.780 (−0.121) | **Judge-general effect** — same direction, similar magnitude |
| `answer_relevancy` | 0.857 | 0.901 (+0.044) | 0.871 (+0.014) | Same direction, but gpt-4o's effect is ~3x DeepSeek's — judge-specific magnitude, not generic |
| `faithfulness` | 0.899 | 0.902 (+0.003) | 0.915 (+0.016) | **No consistent effect** — a real revision of the original finding below |

Both judges' scores were attached side by side to the same Langfuse traces, so any single query's dual-judge comparison is inspectable in the UI:

![Trace detail with both judges' scores attached](docs/langfuse/trace_detail_dual_judge_scores.jpg)

Aggregated across all sampled traces, Langfuse's built-in score analytics confirm the same direction found per-model above (`faithfulness`: local mean 0.90 vs gpt-4o mean 0.94, from the original 30-row sample):

![Langfuse analytics: faithfulness score comparison](docs/langfuse/analytics_faithfulness_comparison.jpg)

**Conclusion, revised with full data:** `context_precision`/`context_recall` are a genuine, systematic **local-vs-frontier judge effect** — near-identical magnitude between two architecturally unrelated vendors (OpenAI, DeepSeek), not a gpt-4o quirk, not sampling noise. `answer_relevancy` keeps its direction but the magnitude is judge-specific. The initial 30-row sample also suggested `faithfulness` scored consistently higher under gpt-4o across all 3 models — with full 150-row coverage and a second independent judge, that does **not** hold: per-model deltas are mixed in sign and the aggregate is nearly flat for both frontier judges. This is a real revision of the original claim, not just "confirmed at scale." Practical implication stands regardless: **any report citing RAGAS numbers should name the judge model.**

## Results (judge-size study + V2.0 matrix quality — 22 Aug 2026)

Full report: [`results/REPORT_JUDGE_SIZE_AND_V2_MATRIX.md`](results/REPORT_JUDGE_SIZE_AND_V2_MATRIX.md). Per-row scores in [`results/judge_size/`](results/judge_size/).

The section above establishes that the judge model matters. It leaves one thing unresolved: the
local judge was both **smaller** *and* a **different family** from the frontier judges, so the two
explanations were confounded. This adds the missing middle point — **Mistral Nemo 12B**, the same
family and architecture as the 7B judge, differing only in size — scored on the same 150 rows.

| Judge | context_precision | Paired diff vs Nemo | 95% CI | Distinguishable? |
|---|---|---|---|---|
| Mistral **7B** (local) | 0.963 | −0.186 | [−0.240, −0.131] | **Yes** |
| **Nemo 12B** (local) | **0.777** | — | — | — |
| gpt-4o (API) | 0.767 | +0.010 | [−0.024, +0.045] | No |
| DeepSeek (API) | 0.776 | −0.002 | [−0.046, +0.042] | No |

**A self-hosted 12B judge is statistically indistinguishable from both frontier judges on `context_precision`; the 7B is not.** Same family as the 7B, so this is a **capacity** effect, not a vendor one — the threshold sits below 12B. Two caveats that matter: `context_recall` does *not* follow (Nemo lands midway, closing only ~40% of the gap), and the 7B's per-row σ is 0.075 against ~0.34 for the other three — it barely discriminates, rather than merely scoring high.

**Applying that judge to the full V2.0 matrix** (45 runs × 50 questions = 2,250 rows, 2 missing cells):

| Config | ctx_precision | ctx_recall | faithfulness | ans_relevancy |
|---|---|---|---|---|
| A baseline | 0.777 | 0.866 | 0.920 | 0.880 |
| B HyDE | 0.774 | 0.890 | 0.923 | 0.882 |
| C reranker | 0.847 | **0.950** | 0.910 | **0.931** |
| D router | 0.838 | 0.878 | 0.939 | 0.783 |
| **C+D** | **0.867** | 0.943 | 0.939 | 0.840 |

Deltas are paired per (model, repetition, question), n = 450, so their uncertainty is measurable. **HyDE moves nothing**: not one of its four metrics is distinguishable from the baseline. C+D gains +0.090 context precision, 95% CI [+0.064, +0.116]. And **no configuration changes `faithfulness`** — that metric has now shown no effect across four judges *and* five retrieval configurations.

**`hit@5` was wrong in both directions.** It scored the reranker at +1 question (21 vs 20, indistinguishable from noise) where RAGAS measures +0.070 precision / +0.084 recall consistently across all three models — and it scored HyDE as the highest-ceiling config where RAGAS measures −0.003 precision at ~3× the latency. A metric that errs in both directions is not pessimistic or optimistic; it is not measuring what it was assumed to measure. `hit@5` only asks whether the gold filename is in the top 5, never whether the retrieved passage is usable.

**Infrastructure finding:** scoring 2,250 rows took **1h08 instead of 10h08 on the same GPU**, by running six client processes against one vLLM server rather than raising RAGAS's `max_workers`. The server was idle 75% of the time with an empty queue — the bottleneck was client-side. Raising `max_workers` to 64 instead ran 27% "faster" while silently losing 47 of 50 `context_precision` cells to timeouts, which only surfaced because the scorer counts missing cells per metric.

## External validation on a public dataset — FinanceBench (22–30 Aug 2026)

Everything above runs on NexusPay, a **synthetic** knowledge base written for this benchmark. That
is a fair criticism of it, so the same questions were re-asked on a public dataset of **real**
documents: [FinanceBench](https://github.com/patronus-ai/financebench) — 150 analyst questions over
84 SEC 10-K/10-Q filings (12,013 pages), with reference answers written by financial analysts.

Full report: [`results/REPORT_FINANCEBENCH.md`](results/REPORT_FINANCEBENCH.md). Data and a
no-GPU reproduction script: [`results/financebench/`](results/financebench/).

**Setup:** three parsers (`pdftotext`, `pdfplumber`, `docling`) × 3 generators (the same Gemma 3 27B
/ Qwen 32B / Gemma 4 31B AWQ used above) × 3 retrieval scenarios. 4,500 generations, run serially,
0 errors. 150 answers labelled blind by a human annotator; 4 LLM judges scored on those same cases.

**Headline findings:**

- **Retrieval is the bottleneck, and the replication is the evidence.** All three models lose
  **exactly 42.3 points** going from "here is the page" to "search 84 filings"
  (`+0.423 [+0.269, +0.577]`, identical to three decimals across two vendors). Models differ;
  retrieval is what they share.

  | | gold page | correct filing given | search 84 filings |
  |---|---|---|---|
  | mean of 3 models | 0.897 | 0.603 | 0.474 |

- **Only 6% of failures are picking the wrong company.** 54% retrieve the right filing and the
  wrong page. Thirty pages of a 10-K mention *property, plant and equipment*; one carries the
  figure, and semantic similarity does not separate "discusses X" from "contains the value of X".
- **PDF parser choice does not change answer accuracy** — 36 paired comparisons, none
  distinguishable from zero. `pdftotext` (0.5 min CPU) matches `docling` (29.5 min GPU). The
  decision rule was pre-registered before the data existed.
- **Five retrieved pages are worse than one, even when the right one is among them** — accuracy
  drops 21–28 points in the corpus scenario. `hit@5` cannot see this by construction, which extends
  the `hit@5` critique in the section above to a second, independent dataset.
- **Global agreement does not validate an LLM judge.** Four judges on the same 150 cases:

  | judge | catches real errors | global agreement |
  |---|---|---|
  | Nemo 12B (local) | 42.9% | 83.7% |
  | Claude | 57.1% | 87.8% |
  | Gemini | 71.4% | **91.8%** |
  | *approve everything unread* | *0.0%* | *90.5%* |

  The best judge beats looking at nothing by **1.4 points** on the metric usually published, while
  error detection separates the judges by **71**. This is the meta-evaluation the judge-size study
  above could not do: it had no human ground truth to check against.
- **A human gold standard has errors too, and cross-judge disagreement finds them.** 17 of 150
  labels (11%) proved revisable — in both directions, so not a simple leniency bias. Two of them
  are errors in the *dataset's* reference, not the annotation.
- **An evaluator cannot audit the exam it is sitting.** When one of the three judges arbitrated the
  gold standard, its own score rose to 100%. The contamination is measured rather than assumed: 82%
  of the revisions match that judge's own prior verdicts, against ~50% for the other two. The
  published table therefore stays on the original, independent labels.

**Comparison with the original paper, stated carefully:** with the evidence page supplied, the
three open 4-bit models reach **84.7%** against the **85.3%** the FinanceBench paper reports for
GPT-4-Turbo in its Oracle configuration. Same dataset, humans judging on both sides — but the
paper's annotators are financial experts and this study's are not, the answer categories differ,
and only the Oracle row is comparable at all. Indicative, not strict; the four differences are
listed in the report.

**Not done, and why:** HyDE and re-ranking were planned and skipped. Perfect retrieval would reach
0.667–0.727 here, so headroom over the current 0.474 is ~20–25 points; a re-ranker measured at
+0.05 `hit@k` on this corpus captures about 3 of them. Reducing `k` was proposed and discarded the
same day — at `k=1` the gold page is present 19.2% of the time versus 63.5% at `k=5`, so projected
accuracy would fall from 0.481 to ~0.215.

## Related work

This benchmark's retrieval/embedding infrastructure (chunking, bge-m3 embeddings, `router.py`) is reused as the foundation for [llm-agent-mcp-eval](https://github.com/ko2javier/llm-agent-mcp-eval), a follow-up project that adds tool-calling agent behavior on top of it — MCP, multi-turn persona evaluation, and a tool-design bug root-caused across three model vendors. Findings from both projects are synthesized in the [Evaluation Engineering — Calibration Record](https://cv.ko2-oreilly.com/calibration-record).

## Next steps

- **Populate `api_facts.source_file`** — the column exists in the schema but the seed never fills it, so router rows emit the literal string `api_facts` as their source. That is why `hit@5` reads 0/25 on router runs: a metric artefact, not a failure. Filling it would make retrieval metrics meaningful on the deterministic path.
- **Investigate the router's `answer_relevancy` drop (−0.098)** — the mechanism is understood (raw facts produce terse answers) but not whether it is a defect. For a deterministic question a terse exact answer *is* correct, so the metric may be penalising desired behaviour. Needs per-response inspection, not more averages.
- **Locate the judge-size threshold** — 12B is enough on this corpus; 9B or 10B may also be. One more size in the same family would turn a single point into a curve.
- **Quality-per-dollar view** — combine cost and RAGAS quality into one score (e.g. faithfulness / cost-per-query) instead of two separate tables, across all of Exp A/B/C/D.
- **Model the break-even under realistic traffic** (bursty + scale-to-zero / spot pricing) instead of GPU-24/7 — the report flags this as the biggest unmeasured nuance in the cost conclusion (§8.2).

## Repository structure
docs/reference/    # Rate limits, endpoints, error codes
docs/guides/       # Narrative guides (18 files)
dataset/           # Golden dataset — 50 questions
scripts/           # chunker.py, ingest.py, evaluator.py, hyde.py, router.py, ragas_eval.py
sql/               # PostgreSQL schema and seed data
results/           # Exp A results: report + per-model CSVs (latency, cost, RAGAS scores)
results/financebench/  # FinanceBench: judge-vs-human labels, pre-registration, repro script

## Author

K. Jabier O'Reilly — [cv.ko2-oreilly.com](https://cv.ko2-oreilly.com) — [@ko2javier](https://github.com/ko2javier) — [Calibration Record](https://cv.ko2-oreilly.com/calibration-record)

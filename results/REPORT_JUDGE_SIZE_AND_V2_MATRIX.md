# How big does a RAGAS judge need to be?

**A self-hosted 12B judge is statistically indistinguishable from gpt-4o and DeepSeek on
`context_precision`. A 7B judge is not.**

This study closes a gap left open by an earlier finding in this repository: RAGAS scores depend
systematically on which model does the judging, but the evidence only covered two extremes — a 7B
model running locally and two frontier models behind paid APIs. Nothing in between. This report
adds the middle point, and finds that the effect is **capacity**, not vendor.

A second experiment, run on the same infrastructure, scores the full V2.0 configuration matrix
(2,250 rows) with that validated judge — the first time those runs have been measured on answer
quality rather than retrieval hit-rate alone.

---

## 1. Background: why the judge is a variable, not an oracle

Phase 2 evaluates a self-hosted RAG pipeline over NexusPay, a synthetic payments-API corpus (24
markdown files, 50 golden questions split evenly between semantic and deterministic).

Quality is scored with RAGAS, which needs an LLM to act as judge. In July that judge was chosen to
be **local** (`mistralai/Mistral-7B-Instruct-v0.3`) rather than a paid API — a deliberate
cost/risk trade-off, since the alternative meant putting an API key on a rented third-party GPU
instance.

In August that trade-off was measured. Re-scoring the same 150 rows with **gpt-4o** and with
**DeepSeek-v4-pro** produced a systematic gap, in the same direction for both vendors:

| Metric | Mistral 7B (local) | gpt-4o | DeepSeek |
|---|---|---|---|
| `context_precision` | 0.963 | 0.767 | 0.776 |
| `context_recall` | 0.901 | 0.807 | 0.780 |

Two architecturally unrelated providers landing on nearly the same number, far from the local
judge, ruled out a vendor quirk. What it could not distinguish was **size** from **family**: the
local judge was both smaller *and* a different lineage from the frontier models.

## 2. Design

**Judge under test: `mistralai/Mistral-Nemo-Instruct-2407` (12B).**

Chosen so that exactly one variable changes. Nemo is the same company, family and architecture
(`MistralForCausalLM`) as the incumbent 7B judge; only the parameter count differs. A judge from a
different vendor would have changed two things at once, making any difference unattributable.

**Why no Llama:** Meta publishes no dense model between 8B and 70B (3.1 is 8/70/405B; 3.2 is
1/3B plus 11B and 90B vision models; 3.3 is 70B only; Llama 4 is MoE). The gap in the table is
literally the gap in Meta's catalogue.

**Same rows.** The judge scored *exactly* the 150 rows the other three judges scored. Scoring
different rows would have produced a number comparable to nothing.

**Same metric configuration.** `answer_relevancy` depends on the embedding model and on RAGAS's
`strictness` parameter, so the original run's setup was reproduced: `BAAI/bge-m3` served locally,
default `strictness`. RAGAS emitted a warning that the judge returned 1 generation where 3 were
requested — before assuming this broke comparability, the original run's log was checked and found
to contain the same warning 103 times. Both runs score with one generation. Verified, not assumed.

## 3. Result: 150/150 rows, no missing cells

| Judge | context_precision | context_recall | faithfulness | answer_relevancy |
|---|---|---|---|---|
| Mistral **7B** (local) | 0.963 | 0.901 | 0.899 | 0.857 |
| **Mistral Nemo 12B (local)** | **0.777** | 0.859 | 0.920 | 0.882 |
| gpt-4o (API) | 0.767 | 0.807 | 0.902 | 0.901 |
| DeepSeek (API) | 0.776 | 0.780 | 0.915 | 0.871 |

All four judges scored the same rows, so the correct comparison is **paired** — the per-row
difference removes question difficulty as a source of variance. Mean difference against Nemo 12B,
with 95% confidence intervals (`±1.96·σ/√n`):

| Comparison | Difference | 95% CI | Distinguishable? |
|---|---|---|---|
| Nemo − gpt-4o | +0.010 | [−0.024, +0.045] | **No** — includes zero |
| Nemo − DeepSeek | −0.002 | [−0.046, +0.042] | **No** — includes zero |
| Nemo − Mistral 7B | −0.186 | [−0.240, −0.131] | **Yes** — excludes zero by a wide margin |

Since Nemo shares family and architecture with the 7B judge, "Mistral scores differently" is ruled
out. **The effect is capacity, and the threshold sits below 12B.**

**A related observation from the same calculation:** the 7B judge's per-row standard deviation is
**0.075**, against **~0.34** for the other three. It does not merely score higher — it barely
discriminates, assigning nearly the same score to everything. That is a difference in behaviour,
not just calibration.

### 3.1. What this does not show

- **`context_recall` does not follow.** Nemo lands at 0.859, roughly midway between the 7B (0.901)
  and the frontier judges (0.807 / 0.780). "Size explains everything" is false: it explains almost
  all of `context_precision` and about half of `context_recall`.
- **`faithfulness` is flat across all four judges** (0.899–0.920). No judge effect of any kind.
- **One intermediate point, one domain.** This says 12B is enough; it does not locate the threshold
  (it could be 9B), and the corpus is synthetic. Whether the same holds on a real, human-annotated
  corpus is untested.
- **A smaller judge has a reliability cost that averages hide.** In the 9,000 cells of the matrix
  evaluation, Nemo produced 2 schema violations — emitting `"verdict": 0.5` where RAGAS requires a
  binary decision, which pydantic rejects. 0.02% of cells, but the frontier judges produced none.

## 4. Applying the judge: the V2.0 matrix, 2,250 rows

The matrix compares five retrieval configurations across 3 models × 3 repetitions:

| Config | What it does | Hypothesis under test |
|---|---|---|
| **A** baseline | bge-m3 embedding → top-5 by cosine over pgvector (HNSW) | reference point |
| **B** HyDE | LLM writes a hypothetical answer first; that gets embedded instead of the question | a question and its answer are worded differently, so embedding a pseudo-answer should retrieve better |
| **C** reranker | retrieve a pool of 100 by cosine, reorder with `bge-reranker-v2-m3` cross-encoder | a cross-encoder sees query and document together, so it judges relevance better than a bi-encoder can |
| **D** router | rule-based classifier; deterministic questions become a SQL lookup in `api_facts` instead of a vector search | for factual questions, text search is the wrong tool — a fact belongs in a table |
| **C+D** | reranker for semantic, router for deterministic | — |

Until now these had only been measured on `hit@5`, latency and token count. None of those measures
answer quality.

| Config | ctx_precision | ctx_recall | faithfulness | ans_relevancy |
|---|---|---|---|---|
| A baseline | 0.777 | 0.866 | 0.920 | 0.880 |
| B HyDE | 0.774 | 0.890 | 0.923 | 0.882 |
| C reranker | 0.847 | **0.950** | 0.910 | **0.931** |
| D router | 0.838 | 0.878 | 0.939 | 0.783 |
| **C+D** | **0.867** | 0.943 | 0.939 | 0.840 |

Coverage: 2 missing cells out of 9,000 (0.02%), both the schema violations described above.

Each configuration ran the same questions on the same models and repetitions, so deltas can be
paired per (model, repetition, question) and their uncertainty estimated directly — the same
treatment applied to the judge comparison in §3. Delta against the A baseline, 95% CI, n = 450:

| Config | ctx_precision | ctx_recall | faithfulness | ans_relevancy |
|---|---|---|---|---|
| B HyDE | −0.003 [−0.028, +0.022] | +0.023 [−0.004, +0.051] | +0.003 [−0.015, +0.021] | +0.002 [−0.016, +0.020] |
| C reranker | **+0.070 [+0.049, +0.091]** | **+0.084 [+0.058, +0.109]** | −0.010 [−0.028, +0.008] | **+0.050 [+0.030, +0.070]** |
| D router | **+0.061 [+0.036, +0.085]** | +0.011 [−0.004, +0.027] | +0.020 [−0.001, +0.041] | **−0.098 [−0.120, −0.075]** |
| C+D | **+0.090 [+0.064, +0.116]** | **+0.076 [+0.049, +0.102]** | +0.019 [−0.004, +0.042] | **−0.041 [−0.071, −0.010]** |

Bold marks intervals excluding zero. Two consequences worth stating plainly:

- **HyDE moves nothing.** Not one of its four metrics is distinguishable from the baseline, on 450
  paired observations.
- **No configuration changes `faithfulness`.** The apparent 0.939 for D and C+D in the table above
  is not distinguishable from the baseline's 0.920, and neither is C's apparent −0.010. That metric
  has now shown no effect across four judges *and* five retrieval configurations.

### 4.1. `hit@5` was wrong in both directions

| | What `hit@5` said | What RAGAS says |
|---|---|---|
| **C reranker** | +1 question (21 vs 20) — indistinguishable from noise | **+0.070 precision, +0.084 recall** — substantial, consistent across all 3 models |
| **B HyDE** | the highest ceiling of all configs (21–23/25) | **−0.003 precision** — no improvement, at ~3× the latency (1.49–2.34s vs 0.53–0.86s) |

A metric that errs in *both* directions is not merely pessimistic or optimistic — it is not
measuring what it was assumed to measure. `hit@5` only asks whether the gold filename appears among
the top 5; it says nothing about whether the retrieved passage is usable.

The mechanism was visible in a single question (Q034) before it was measurable: the reranker
preferred `testing_guide.md` over the gold `architecture_overview.md`, which `hit@5` scored as a
miss — yet with the baseline's chunks (which *did* include the gold file) two of the three models
refused to answer, and with the reranker's chunk they answered correctly.

### 4.2. The router's cost, stated as an open question

**D drops `answer_relevancy` by 0.098**, the largest negative movement in the table. The mechanism
is clear: the router returns a raw fact from `api_facts`, so answers become terse.

It is not clear that this is a defect. For a deterministic question — *"what is the minimum partial
refund amount?"* — a terse, exact answer is the correct one, and the metric may be penalising
desired behaviour. This is recorded as an open question, not a finding: confirming it requires
inspecting individual responses, not reading averages.

### 4.3. Reproducibility

Standard deviation of `context_precision` across the 3 repetitions ranges from 0.004 (router —
a SQL lookup has no sampling noise) to 0.017 (HyDE — the only config whose retrieval depends on an
LLM generation). Within each config, the three evaluated models produce near-identical numbers:

```
C+D:  gemma3_27b 0.862   gemma4_31b 0.868   qwen32b 0.872
```

Retrieval quality does not depend on the generating model, which is what should happen and is
rarely verified.

## 5. Infrastructure: 2,250 rows in 1h08 on one GPU

Scored single-process, this job would have taken **10h08**. The measurement that changed the plan:

```
GPU utilisation 0%.   KV cache 2%.   Waiting: 0 requests, always.
```

`Waiting: 0` is decisive — the server never had a queue. It was idle, waiting for work. The
bottleneck was the client: RAGAS dispatches near-serially regardless of the concurrency it is given.

The obvious fix made things worse, silently:

| Configuration | 50 rows / 200 evaluations | Data |
|---|---|---|
| 1 process, `max_workers=16` | 13:47 | complete |
| 1 process, `max_workers=64` | 10:00 | **47 of 50 `context_precision` cells lost**, 59 timeouts |

27% "faster" because it was failing faster. With more coroutines in flight, each request's wait
grows past the 180s timeout and exceptions become silent `NaN`s. Without a per-metric missing-cell
count, that run would have entered this table as valid.

**The fix was six Python processes against one vLLM server**, each keeping the 16 workers that
demonstrably work. Each process owns an event loop, so N processes are N independent queues rather
than one queue N times longer.

| | 1 process | 6 processes |
|---|---|---|
| Concurrency observed at vLLM | `Running: 0–2` | `Running: 15–60` |
| 2,250 rows | 10h08 | **1h08** |
| GPU cost | ~$6.60 | **~$1.00** |

Same GPU, no additional hardware. Partitioning a larger GPU with MIG would have been *worse*: each
partition needs its own copy of the weights (22.8 GB), so an 80 GB card split in two yields less
total KV cache — and two schedulers batch less efficiently than one.

## 6. Reproducing

Scoring is done by [`scripts/ragas_nemo_compare.py`](../scripts/ragas_nemo_compare.py), which
auto-detects both CSV schemas in this repo (RAGAS's own column names, and `evaluator.py`'s).

```bash
# 1. pinned deps -- these match the stack every reference number was measured on
pip install vllm==0.26.0 transformers==5.14.1 ragas datasets pandas \
            langchain-openai langchain-huggingface sentence-transformers

# 2. judge. NO --chat-template and NO --tokenizer-mode: this repo resolves to Mistral's own
#    tokenizer/config path, which is exactly what the 7B judge used and works as-is.
vllm serve mistralai/Mistral-Nemo-Instruct-2407 --port 8081 \
     --max-model-len 16384 --gpu-memory-utilization 0.82 \
     --ignore-patterns "model-*.safetensors"

# 3. embeddings -- only AFTER vLLM answers /health (see note below).
#    PORT is not optional: embed_server_batching.py defaults to 8081, where vLLM listens.
PORT=8083 python scripts/embed_server_batching.py

# 4. score. One process per CSV, 16 workers each -- see §5 for why not more workers.
python scripts/ragas_nemo_compare.py --csv-dir <dir> \
       --judge-url http://localhost:8081/v1 \
       --judge-model mistralai/Mistral-Nemo-Instruct-2407 \
       --embed-url http://localhost:8083/v1
```

Startup order is not cosmetic: vLLM sizes its KV cache by measuring free GPU memory at startup, so
launching the embedding server concurrently makes it measure against a transient peak.

Environment note: on Blackwell (`sm_120`) FlashInfer's JIT sampling kernel fails to build; set
`VLLM_USE_FLASHINFER_SAMPLER=0` to use vLLM's native sampler instead. This changes the top-k/top-p
kernel, not the sampling algorithm — and the judge runs at `temperature=0` regardless.

Hardware used: NVIDIA RTX PRO 5000 Blackwell 48 GB, ~4.5 hours, ~$3 total.

Raw per-row scores for the judge study are in [`judge_size/`](judge_size/) (3 files, 150 rows).

## 7. Dataset correction made during this work

`dataset/golden_dataset.json` declared three files under `guides/` that actually live in
`reference/` — 15 occurrences, affecting 13 of the 25 semantic questions. Every published metric had
been computed by comparing **filenames only**, a workaround that hid the mislabelling.

The labels were corrected and the effect measured across the 9 baseline runs:

| Comparison | hit@5 |
|---|---|
| Filename only (the previous workaround) | 20/25 |
| Strict path, **corrected** labels | **20/25** |
| Strict path, old labels | 7/25 |

No published figure moves, and the workaround is no longer needed. RAGAS metrics were never
affected — they compare text, not paths.

**Known and not fixed:** `api_facts` declares a `source_file` column that the seed never populates,
so router rows emit the literal string `api_facts` as their source. This is why `hit@5` reads 0/25
on router runs — a metric artefact, not a failure. Populating it would make that metric meaningful
for the deterministic path.

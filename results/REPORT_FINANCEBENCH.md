# FinanceBench: what a PDF parser changes, and what an LLM judge is worth

*A retrieval-and-generation study on 84 real SEC filings, with human ground truth.*
*Aug 22–30, 2026. All figures below were measured in this study unless attributed to the original paper.*

---

## Summary

This study asks two questions on the public [FinanceBench](https://github.com/patronus-ai/financebench)
dataset (150 analyst questions over 84 SEC 10-K/10-Q filings):

1. **Does PDF parsing quality change the answers?** Measured end to end, not just at retrieval.
2. **Is an LLM judge trustworthy?** Measured against human labels on the same 150 cases.

**Findings, in order of how much they hold up:**

- **Parser choice does not change answer accuracy** — 36 paired comparisons, three models, two
  scoring rules, no distinguishable difference. `pdftotext` (0.5 min CPU) matches `docling`
  (29.5 min GPU).
- **Retrieval, not the model, is the bottleneck.** Three models from two vendors each lose
  **42.3 points** when moving from "here is the page" to "search 84 filings" — identical to three
  decimal places, which points at the component they share.
- **Global agreement does not validate an LLM judge.** The best of four judges scores 91.8%
  against a 90.5% floor from approving everything unread. What separates judges is how many real
  errors they catch: 0% to 71%.
- **A human gold standard has errors too, and cross-judge disagreement finds them** — 17 of 150
  labels (11%) proved revisable, in both directions.
- **An evaluator cannot audit the exam it is sitting.** When one of the three judges arbitrated
  the gold standard, its own score rose to 100%. The contamination is measured, not assumed.

**Scope limits, stated up front:** N=150 questions (52 with machine-checkable numeric answers);
one embedding model (bge-m3); three open-weight models, all 4-bit AWQ quantized; human labels
produced by one non-specialist annotator. Details in [Limitations](#limitations).

---

## Setup

| | |
|---|---|
| **Dataset** | FinanceBench (Patronus AI), 150 questions, 84 SEC filings, 12,013 pages |
| **Parsers** | `pdftotext` (poppler), `pdfplumber`, `docling` — 36,039 page extractions |
| **Embeddings** | `BAAI/bge-m3`, 1 page = 1 chunk (0 of 36,039 pages exceed 8,192 tokens) |
| **Generators** | Gemma 3 27B AWQ · Qwen 2.5 32B AWQ · Gemma 4 31B AWQ (vLLM) |
| **Judges** | Mistral Nemo 12B (local), Claude, Gemini |
| **Human labels** | 150 blind labels by one annotator; 17 later revised (see [§6](#6-auditing-the-gold-standard)) |
| **Hardware** | RTX 5880 Ada 49 GB / L40 45 GB on Vast.ai |
| **Generations** | 4,500 (1,800 gold-page + 2,700 end-to-end), 0 errors, 0 lost rows |

Three retrieval scenarios are used throughout:

| scenario | what the model gets | search space |
|---|---|---|
| **gold page** | the evidence page(s), no retrieval | — |
| **intra-doc** | top-5 pages from the correct filing | ~139 pages |
| **corpus** | top-5 pages from everything | 12,013 pages |

---

## 1. Parser choice does not change answer accuracy

Each of the 150 questions was answered from **the same page written four ways**: the three parsers
plus `evidence_text_full_page` shipped with the dataset. Same question, same page, only the text
differs — a fully paired design.

**Result: none of the 36 comparisons (4 variants × 3 models × 2 scoring rules) is distinguishable
from zero.** The dataset's own reference text does not win either, which matters: if parsing were
degrading comprehension, the un-parsed reference would stand out. It does not.

This holds end to end as well — no parser difference in either retrieval scenario.

**Decision rule was pre-registered before looking at the data** (three branches: no effect →
`pdftotext`; docling wins → docling, and its 60× cost becomes a publishable finding; pdfplumber
loses figures → discarded). The first branch fired. **`pdftotext` is fixed** for the rest of the
study.

**Honest reading:** with 52 questions at ~90% accuracy, only large differences are detectable.
This does not show the parsers are identical. It shows that if a difference exists it is small,
and that choosing a parser for answer quality optimizes something that does not move.

A caveat worth keeping: at the extraction stage, `pdfplumber` silently dropped an entire column
of a Johnson & Johnson table (21 of 40 figures). `hit@k` scores that as a perfect hit — it only
checks whether the page was retrieved, never what survived inside it. Generation is what can see
that class of failure, and at N=150 it did not surface as a measurable effect.

---

## 2. Retrieval is the bottleneck, and the evidence is the replication

```
                gold page   intra-doc   corpus
Gemma 3           0.885       0.558      0.462
Qwen 32B          0.904       0.596      0.481
Gemma 4           0.904       0.654      0.481
mean              0.897       0.603      0.474
```

**All three models lose exactly 42.3 points** between gold page and corpus:
`+0.423 [+0.269, +0.577]` paired 95% CI, identical to three decimals across three models from two
vendors. Models differ in architecture, vendor and size; **the retrieval stage is what they
share**, so that is where the loss lives.

### Why retrieval fails — measured, not theorized

```
searching 84 filings (pdftotext)
  finds the right page                     40%
  right filing, wrong page                 54%   <- the actual problem
  wrong filing entirely                     6%
  on average 2.35 of the 5 pages belong to another company
```

**Only 6% of failures are company confusion.** With the correct document handed over (the router
scenario), the system still fails 44% of the time, with zero cross-company contamination. The
hard part is telling **which page of a 139-page 10-K holds the number**: thirty pages mention
*property, plant and equipment* — cash-flow statement, balance sheet, notes, MD&A, risk factors,
depreciation policy — and one carries the figure. Semantic similarity does not separate "discusses
X" from "contains the value of X".

**Knowing which document to open is worth more than any parser.** The intra-doc vs corpus gap is
`+0.160 [+0.107, +0.220]`, roughly 2.7× the largest parser effect measured at retrieval — and it
was significant in 15 of 15 measurements.

### Five pages is worse than one, even when the right one is included

Paired comparison, same questions, correct page present in both conditions:

```
intra-doc   accuracy drops 10–16 points
corpus      accuracy drops 21–28 points   (5 of 6 comparisons significant)
```

Handing over 5 pages when 1 suffices **lowers accuracy even though the right page is there**, and
the damage is worse in corpus, where the distractors come from other companies. `hit@5` cannot see
this by construction: it scores a page found alone and a page found in noise identically.

---

## 3. The ceiling on any retrieval-side improvement

Splitting the 52 numeric questions by whether the gold page reached the prompt:

```
corpus, pdftotext        page present   page absent    coverage
  Gemma 3                    0.667         0.105       33/52 = 63.5%
  Gemma 4 / Qwen 32B         0.727         0.053
```

The measured accuracy decomposes exactly:
`0.635 × 0.727 + 0.365 × 0.053 = 0.481`, matching the observed 0.481. This makes the decomposition
usable for projecting decisions without spending GPU time.

**Two consequences.**

**A ceiling.** Perfect retrieval would reach **0.667–0.727**, not the 0.897 of the gold-page
condition — because top-5 noise costs its own 17 points (§2). Available headroom over the current
0.474 is roughly **+20 to +25 points**. A re-ranker measured at +0.05 `hit@k` on this corpus
captures about **3** of them — one sixth.

**No parametric contamination.** When the gold page is absent, accuracy is **0.053**, and exactly
**0.000** for Gemma 3 in the intra-doc scenario. The models do not recall these figures from
pre-training and do not guess them. Everything they get right comes from the retrieved document.

### What we did not do, and why

HyDE and re-ranking were planned and **not executed**. The decomposition above is the reason: they
address coverage, and capture a small share of the available headroom, against a problem whose
measured cause is ranking pages that all discuss the same topic. Re-ranking on this corpus was
also measured to be **conditional** — `−0.02` intra-doc, `+0.05` corpus — so its sign depends on
pool size rather than being an unconditional improvement.

One idea was proposed and discarded the same day. Since 5 pages cost 17 points versus 1, reducing
`k` looks attractive. Coverage by `k` rules it out:

```
coverage, 52 numeric      k=1     k=2     k=3     k=4     k=5
  intra-doc              32.7%   57.7%   67.3%   71.2%   75.0%
  corpus                 19.2%   25.0%   42.3%   55.8%   63.5%
```

At `k=1` in corpus the gold page is present 19.2% of the time. Projected through the validated
decomposition, accuracy would fall from 0.481 to ~0.215 — losing four times more in coverage than
is gained in noise. **`k=5` is already near optimal.** Accuracy at `k`=2, 3 and 4 was **not**
measured; only the endpoints are known (0.897 and 0.727) and the curve between them is not
interpolated here.

---

## 4. Global agreement does not validate an LLM judge

Four judges scored the same 150 cases blind — no judge saw which model or parser produced an
answer, the same condition the human annotator worked under.

```
                       catches real errors      global agreement
Nemo 12B (local)          6/14 = 42.9%               83.7%
Claude                    8/14 = 57.1%               87.8%
Gemini                   10/14 = 71.4%               91.8%
approve everything        0/14 =  0.0%               90.5%
```

**The best judge beats "approve everything unread" by 1.4 points.** Global agreement is the metric
most commonly published to validate LLM judges, and on this set it does not separate the best
judge from not looking at all — because 89% of answers are correct and saying "yes" to everything
scores 90.5%.

What does separate them is **how many real errors they catch**: 0% to 71%. Any report on judge
quality should lead with that number and print the trivial baseline next to it.

**Judges order by capability — 43% → 57% → 71% — and none reaches 100%.** The 12B local judge is
clearly short, and it is the judge that scored 2,250 rows in the preceding phase of this work.

**There is no single shared blind spot.** Only 2 of 14 errors escape all three judges, and one
case (AMCOR quick ratio) is caught by the 12B judge and missed by both frontier judges — which
rules out a simple "more capability, more hits" story. The pattern is that **judges struggle
exactly where the dataset's own reference is arguable**.

### Judges are reproducible under batching; generators are not

```
40 verdicts serial vs the same 40 in parallel  ->  40 identical, 0 different
```

Whatever is wrong with a small judge is **criterion, not instability**. This matters because the
generation side behaves differently — see §7.

---

## 5. Consequence for LLM-judged scores generally

The 2,250 RAGAS scores in the preceding phase were produced by the judge that catches 43% of real
errors. That does not invalidate comparisons between configurations — the bias applies equally to
all of them — but it does constrain what can be claimed:

> **Absolute scores from an LLM judge are not trustworthy. Differences between configurations
> measured with the same judge are.**

---

## 6. Auditing the gold standard

Cross-checking human labels against three judges surfaced disagreements. Reviewing **only** the
cases where judges contradicted the human would correct in one direction — the one that flatters
the judges — so all **31 cases with any disagreement** were reviewed, both ways.

**17 of 150 labels (11%) were revised: 10 too lenient, 7 too strict.** The 9 cases where a single
judge dissented were reviewed and none changed — there the judge was wrong, which is what shows
the review was not biased by construction.

Three internal inconsistencies made the case without relying on any judge: the same question,
labeled differently across repeats. For 3M, `"Consumer"` (correct) and `"Translation"` (currency
translation, not a segment) were both marked correct. For AES, the identical figure 12.14 was
marked correct once and incorrect once. In each group **the original labels rewarded the worse
answer and penalized the better one**.

**Two of the 17 are dataset errors, not annotation errors.** FinanceBench answers "Yes, Pfizer is
spinning off Upjohn" for a Q2-2023 filing; Upjohn was spun off in November 2020 (merged with Mylan
into Viatris), and the cited evidence only describes residual separation costs. The models
answered that no spin-off is in progress and explained why. That case's `justification` field is
empty. **Public reference datasets age.**

### An evaluator cannot audit its own exam

The revised standard was arbitrated by Claude — **one of the three judges being measured**. Scored
against it, Claude and Gemini both reach 100%. That is not a result; it is the circularity made
visible, and it is measurable:

```
Of the 17 revisions, how many match what each judge already said?
   Claude (the arbiter)   14/17 = 82%
   Gemini                  9/17 = 53%
   Nemo                    8/17 = 47%
```

**82% against ~50%.** The arbiter rewrote the standard toward its own prior criterion.

**Therefore the published judge table is the one built on the original, independent human labels.**
The revised standard is reported as sensitivity analysis with the 82% stated alongside, and only
the row for **Nemo** is interpretable there — it did not participate and does not share a model
with the arbiter: **42.9% → 75.0%**, meaning a substantial share of its apparent errors were
errors in the yardstick.

**The headline survives both standards**, which is the point worth keeping:

| | best judge's margin over "approve everything" | range in error detection |
|---|---|---|
| original standard | +1.4 pts | 0% → 71% |
| revised standard | +8.6 pts | 0% → 100% |

In both, global agreement separates by a few points while error detection separates by tens.

---

## 7. Reproducibility notes

Three distinct ways to break reproducibility, all measured on the same day with
`temperature=0, seed=0`:

```
send requests in parallel     ->  17 of 40 answers change
restart the vLLM server       ->  23 of 40 change     <- the largest effect
repeat serially, same server  ->   0 of 40 (Gemma 3, Gemma 4)
                                   4 of 40 (Qwen, the exception)
```

The serial-vs-serial control is what gives the first row meaning: serial execution is exactly
reproducible, so the 17 divergences are attributable to vLLM batching — grouping requests changes
floating-point summation order, which changes the token chosen at near-ties.

**It is not only phrasing.** On 3M's quick ratio, serial execution uses
`(current assets − inventories) / current liabilities` and batched execution uses
`(cash + investments + receivables) / current liabilities` — a different formula and a different
final number, landing directly on the variable being scored.

**All 4,500 generations were therefore run serially**, at 4.95 s/generation versus 1.43 s batched
(3.5×), ≈1.40 USD of additional GPU time. A fourth source sits in retrieval: recomputing the same
search on CPU (numpy) instead of GPU (torch) changed **1 question of 150**, one whose gold page
sits 5th or 6th with a 0.00008 score gap — so a 1–2 question difference between parsers is within
hardware noise, not only sampling noise.

**Practical consequence:** a run can only be reproduced exactly if the inference server is not
restarted in between.

---

## 8. Comparison with the original paper

The FinanceBench paper reports Table 2 over the same 150 questions, judged by human annotators.
Only its **Oracle** row (evidence pages supplied in the prompt) is comparable to anything measured
here; the retrieval rows are not, for reasons listed below.

```
Oracle / correct page supplied, humans judging on both sides
  paper : GPT-4-Turbo, 150 questions, expert financial annotators     85.3%   (128/22/0)
  here  : 3 open 4-bit AWQ models, 150 cases, non-specialist labels   84.7%
```

**With the evidence page supplied, three open-weight 4-bit models reach accuracy of the same order
as the figure the original paper reports for GPT-4-Turbo in its Oracle configuration. The
comparison is indicative, not strict.**

**Why it is not strict — four differences, all favouring the paper:**

1. **Who labels.** The paper uses expert financial annotators from the research team. Here, one
   non-specialist annotator, with 17 labels subsequently arbitrated by an LLM.
2. **Different categories.** The paper uses *correct / incorrect / did not answer*, with a
   documented rubric (minor deviations accepted; getting the number right while reasoning against
   the evidence does **not** count as correct). This study uses *correct / incorrect / unclear*,
   which is not the same partition.
3. **Depth of review.** One expert pass per case there; here a partial second pass over 31 of 150.
4. **The other rows do not transfer at all.** The paper's shared store indexes **360** documents
   against 84 here, and its 68% "did not answer" rate for GPT-4-Turbo measures caution, not
   accuracy — the models used here almost never decline. A 19% that includes a model choosing
   silence and a 47% from models that always answer are not measuring the same thing.

---

## Limitations

- **N=150** questions, of which **52** have machine-checkable numeric answers. Most contrasts are
  underpowered; *"not distinguishable from zero" is not "equal to zero"*.
- **~50 cases per generator** in the judge study. Pre-registered: a 15-point difference between
  models is detected only 46% of the time, so **between-model differences are treated as noise**
  unless they exceed 30 points. Global agreement is adequately powered (±5 to ±6 points).
- **One embedding model** (bge-m3). A different embedder could reorder the parsers.
- **Human labels come from one non-specialist annotator**, with 17 arbitrated by an LLM that is
  itself one of the judges under evaluation.
- **Two cases are declared contaminated**: the annotator consulted an external model before
  deciding. Agreement is reported both with and without them.
- **The dataset reference is not ground truth.** Beyond the Upjohn case, an AmEx answer of
  `$1.66⅔` penalized a parser for being right.
- **Accuracy at `k`=2, 3, 4 was not measured**; only `k`=1 (unpaired, gold page alone) and `k`=5.

---

## Pre-registration

Two decisions were written down before the corresponding data existed, to prevent post-hoc
rationalization:

- **The parser decision rule and the retrieval-ceiling prediction**, written into the stage-3
  report before generation: three named branches for the parser choice, plus the prediction that
  the end-to-end arm would show a null parser effect, with the reason (only 18 of 150 questions
  have any margin in which a parser could matter). Both held. *(Stage reports are Spanish-language
  and not published here — see [What is published here](#what-is-published-here-and-what-is-not).)*
- **[`financebench/PREREGISTRATION_JUDGE.md`](financebench/PREREGISTRATION_JUDGE.md)** — verdicts
  rather than RAGAS scores (to avoid introducing a threshold choice into the measurement); blind
  judging; the same 150 cases and no others; the power calculation quoted under Limitations;
  unclear cases reported separately rather than forced to a side; and the mandatory
  serial-vs-parallel judge control. Published in full.

---

## What is published here, and what is not

This repository carries the **minimum needed to check the judge results independently** — not the
full working tree.

| file | what it is |
|---|---|
| [`financebench/judge_vs_human_150.csv`](financebench/judge_vs_human_150.csv) | **the reusable artefact.** 150 rows: FinanceBench id, generator, parser variant, both human labellings, all three judges' verdicts, and which 2 cases are declared contaminated |
| [`financebench/reproduce_judge_tables.py`](financebench/reproduce_judge_tables.py) | regenerates every judge table in §4 and §6 from that CSV. No GPU, no API keys, no downloads — only numpy |
| [`financebench/PREREGISTRATION_JUDGE.md`](financebench/PREREGISTRATION_JUDGE.md) | the judge pre-registration, written before the judge was run |

```
python results/financebench/reproduce_judge_tables.py
```

**Deliberately not published**, to keep this to a reasonable size: the 4,500 generated answers, the
36,039 page extractions, the 84 source PDFs (redistribution is not ours to make — they come from
the [FinanceBench](https://github.com/patronus-ai/financebench) dataset), the 12,013 × 3 embedding
vectors, and the Spanish-language stage reports for parsing, corpus construction and `hit@k`.

**What that means for a reader:** the judge and gold-standard findings (§4, §6) are fully
recomputable from what is here. The generation and retrieval figures (§1, §2, §3) are **reported,
not reproducible from this repository** — reproducing those needs the raw answers and a GPU. Where
the distinction matters, it is stated at the point of the claim.

Per-question indicators were stored throughout the original work rather than aggregates alone, so
any paired interval can be recomputed without re-renting a GPU.

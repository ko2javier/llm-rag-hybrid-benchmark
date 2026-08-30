# Pre-registration — LLM judge vs human labelling

**Written 25 Aug 2026, BEFORE running the judge and before seeing any of its output.**

This exists for the same reason as the parser decision rule: if thresholds and criteria are chosen
after seeing the numbers, you end up picking the cut that looks best. That invalidates the result.

*Translated from the Spanish original (`PREREGISTRO_JUEZ.md`). Content unchanged; the two
paragraphs describing which specific local files hold the data were dropped as repo-irrelevant.*

---

## 1. What is measured

**Agreement between the LLM judge (Mistral Nemo 12B) and the human labelling**, over the **same
150 cases** already labelled by the human annotator (133 correct · 14 incorrect · 3 unclear).

Not one case more: anything the judge scores and the human did not is useless for comparison.

## 2. What the judge is asked for

**A verdict, not a continuous score.** Same material the human saw and the same question:
question + the dataset's analyst answer + the model's answer → `correct` / `incorrect` / `unclear`.

**The judge does not see** which model or which parser produced the answer, exactly like the human.

**Why a verdict and not RAGAS:** RAGAS returns four continuous 0–1 numbers. Comparing those to
binary labels forces a threshold choice, and that threshold is a new variable injected into the
very measurement being kept clean. A verdict answers exactly the question the human answered.

## 3. IF RAGAS is run anyway — threshold fixed HERE

If a continuous-metric comparison is also wanted, use **`answer_correctness`** (the only one that
compares the answer against the reference truth), with the cut fixed at:

```
answer_correctness >= 0.50  ->  counts as CORRECT
answer_correctness <  0.50  ->  counts as INCORRECT
```

**0.50 because it is the midpoint of the scale, not because it is convenient.** If other cuts are
explored afterwards, the full agreement-vs-threshold curve is reported — never a single threshold
picked after seeing the data.

## 4. What can and cannot be concluded — decided in advance

| question | answerable? | why |
|---|---|---|
| How much do judge and human agree overall? | **Yes** | 150 cases → ±5 to ±6 points |
| Does the judge penalise terse answers? | **Yes** | the short stratum is a census (81 of 81), no sampling error |
| Does the judge agree better with one generator than another? | **No** | ~50 per model: a 15-point difference is detected 46% of the time |

**Power, calculated 25 Aug before running anything** (50 vs 50 cases, base agreement 0.85):

```
true difference    probability of detecting it
  5 points                  11%
 10 points                  26%
 15 points                  46%
 20 points                  67%
 30 points                  93%
```

**Registered consequence:** if a between-model difference shows up in the results, it is treated as
**noise**, not a finding, unless it exceeds 30 points. This is written now precisely so it cannot
be rationalised later.

## 5. The 3 unclear cases

The human marked 3 as `unclear`. They are reported separately and **not forced to either side**.
Primary agreement is computed over the 147 with a clear verdict; the handling of the unclear ones
is reported as a note, not hidden.

## 6. Declared contamination

In 2 of the 150 cases the annotator consulted an external model before deciding. **The annotator's
own judgement prevailed both times.** Minimal impact, but it is declared because this step measures
exactly independent human judgement. Agreement excluding those 2 cases is also reported.

## 7. Additional control: is the judge reproducible?

On 25 Aug, batching was measured to change the *generator's* answers (17 of 40 in parallel, 23 of
40 after restarting the server). The RAGAS scores of the preceding phase were produced with 6
parallel processes and that was never checked.

The same control is run on the judge: **40 cases serially against the same 40 in parallel.** If the
verdicts diverge, that is a publishable finding on its own — *an LLM judge's score depends on which
batch it landed in* — and it would force a caveat on the 2,250 scores of the preceding phase.

---

## Outcome

All seven points above were honoured. The batch control returned **40/40 identical verdicts**: the
judge is reproducible under batching, so its limitation is criterion, not instability. The
between-model differences that appeared were below 30 points and are reported as noise, as
registered. See `../REPORT_FINANCEBENCH.md` §4 and §7.

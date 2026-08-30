#!/usr/bin/env python3
"""Reproduce every judge table in REPORT_FINANCEBENCH.md from judge_vs_human_150.csv.

    python reproduce_judge_tables.py

No GPU, no API keys, no downloads. Only numpy.

WHAT THIS CHECKS, AND WHY IT IS THE FIRST THING PRINTED
-------------------------------------------------------
133 of the 150 answers are correct, so a judge that approves everything unread already
scores 90.5% global agreement. Global agreement therefore cannot tell a good judge from
no judge at all, and it is the metric usually published for this purpose.

What does separate judges is how many of the *real* errors they catch. That number is
printed first, with the do-nothing baseline directly underneath.

THE TWO GOLD STANDARDS
----------------------
`human_original` is the independent human labelling (one annotator, blind to model and
parser). `human_revised` is the same labelling after auditing the 31 cases where any
judge disagreed; 17 labels changed, in both directions.

The revised standard was arbitrated by Claude, which is one of the three judges scored
here. Its verdicts therefore cannot be scored against it, and this script says so out
loud rather than printing a flattering number. See section 6 of the report.
"""
from __future__ import annotations

import csv
import os

import numpy as np

RNG = np.random.default_rng(20260830)
JUDGES = ["judge_nemo_12b", "judge_claude", "judge_gemini"]
ARBITER = "judge_claude"
HERE = os.path.dirname(os.path.abspath(__file__))


def boot(v: np.ndarray, n: int = 10_000) -> tuple[float, float, float]:
    if len(v) == 0:
        return 0.0, 0.0, 0.0
    idx = RNG.integers(0, len(v), size=(n, len(v)))
    m = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def table(rows: list[dict], gold: str, note: str = "") -> None:
    n_correct = sum(1 for r in rows if r[gold] == "correct")
    n_wrong = sum(1 for r in rows if r[gold] == "incorrect")
    n_unclear = sum(1 for r in rows if r[gold] == "unclear")
    clear = [r for r in rows if r[gold] in ("correct", "incorrect")]
    errors = [r for r in rows if r[gold] == "incorrect"]
    baseline = n_correct / len(clear)

    print("=" * 78)
    print(f"GOLD STANDARD: {gold}   ({n_correct} correct / {n_wrong} incorrect / {n_unclear} unclear)")
    if note:
        print(f"  {note}")
    print("=" * 78)
    print(f"  {'judge':<16}{'catches real errors':>28}{'global agreement':>24}")

    for j in JUDGES:
        caught = np.array([1.0 if r[j] == "incorrect" else 0.0 for r in errors])
        m, lo, hi = boot(caught)
        agree = np.array([1.0 if r[j] == r[gold] else 0.0 for r in clear])
        am, alo, ahi = boot(agree)
        flag = "  <- arbiter, not scorable here" if (gold == "human_revised" and j == ARBITER) else ""
        print(f"  {j:<16}{int(caught.sum()):>4}/{len(errors):<3} = {m:>6.1%} [{lo:.0%},{hi:.0%}]"
              f"{am:>14.1%} [{alo:.0%},{ahi:.0%}]{flag}")

    print(f"  {'approve all':<16}{0:>4}/{len(errors):<3} = {0.0:>6.1%}"
          f"{baseline:>14.1%}   <- looks at nothing")
    best = max(np.mean([1.0 if r[j] == r[gold] else 0.0 for r in clear]) for j in JUDGES)
    print(f"\n  best judge's margin over looking at nothing: {best - baseline:+.1%}")
    print(f"  range in error detection:                    0% -> {max(np.mean([1.0 if r[j]=='incorrect' else 0.0 for r in errors]) for j in JUDGES):.0%}")
    print()


def main() -> None:
    with open(os.path.join(HERE, "judge_vs_human_150.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"cases: {len(rows)}\n")

    table(rows, "human_original",
          "independent of all three judges -- this is the table published in the report")
    table(rows, "human_revised",
          "SENSITIVITY ONLY: arbitrated by one of the judges below")

    # ---------- how much the audit echoes each judge ----------
    changed = [r for r in rows if r["revised"] == "yes"]
    print("=" * 78)
    print("CIRCULARITY CHECK -- of the 17 revised labels, how many match what each")
    print("judge already said before the audit?")
    print("=" * 78)
    for j in JUDGES:
        hits = sum(1 for r in changed if r[j] == r["human_revised"])
        tag = "   <- the arbiter" if j == ARBITER else ""
        print(f"  {j:<16}{hits:>3}/{len(changed)} = {hits/len(changed):>5.0%}{tag}")
    print("\n  The arbiter agrees with itself far more than the others do. That is why its")
    print("  100% against the revised standard is an artefact, not a result.\n")

    # ---------- declared contamination ----------
    clean = [r for r in rows if r["declared_contaminated"] != "yes"
             and r["human_original"] in ("correct", "incorrect")]
    print("=" * 78)
    print(f"DECLARED CONTAMINATION -- 2 cases where the annotator consulted an external")
    print(f"model before deciding. Agreement excluding them ({len(clean)} cases):")
    print("=" * 78)
    for j in JUDGES:
        a = np.mean([1.0 if r[j] == r["human_original"] else 0.0 for r in clean])
        print(f"  {j:<16}{a:>8.1%}")

    # ---------- per generator, with the pre-registered warning ----------
    print()
    print("=" * 78)
    print("PER GENERATOR -- the pre-registration states this is NOT conclusive:")
    print("with ~50 cases per model, a 15-point difference is detected 46% of the time.")
    print("Treat any difference below 30 points as noise.")
    print("=" * 78)
    for g in sorted({r["generator"] for r in rows}):
        sub = [r for r in rows if r["generator"] == g
               and r["human_original"] in ("correct", "incorrect")]
        line = "  ".join(
            f"{j.replace('judge_',''):>9}={np.mean([1.0 if r[j]==r['human_original'] else 0.0 for r in sub]):.1%}"
            for j in JUDGES)
        print(f"  {g:<10} n={len(sub):>3}   {line}")


if __name__ == "__main__":
    main()

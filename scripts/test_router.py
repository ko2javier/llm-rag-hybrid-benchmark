"""
Test router.py against all deterministic questions (Q001-Q025) from the golden dataset.
Prints per-question results and saves to output/router_test_results.json.
"""

import json
import sys
import os

# Allow importing router from the same scripts/ directory
sys.path.insert(0, os.path.dirname(__file__))
from router import classify

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "golden_dataset.json")
OUTPUT_PATH  = os.path.join(os.path.dirname(__file__), "..", "output", "router_test_results.json")

with open(DATASET_PATH, encoding="utf-8") as f:
    dataset = json.load(f)

deterministic = [q for q in dataset if q["type"] == "deterministic"]

results = []
pass_count = 0
fail_count = 0

for item in deterministic:
    qid           = item["id"]
    expected_ft   = item["fact_type"]
    question      = item["question"]

    router_result = classify(question)
    rt = router_result["type"]
    rf = router_result.get("fact_type", None)

    passed = rt == "deterministic" and rf == expected_ft
    label  = "PASS" if passed else "FAIL"

    if passed:
        pass_count += 1
    else:
        fail_count += 1

    router_summary = f"type={rt}"
    if rf:
        router_summary += f", fact_type={rf}"

    print(f"{qid} | expected={expected_ft:<12} | router=({router_summary}) | {label}")

    results.append({
        "id":             qid,
        "question":       question,
        "expected_type":  "deterministic",
        "expected_ft":    expected_ft,
        "router_result":  router_result,
        "result":         label,
    })

print()
print(f"TOTAL: {pass_count} PASS, {fail_count} FAIL out of {len(deterministic)} questions")

output = {
    "summary": {
        "total":      len(deterministic),
        "pass":       pass_count,
        "fail":       fail_count,
    },
    "details": results,
}

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to {OUTPUT_PATH}")

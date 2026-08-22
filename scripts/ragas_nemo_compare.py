"""
Re-scores the full Fase 2 V1.0 RAG results (50 rows per model, 150 total) with Mistral Nemo 12B as
the RAGAS judge, served locally by vLLM -- no API, no rate limits, no third-party key.

Why Nemo 12B: the existing 3-judge table (POSTMORTEM.md H1) only has "Mistral 7B local" at
context_precision 0.963 and two frontier judges by API at 0.767 / 0.776. There is no intermediate
point. Nemo 12B keeps the FAMILY constant against the existing 7B judge and varies only SIZE, so a
monotonic drop would be evidence of a capacity effect rather than a Mistral idiosyncrasy.

Scores the SAME 150 rows as the other three judges (the base *_ragas.csv files) -- that is the
whole point; scoring different rows would produce a number comparable to nothing.

Runs in two places without edits:
  * on the Vast instance next to vLLM   -> --csv-dir /workspace/csv  (Langfuse absent, auto-skipped)
  * locally against an SSH tunnel       -> defaults below

AnswerRelevancy keeps strictness=1 and local embeddings to match ragas_deepseek_compare.py.
NOTE: the gpt-4o run used RAGAS's default strictness=3, so answer_relevancy is already only
loosely comparable across the existing table; context_precision/context_recall (the metrics H1
actually rests on) are unaffected by strictness.
"""

import argparse
import ast
import warnings
from pathlib import Path

import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall

warnings.filterwarnings("ignore", category=DeprecationWarning)

METRIC_COLS = (
    ("faithfulness", "ragas.faithfulness"),
    ("answer_relevancy", "ragas.answer_relevancy"),
    ("llm_context_precision_with_reference", "ragas.context_precision"),
    ("context_recall", "ragas.context_recall"),
)

parser = argparse.ArgumentParser()
parser.add_argument("--csv-dir", default=r"D:\LLM_Testing\Prueba2\Fase2_Resultados\01_CSV_Evaluacion")
parser.add_argument("--judge-url", default="http://localhost:8081/v1", help="vLLM OpenAI-compatible endpoint")
parser.add_argument("--judge-model", default="mistralai/Mistral-Nemo-Instruct-2407")
parser.add_argument("--embed-url", default=None,
                    help="OpenAI-compatible embedding endpoint (embed_server_batching.py, bge-m3). "
                         "When set, answer_relevancy matches the ORIGINAL Mistral-7B judge run "
                         "exactly. When omitted, falls back to local MiniLM + strictness=1 and "
                         "answer_relevancy is NOT comparable with the V1.0 numbers.")
parser.add_argument("--embed-model", default="BAAI/bge-m3")
parser.add_argument("--suffix", default="nemo12b", help="output filename + Langfuse score suffix")
parser.add_argument("--max-workers", type=int, default=16, help="no TPM cap on a local vLLM -- push it")
parser.add_argument("--no-langfuse", action="store_true")
args = parser.parse_args()

csv_dir = Path(args.csv_dir)

# Langfuse is best-effort: absent on the Vast instance, and the scores are fully reproducible from
# the output CSVs anyway. Never let it block a 2-hour judge run.
client = None
if not args.no_langfuse:
    try:
        import os

        from langfuse import Langfuse

        with open(Path(__file__).parent / ".env", encoding="utf-8") as f:
            for line in f:
                if line.startswith("LANGFUSE_INIT_PROJECT_PUBLIC_KEY="):
                    os.environ["LANGFUSE_PUBLIC_KEY"] = line.strip().split("=", 1)[1]
                elif line.startswith("LANGFUSE_INIT_PROJECT_SECRET_KEY="):
                    os.environ["LANGFUSE_SECRET_KEY"] = line.strip().split("=", 1)[1]
        os.environ.setdefault("LANGFUSE_HOST", "http://localhost:3000")
        candidate = Langfuse()
        if candidate.auth_check():
            client = candidate
        else:
            print("Langfuse reachable but auth failed -- continuing without it.")
    except Exception as exc:  # noqa: BLE001 -- any failure here is non-fatal by design
        print(f"Langfuse unavailable ({type(exc).__name__}) -- continuing without it.")

judge = LangchainLLMWrapper(ChatOpenAI(
    model=args.judge_model,
    temperature=0,
    api_key="EMPTY",  # vLLM ignores it, but the OpenAI client refuses to start without one
    base_url=args.judge_url,
    timeout=180,
    max_retries=5,
))
# POSTMORTEM E9 (a number published without controlling the variable): the V1.0 Mistral-7B judge
# was run by scripts/ragas_eval.py with bge-m3 embeddings over embed_server_batching.py and RAGAS's
# DEFAULT AnswerRelevancy (strictness=3). ragas_deepseek_compare.py deviated (MiniLM, strictness=1)
# because DeepSeek's API rejects n>1 -- a local vLLM has no such limit, so here we can reproduce the
# original configuration exactly. context_precision/context_recall/faithfulness are LLM-only and
# unaffected either way; answer_relevancy is affected by BOTH knobs, so this is the only path that
# makes it comparable with the V1.0 row.
if args.embed_url:
    judge_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        model=args.embed_model,
        base_url=args.embed_url,
        api_key="not-needed",   # local embed server
        tiktoken_enabled=False,  # embed_server_batching.py takes raw strings, not token ids
    ))
    answer_relevancy_metric = AnswerRelevancy(embeddings=judge_embeddings)  # strictness=3, the default
    print(f"answer_relevancy: bge-m3 @ {args.embed_url}, strictness=3 -- comparable with V1.0")
else:
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    answer_relevancy_metric = AnswerRelevancy(strictness=1, embeddings=judge_embeddings)
    print("answer_relevancy: MiniLM + strictness=1 -- NOT comparable with V1.0, report separately")

metrics = [
    Faithfulness(),
    answer_relevancy_metric,
    LLMContextPrecisionWithReference(),
    LLMContextRecall(),
]

# Two different CSV schemas feed this script:
#   * V1.0 *_ragas.csv           -- already in RAGAS's own column names
#   * V2.0 matrix CSVs           -- evaluator.py's names (question/generated_answer/...)
# Verified against both files on disk, not assumed: the context column is a serialized list in
# both cases (JSON double quotes in V2.0, Python repr single quotes in V1.0) -- literal_eval reads
# either.
SCHEMAS = {
    "ragas": {"user_input": "user_input", "retrieved_contexts": "retrieved_contexts",
              "response": "response", "reference": "reference"},
    "evaluator": {"user_input": "question", "retrieved_contexts": "retrieved_context",
                  "response": "generated_answer", "reference": "ideal_answer"},
}


def pick_schema(columns) -> dict[str, str]:
    for name, mapping in SCHEMAS.items():
        if all(c in columns for c in mapping.values()):
            return mapping
    raise SystemExit(f"Unrecognised CSV schema. Columns present: {sorted(columns)}")


csv_paths = sorted(p for p in csv_dir.glob("*.csv") if not p.stem.endswith("_full"))
if not csv_paths:
    raise SystemExit(f"No input CSV found in {csv_dir}")

for csv_path in csv_paths:
    model = csv_path.stem.replace("_ragas", "")
    df = pd.read_csv(csv_path)
    cols = pick_schema(df.columns)
    rows = []
    for _, row in df.iterrows():
        raw = row[cols["retrieved_contexts"]]
        try:
            contexts = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            contexts = [raw]
        rows.append({
            "user_input": row[cols["user_input"]],
            "retrieved_contexts": contexts,
            "response": row[cols["response"]],
            "reference": row[cols["reference"]],
        })

    print(f"=== {model}: scoring {len(rows)} rows with {args.judge_model} ===", flush=True)
    ds = EvaluationDataset.from_list(rows)
    result = evaluate(
        dataset=ds, metrics=metrics, llm=judge, show_progress=True, raise_exceptions=False,
        # No provider rate limit here -- the ceiling is vLLM's own scheduler. max_retries stays high
        # because a 12B judge fails RAGAS's structured-output parsing more often than a frontier one.
        run_config=RunConfig(max_workers=args.max_workers, max_retries=15, max_wait=90),
    )
    scored = result.to_pandas()

    if client is not None:
        for i, srow in scored.iterrows():
            trace_id = client.create_trace_id(seed=f"fase2-{model}-{i}")  # same seed as ingest.py -> same trace
            for metric_col, score_name in METRIC_COLS:
                if metric_col in srow and pd.notna(srow[metric_col]):
                    client.create_score(
                        trace_id=trace_id,
                        name=f"{score_name}.{args.suffix}_judge",
                        value=float(srow[metric_col]),
                        data_type="NUMERIC",
                        comment=f"judge: {args.judge_model} self-hosted via vLLM, full re-score (all 50 rows)",
                    )

    out_path = csv_dir / f"{model}_ragas_{args.suffix}_full.csv"
    scored.to_csv(out_path, index=False)
    print(f"  saved {out_path}")

    # Completeness report -- the run is only useful if it is complete, so surface gaps immediately
    # instead of discovering them at analysis time (see the gpt-4o run's 34% silent missingness).
    for metric_col, _ in METRIC_COLS:
        if metric_col in scored:
            missing = int(scored[metric_col].isna().sum())
            flag = "  <-- INCOMPLETE" if missing else ""
            print(f"  {metric_col:<40} missing {missing:>3}/{len(scored)}{flag}")

if client is not None:
    client.flush()
print("Done.")

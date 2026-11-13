"""Task 1-3 orchestration used by the notebook and scripts/run_all.py."""

from __future__ import annotations

import time

import mlflow
import pandas as pd
from mlflow.genai.scorers import scorer

from eduassist import config
from eduassist.llm_client import LLMClient
from eduassist.metrics import (
    is_valid_json,
    keyword_coverage,
    latency_summary,
    matches_schema,
    projected_monthly_cost,
    request_cost,
)

PROMPT_TEXT_COL = "response_text"


def render(template: str, question: str) -> str:
    return template.replace("{{question}}", question)


def run_questions(client: LLMClient, template: str, version: str,
                  questions: list[dict], delay: float) -> pd.DataFrame:
    """Send every question once with the given prompt template; one row per call."""
    rows = []
    for i, q in enumerate(questions):
        res = client.ask(render(template, q["question"]), q["id"], version)
        row = res.to_dict()
        row.update(request_cost(row["input_tokens"], row["output_tokens"],
                                client.s.price_input_per_m,
                                client.s.price_output_per_m))
        row["valid_json"] = is_valid_json(row[PROMPT_TEXT_COL])
        row["schema_valid"] = matches_schema(row[PROMPT_TEXT_COL])
        row["quality_keyword_coverage"] = keyword_coverage(
            row[PROMPT_TEXT_COL], q.get("keywords", []))
        rows.append(row)
        status = row["status"]
        print(f"  {q['id']} {version}: {status:<8} {row['latency_seconds']:.3f}s "
              f"in={row['input_tokens']} out={row['output_tokens']}")
        if i < len(questions) - 1 and delay > 0:
            time.sleep(delay)
    return pd.DataFrame(rows)


def latency_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["question_id", "prompt_version", "model_name", "start_time", "end_time",
            "latency_seconds", "status"]
    return df[cols]


def cost_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["question_id", "input_tokens", "output_tokens", "total_tokens",
            "input_cost", "output_cost", "total_cost"]
    return df[cols]


def summarise_version(df: pd.DataFrame, version: str) -> dict:
    ok = df[df["status"] == "success"]
    base = ok if len(ok) else df
    lat = latency_summary(base["latency_seconds"].tolist())
    avg_cost = float(base["total_cost"].mean())
    return {
        "prompt_version": version,
        "n_requests": int(len(df)),
        "success_rate": float(len(ok) / len(df)),
        "avg_latency_seconds": lat["mean"],
        "median_latency_seconds": lat["median"],
        "max_latency_seconds": lat["max"],
        "avg_input_tokens": float(base["input_tokens"].mean()),
        "avg_output_tokens": float(base["output_tokens"].mean()),
        "avg_total_tokens": float(base["total_tokens"].mean()),
        "avg_cost_per_request": avg_cost,
        "projected_monthly_cost_10k": projected_monthly_cost(
            avg_cost, config.MONTHLY_REQUESTS),
        "valid_json": float(df["valid_json"].mean()),
        "schema_valid_rate": float(df["schema_valid"].mean()),
        "quality_keyword_coverage": float(df["quality_keyword_coverage"].mean()),
    }


# ---------------- MLflow GenAI scorers (boolean feedback -> mean) ----------------
@scorer
def valid_json(outputs) -> bool:
    """True if the response is a JSON object."""
    return is_valid_json(str(outputs))


@scorer
def schema_valid(outputs) -> bool:
    """True if the response follows the exact EduAssist JSON schema."""
    return matches_schema(str(outputs))


@scorer
def keyword_quality(outputs, expectations) -> float:
    """Share of expected key concepts mentioned in the answer."""
    return keyword_coverage(str(outputs), expectations.get("keywords", []))


def evaluate_version(client: LLMClient, prompt_version_obj, label: str,
                     questions: list[dict], delay: float) -> tuple[dict, pd.DataFrame]:
    """One MLflow run per prompt version: calls, metrics, scorers, artifacts."""
    uri = f"prompts:/{config.PROMPT_NAME}/{prompt_version_obj.version}"
    with mlflow.start_run(run_name=f"prompt_{label}") as run:
        prompt = mlflow.genai.load_prompt(uri)  # loading inside the run links it
        df = run_questions(client, prompt.template, label, questions, delay)
        summary = summarise_version(df, label)
        summary["registry_version"] = int(prompt.version)
        mlflow.set_tags({"prompt_label": label, "prompt_uri": uri,
                         "task": "prompt_evaluation"})
        mlflow.log_params({**client.s.public_dict(), "prompt_name": config.PROMPT_NAME,
                           "prompt_version": prompt.version, "prompt_label": label,
                           "api_version": client.api_version,
                           "n_questions": len(questions)})
        mlflow.log_metrics({k: v for k, v in summary.items()
                            if isinstance(v, float)})
        data = [{"inputs": {"question": q["question"]},
                 "outputs": r[PROMPT_TEXT_COL],
                 "expectations": {"keywords": q.get("keywords", [])}}
                for q, r in zip(questions, df.to_dict("records"))]
        mlflow.genai.evaluate(data=data,
                              scorers=[valid_json, schema_valid, keyword_quality])
        mlflow.log_table(df, artifact_file=f"responses_{label}.json")
        summary["run_id"] = run.info.run_id
    return summary, df

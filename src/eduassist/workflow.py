"""End-to-end workflow shared by the notebook and scripts/run_all.py."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import mlflow
import pandas as pd

from eduassist import config
from eduassist.llm_client import LLMClient
from eduassist.metrics import latency_summary, projected_monthly_cost
from eduassist.pipeline import (
    cost_table,
    evaluate_version,
    latency_table,
    run_questions,
)
from eduassist.prompt_registry import (
    choose_best,
    promote,
    prompt_diff,
    register_all_prompts,
)

SUMMARY_COLS = ["prompt_version", "registry_version", "avg_latency_seconds",
                "avg_input_tokens", "avg_output_tokens", "avg_cost_per_request",
                "valid_json", "schema_valid_rate", "quality_keyword_coverage",
                "success_rate", "projected_monthly_cost_10k", "recommended", "run_id"]


def setup_mlflow(settings) -> None:
    mlflow.set_tracking_uri(settings.tracking_uri)
    mlflow.set_experiment(config.EXPERIMENT_NAME)


def task1_task2(client: LLMClient, questions: list[dict], label: str = "v1"):
    """Five requests with one prompt: latency table, stats, tokens and cost."""
    template = config.load_prompt_template(label)
    with mlflow.start_run(run_name="task1_task2_latency_cost"):
        df = run_questions(client, template, label, questions, client.s.request_delay)
        ok = df[df["status"] == "success"]
        stats = latency_summary(ok["latency_seconds"].tolist())
        avg_cost = float(ok["total_cost"].mean())
        monthly = projected_monthly_cost(avg_cost, config.MONTHLY_REQUESTS)
        mlflow.set_tags({"task": "task1_task2", "prompt_label": label})
        mlflow.log_params({**client.s.public_dict(), "api_version": client.api_version})
        mlflow.log_metrics({f"latency_{k}_seconds": v for k, v in stats.items()})
        mlflow.log_metrics({"avg_cost_per_request": avg_cost,
                            "projected_monthly_cost_10k": monthly,
                            "avg_input_tokens": float(ok["input_tokens"].mean()),
                            "avg_output_tokens": float(ok["output_tokens"].mean())})
        config.OUTPUTS_DIR.mkdir(exist_ok=True)
        out = df.drop(columns=["response_text"])
        out.to_csv(config.OUTPUTS_DIR / "latency_tokens_cost.csv", index=False)
        pd.DataFrame([stats]).to_csv(config.OUTPUTS_DIR / "latency_summary.csv",
                                     index=False)
        mlflow.log_artifact(str(config.OUTPUTS_DIR / "latency_tokens_cost.csv"))
    slowest = df.loc[df["latency_seconds"].idxmax()]
    return {"requests": df, "latency_table": latency_table(df),
            "cost_table": cost_table(df), "latency_stats": stats,
            "avg_cost": avg_cost, "monthly_cost": monthly, "slowest": slowest}


def task3(client: LLMClient, questions: list[dict]):
    """Register prompts, evaluate each version, compare, promote the best."""
    registered = register_all_prompts()
    diffs = {}
    labels = list(config.PROMPT_VERSIONS)
    for a, b in zip(labels, labels[1:]):
        d = prompt_diff(registered[a].template, registered[b].template,
                        f"{a} (registry v{registered[a].version})",
                        f"{b} (registry v{registered[b].version})")
        (config.OUTPUTS_DIR / f"prompt_diff_{a}_{b}.txt").write_text(d + "\n")
        diffs[f"{a}->{b}"] = d

    summaries, details = [], []
    for label in labels:
        print(f"Evaluating prompt {label} ...")
        summary, df = evaluate_version(client, registered[label], label, questions,
                                       client.s.request_delay)
        summaries.append(summary)
        details.append(df)

    best = choose_best(summaries)
    promote(best["registry_version"])
    for s in summaries:
        is_best = s["prompt_version"] == best["prompt_version"]
        s["recommended"] = "yes" if is_best else "no"
        with mlflow.start_run(run_id=s["run_id"]):
            mlflow.set_tag("recommended", s["recommended"])

    table = pd.DataFrame(summaries)[SUMMARY_COLS]
    table.to_csv(config.OUTPUTS_DIR / "prompt_evaluation_results.csv", index=False)
    detail = pd.concat(details, ignore_index=True)
    detail.to_csv(config.OUTPUTS_DIR / "prompt_evaluation_requests.csv", index=False)
    prod = mlflow.genai.load_prompt(f"prompts:/{config.PROMPT_NAME}@production")
    return {"registered": registered, "diffs": diffs, "table": table,
            "detail": detail, "best": best, "production_version": prod.version}


def write_metadata(client: LLMClient) -> dict:
    meta = {**client.s.public_dict(), "api_version": client.api_version,
            "mlflow_version": mlflow.__version__,
            "experiment": config.EXPERIMENT_NAME, "prompt_name": config.PROMPT_NAME,
            "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (config.OUTPUTS_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    return meta

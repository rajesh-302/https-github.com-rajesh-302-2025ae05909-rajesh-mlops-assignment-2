"""Register, load and promote prompt versions in the MLflow Prompt Registry."""

from __future__ import annotations

import difflib

import mlflow

from eduassist import config


def register_all_prompts(name: str = config.PROMPT_NAME) -> dict:
    """Register v1, v2, v3 (in order). Re-running does not create duplicates."""
    existing = {}
    for v in range(1, 50):
        try:
            p = mlflow.genai.load_prompt(f"prompts:/{name}/{v}")
        except Exception:
            break
        existing[p.template.strip()] = p
    registered = {}
    for label, spec in config.PROMPT_VERSIONS.items():
        template = config.load_prompt_template(label)
        if template in existing:
            registered[label] = existing[template]
            continue
        registered[label] = mlflow.genai.register_prompt(
            name=name, template=template, commit_message=spec["commit_message"],
            tags={**spec["tags"], "label": label, "project": "EduAssist AI"},
        )
    return registered


def prompt_diff(a: str, b: str, label_a: str, label_b: str) -> str:
    return "\n".join(difflib.unified_diff(
        a.splitlines(), b.splitlines(), fromfile=label_a, tofile=label_b,
        lineterm=""))


def choose_best(summary_rows: list[dict]) -> dict:
    """Promotion rule.

    1. EduAssist needs machine-readable output -> prefer the highest valid_json
       (a version must reach 1.0 to be production-ready).
    2. Then highest answer quality (keyword coverage).
    3. Then lowest average cost, then lowest average latency.
    """
    return sorted(
        summary_rows,
        key=lambda r: (-r["valid_json"], -r["schema_valid_rate"],
                       -r["quality_keyword_coverage"], r["avg_cost_per_request"],
                       r["avg_latency_seconds"]),
    )[0]


def promote(version: int, name: str = config.PROMPT_NAME,
            alias: str = config.PRODUCTION_ALIAS) -> None:
    mlflow.genai.set_prompt_alias(name, alias=alias, version=version)

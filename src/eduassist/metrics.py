"""Latency statistics, token cost economics and output-quality checks."""

from __future__ import annotations

import json
import statistics

ALLOWED_CONFIDENCE = {"high", "medium", "low"}
REQUIRED_KEYS = {"answer", "confidence", "key_points"}


def latency_summary(latencies: list[float]) -> dict:
    vals = [float(v) for v in latencies]
    return {
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
    }


def request_cost(input_tokens: int, output_tokens: int,
                 price_in_per_m: float, price_out_per_m: float) -> dict:
    """Input/Output/Total cost of one request (formula from the assignment)."""
    input_cost = input_tokens / 1_000_000 * price_in_per_m
    output_cost = output_tokens / 1_000_000 * price_out_per_m
    return {"input_cost": input_cost, "output_cost": output_cost,
            "total_cost": input_cost + output_cost}


def projected_monthly_cost(avg_cost_per_request: float,
                           requests_per_month: int = 10_000) -> float:
    return avg_cost_per_request * requests_per_month


def parse_json(text: str):
    """Strict parse: the WHOLE response must be a JSON object (no code fences)."""
    try:
        obj = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError, AttributeError):
        return None
    return obj if isinstance(obj, dict) else None


def is_valid_json(text: str) -> bool:
    return parse_json(text) is not None


def matches_schema(text: str) -> bool:
    """Valid JSON AND exactly the EduAssist schema."""
    obj = parse_json(text)
    if obj is None or set(obj) != REQUIRED_KEYS:
        return False
    return (isinstance(obj["answer"], str) and obj["answer"].strip() != ""
            and obj["confidence"] in ALLOWED_CONFIDENCE
            and isinstance(obj["key_points"], list) and len(obj["key_points"]) > 0
            and all(isinstance(k, str) for k in obj["key_points"]))


def answer_text(text: str) -> str:
    """Text used for quality scoring: JSON answer+key_points, else the raw text."""
    obj = parse_json(text)
    if obj is not None:
        points = obj.get("key_points", [])
        points = points if isinstance(points, list) else [str(points)]
        return " ".join([str(obj.get("answer", ""))] + [str(p) for p in points])
    return text or ""


def keyword_coverage(text: str, keyword_groups: list[list[str]]) -> float:
    """Share of expected concepts mentioned (each group = list of synonyms)."""
    if not keyword_groups:
        return 0.0
    low = answer_text(text).lower()
    hits = sum(any(k.lower() in low for k in group) for group in keyword_groups)
    return hits / len(keyword_groups)

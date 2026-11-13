"""Configuration: paths, prompt definitions and settings read from .env / env vars.

Secrets are read from environment variables only (optionally loaded from a local,
git-ignored .env file). Nothing secret is ever printed or logged.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "questions.json"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

EXPERIMENT_NAME = "eduassist_llmops_assignment"
PROMPT_NAME = "eduassist-academic-qa"
PRODUCTION_ALIAS = "production"
MONTHLY_REQUESTS = 10_000

PROMPT_VERSIONS = {
    "v1": {
        "file": "v1_basic.txt",
        "commit_message": "v1 basic: minimal instruction, free-text answer (baseline)",
        "tags": {"style": "basic", "output_format": "free_text"},
    },
    "v2": {
        "file": "v2_structured.txt",
        "commit_message": "v2 structured: academic-assistant role, concise, "
        "undergraduate-level language",
        "tags": {"style": "structured", "output_format": "free_text"},
    },
    "v3": {
        "file": "v3_json.txt",
        "commit_message": "v3 json: strict JSON output with answer, confidence, "
        "key_points schema for downstream parsing",
        "tags": {"style": "json", "output_format": "json"},
    },
}

DEFAULT_BASE_URLS = {
    "openai": None,
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "groq": "https://api.groq.com/openai/v1",
    "ollama": "http://localhost:11434/v1",
    "mock": None,
}
KEY_FALLBACKS = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
}


def load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines). Existing env vars take priority."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    provider: str
    model: str
    base_url: str | None
    temperature: float
    max_tokens: int
    request_delay: float
    price_input_per_m: float
    price_output_per_m: float
    currency: str
    price_date: str
    price_source: str
    tracking_uri: str

    def api_key(self) -> str:
        """Return the API key from the environment (never stored on the object)."""
        if self.provider == "mock":
            return "mock"
        if self.provider == "ollama":
            return os.getenv("LLM_API_KEY") or "ollama"
        fallback = KEY_FALLBACKS.get(self.provider, "")
        key = os.getenv("LLM_API_KEY") or (os.getenv(fallback) if fallback else None)
        if not key:
            raise RuntimeError(
                "No API key found. Put LLM_API_KEY=... in the .env file "
                "(never in the code or notebook)."
            )
        return key

    def public_dict(self) -> dict:
        """Settings that are safe to log (no secrets)."""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url or "provider default",
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "price_input_per_m": self.price_input_per_m,
            "price_output_per_m": self.price_output_per_m,
            "currency": self.currency,
            "price_date": self.price_date,
            "price_source": self.price_source,
        }


def get_settings() -> Settings:
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider not in DEFAULT_BASE_URLS:
        raise ValueError(f"Unknown LLM_PROVIDER={provider!r}")
    return Settings(
        provider=provider,
        model=os.getenv("LLM_MODEL", "gpt-4o-mini").strip(),
        base_url=(os.getenv("LLM_BASE_URL", "").strip()
                  or DEFAULT_BASE_URLS[provider]),
        temperature=_float("LLM_TEMPERATURE", 0.0),
        max_tokens=int(_float("LLM_MAX_TOKENS", 400)),
        request_delay=_float("LLM_REQUEST_DELAY", 2.0),
        price_input_per_m=_float("PRICE_INPUT_PER_M", 0.0),
        price_output_per_m=_float("PRICE_OUTPUT_PER_M", 0.0),
        currency=os.getenv("PRICE_CURRENCY", "USD").strip(),
        price_date=os.getenv("PRICE_DATE", "not set").strip(),
        price_source=os.getenv("PRICE_SOURCE_URL", "not set").strip(),
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000").strip(),
    )


def load_questions(path: Path = DATA_FILE) -> list[dict]:
    return json.loads(path.read_text())


def load_prompt_template(version: str) -> str:
    return (PROMPTS_DIR / PROMPT_VERSIONS[version]["file"]).read_text().strip()

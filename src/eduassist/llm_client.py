"""Call an OpenAI-compatible chat API and measure latency + token usage."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from eduassist.config import Settings


@dataclass
class LLMResult:
    question_id: str
    prompt_version: str
    model_name: str
    start_time: str
    end_time: str
    latency_seconds: float
    status: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    attempts: int
    response_text: str
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class LLMClient:
    """Thin wrapper so every request is timed and its usage recorded the same way."""

    def __init__(self, settings: Settings, max_retries: int = 2):
        self.s = settings
        self.max_retries = max_retries
        self._client = None
        if settings.provider != "mock":
            import openai

            self.sdk_version = openai.__version__
            self._client = openai.OpenAI(
                api_key=settings.api_key(), base_url=settings.base_url,
                max_retries=0, timeout=60.0,
            )
        else:
            self.sdk_version = "mock"

    @property
    def api_version(self) -> str:
        return f"{self.s.provider} chat.completions (openai-python {self.sdk_version})"

    # ------------------------------------------------------------------
    def _request(self, prompt: str):
        kwargs = dict(model=self.s.model,
                      messages=[{"role": "user", "content": prompt}],
                      temperature=self.s.temperature)
        try:
            return self._client.chat.completions.create(
                max_tokens=self.s.max_tokens, **kwargs)
        except Exception as exc:  # newer OpenAI models renamed the parameter
            if "max_completion_tokens" not in str(exc):
                raise
            return self._client.chat.completions.create(
                max_completion_tokens=self.s.max_tokens, **kwargs)

    def _mock(self, prompt: str, version: str):
        """Offline deterministic stand-in used by tests / CI (never for results)."""
        time.sleep(0.01)
        if version == "v3":
            text = json.dumps({"answer": "Mock answer about labels and energy.",
                               "confidence": "medium",
                               "key_points": ["mock point one", "mock point two"]})
        else:
            text = "Mock answer: labeled data, clusters, force, mass, ATP energy."
        n_in = max(1, len(prompt) // 4)
        n_out = max(1, len(text) // 4)
        return text, n_in, n_out, "mock-model"

    # ------------------------------------------------------------------
    def ask(self, prompt: str, question_id: str, prompt_version: str) -> LLMResult:
        attempts, error = 0, ""
        while True:
            attempts += 1
            start_iso = _now()
            t0 = time.perf_counter()
            try:
                if self._client is None:
                    text, n_in, n_out, model = self._mock(prompt, prompt_version)
                else:
                    resp = self._request(prompt)
                    text = resp.choices[0].message.content or ""
                    usage = resp.usage
                    n_in = int(getattr(usage, "prompt_tokens", 0) or 0)
                    n_out = int(getattr(usage, "completion_tokens", 0) or 0)
                    model = resp.model or self.s.model
                latency = time.perf_counter() - t0
                return LLMResult(question_id, prompt_version, model, start_iso, _now(),
                                 round(latency, 4), "success", n_in, n_out,
                                 n_in + n_out, attempts, text)
            except Exception as exc:
                latency = time.perf_counter() - t0
                code = getattr(exc, "status_code", None)
                error = f"{type(exc).__name__}: {str(exc)[:200]}"
                retryable = code in (429, 500, 502, 503, 504) or code is None
                if attempts <= self.max_retries and retryable:
                    time.sleep(5 * attempts)
                    continue
                status = f"error_{code or 'network'}"
                return LLMResult(question_id, prompt_version, self.s.model, start_iso,
                                 _now(), round(latency, 4), status,
                                 0, 0, 0, attempts, "", error)

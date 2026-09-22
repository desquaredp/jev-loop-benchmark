"""Environment-only configuration. Secrets are never written to run artifacts."""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=False)


def _float(name, default):
    return float(os.getenv(name, default))


def _int(name, default):
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Model:
    provider: str
    name: str
    input_price: float
    cached_price: float
    output_price: float

    @property
    def prices(self):
        return {
            "input": self.input_price,
            "cached": self.cached_price,
            "output": self.output_price,
        }


@dataclass(frozen=True)
class Settings:
    cheap: Model
    strong: Model
    reasoning_effort: str
    max_completion_tokens: int
    cheap_budget: float
    jev_budget: float
    strong_budget: float
    max_parallel_runs: int
    max_parallel_graders: int
    jev_model: str
    jev_url: str
    jev_input_price: float

    def validate(self, *, paid=False):
        for role, model in (("cheap", self.cheap), ("strong", self.strong)):
            if model.provider not in {"openai", "anthropic"}:
                raise ValueError(f"{role.upper()}_PROVIDER must be openai or anthropic")
            if not model.name:
                raise ValueError(f"{role.upper()}_MODEL is required")
            if paid and model.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY is required by the configured models")
            if paid and model.provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
                raise ValueError("ANTHROPIC_API_KEY is required by the configured models")
        if paid and not jev_api_key():
            raise ValueError("JEV_API_KEY is required for the JEV-led arm")


def _model(prefix, defaults):
    return Model(
        provider=os.getenv(f"{prefix}_PROVIDER", defaults[0]).strip().lower(),
        name=os.getenv(f"{prefix}_MODEL", defaults[1]).strip(),
        input_price=_float(f"{prefix}_INPUT_USD_PER_MILLION", defaults[2]),
        cached_price=_float(f"{prefix}_CACHED_INPUT_USD_PER_MILLION", defaults[3]),
        output_price=_float(f"{prefix}_OUTPUT_USD_PER_MILLION", defaults[4]),
    )


def load_settings():
    return Settings(
        cheap=_model("CHEAP", ("openai", "gpt-5.6-luna", 0.20, 0.02, 1.20)),
        strong=_model("STRONG", ("openai", "gpt-6-astra", 10.0, 1.0, 50.0)),
        reasoning_effort=os.getenv("REASONING_EFFORT", "medium"),
        max_completion_tokens=_int("MAX_COMPLETION_TOKENS", 8000),
        cheap_budget=_float("CHEAP_BUDGET_USD", 3),
        jev_budget=_float("JEV_BUDGET_USD", 3),
        strong_budget=_float("STRONG_BUDGET_USD", 12),
        max_parallel_runs=_int("MAX_PARALLEL_RUNS", 15),
        max_parallel_graders=_int("MAX_PARALLEL_GRADERS", 2),
        jev_model=os.getenv("JEV_MODEL", "jev-1.13.0"),
        jev_url=os.getenv("JEV_URL", "https://api.typesafe.ai/v1/systemone"),
        jev_input_price=_float("JEV_INPUT_USD_PER_MILLION", 0.042),
    )


def jev_api_key():
    return os.getenv("JEV_API_KEY") or os.getenv("TYPESAFE_API_KEY")

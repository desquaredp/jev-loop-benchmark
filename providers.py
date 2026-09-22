"""Minimal OpenAI and Anthropic JSON adapters with normalized usage receipts."""

from __future__ import annotations

import json
import os
import time


class ProviderError(RuntimeError):
    pass


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3]
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise
        value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise json.JSONDecodeError("Expected a JSON object", text, 0)
    return value


class Gateway:
    def __init__(self):
        self._openai = None
        self._anthropic = None

    def complete_json(self, *, provider, model, messages, reasoning_effort, max_tokens):
        retries = max(0, int(os.getenv("PROVIDER_MAX_RETRIES", "3")))
        for attempt in range(retries + 1):
            try:
                if provider == "openai":
                    decision, raw = self._openai_json(model, messages, reasoning_effort, max_tokens)
                elif provider == "anthropic":
                    decision, raw = self._anthropic_json(model, messages, max_tokens)
                else:
                    raise ProviderError(f"Unsupported provider: {provider}")
                raw["transport_retries"] = attempt
                return decision, raw
            except ProviderError:
                raise
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                body = getattr(exc, "body", None) or {}
                error = body.get("error", body) if isinstance(body, dict) else {}
                code = getattr(exc, "code", None) or (error.get("code") if isinstance(error, dict) else None)
                transient = status in {408, 409, 429} or (isinstance(status, int) and status >= 500)
                transient = transient or type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}
                if code in {"insufficient_quota", "credit_balance_exhausted"}:
                    transient = False
                if not transient or attempt >= retries:
                    raise ProviderError(f"{provider} request failed after {attempt} retries: {exc}") from exc
                time.sleep(min(8, 2 ** (attempt + 1)))
        raise AssertionError("unreachable")

    def _openai_json(self, model, messages, reasoning_effort, max_tokens):
        if self._openai is None:
            from openai import OpenAI
            kwargs = {"api_key": os.environ["OPENAI_API_KEY"], "max_retries": 0, "timeout": 120}
            if os.getenv("OPENAI_BASE_URL"):
                kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
            self._openai = OpenAI(**kwargs)
        kwargs = {
            "model": model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "max_completion_tokens": max_tokens,
            "service_tier": "default",
        }
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        response = self._openai.chat.completions.create(**kwargs)
        raw = response.model_dump(mode="json")
        try:
            decision = parse_json(response.choices[0].message.content)
        except json.JSONDecodeError as exc:
            raw["parse_error"] = str(exc)
            decision = {"action": "invalid", "args": {}, "note": "Provider returned invalid JSON."}
        return decision, raw

    def _anthropic_json(self, model, messages, max_tokens):
        if self._anthropic is None:
            from anthropic import Anthropic
            kwargs = {"api_key": os.environ["ANTHROPIC_API_KEY"], "max_retries": 0, "timeout": 120}
            if os.getenv("ANTHROPIC_BASE_URL"):
                kwargs["base_url"] = os.environ["ANTHROPIC_BASE_URL"]
            self._anthropic = Anthropic(**kwargs)
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        conversation = [m for m in messages if m["role"] != "system"]
        response = self._anthropic.messages.create(
            model=model,
            system=system,
            messages=conversation,
            max_tokens=max_tokens,
        )
        text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        dumped = response.model_dump(mode="json")
        usage = dumped.get("usage") or {}
        dumped["usage"] = {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "prompt_tokens_details": {
                "cached_tokens": usage.get("cache_read_input_tokens", 0),
                "cache_creation_tokens": usage.get("cache_creation_input_tokens", 0),
            },
            "provider_usage": usage,
        }
        try:
            decision = parse_json(text)
        except json.JSONDecodeError as exc:
            dumped["parse_error"] = str(exc)
            decision = {"action": "invalid", "args": {}, "note": "Provider returned invalid JSON."}
        return decision, dumped

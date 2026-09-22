import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import benchmark
import config
import loop
import providers


class FakeOpenAIResponse:
    class Choice:
        class Message:
            content = '{"action":"finish","args":{}}'
        message = Message()

    choices = [Choice()]

    def model_dump(self, mode="json"):
        return {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 2},
            }
        }


class PackageTests(unittest.TestCase):
    def test_json_parser_accepts_plain_and_fenced_objects(self):
        self.assertEqual(providers.parse_json('{"x":1}'), {"x": 1})
        self.assertEqual(providers.parse_json('```json\n{"x":2}\n```'), {"x": 2})

    def test_cost_uses_normalized_cache_tokens(self):
        cost = loop.api_cost({
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 2},
        }, {"input": 1, "cached": .1, "output": 2})
        self.assertAlmostEqual(cost, (8 + .2 + 10) / 1_000_000)

    def test_openai_gateway_returns_json_and_normalized_receipt(self):
        gateway = providers.Gateway()
        fake = type("Client", (), {})()
        fake.chat = type("Chat", (), {})()
        fake.chat.completions = type("Completions", (), {
            "create": staticmethod(lambda **kwargs: FakeOpenAIResponse())
        })()
        gateway._openai = fake
        value, raw = gateway.complete_json(
            provider="openai", model="model", messages=[{"role": "user", "content": "x"}],
            reasoning_effort="medium", max_tokens=10,
        )
        self.assertEqual(value["action"], "finish")
        self.assertEqual(raw["usage"]["prompt_tokens"], 10)

    def test_gateway_retries_rate_limit_and_records_it(self):
        gateway = providers.Gateway()
        error = type("RateLimit", (Exception,), {"status_code": 429})()
        responses = iter([error, ({"action": "finish", "args": {}}, {"usage": {}})])

        def call(*args):
            value = next(responses)
            if isinstance(value, Exception):
                raise value
            return value

        with patch.object(gateway, "_openai_json", side_effect=call), patch.object(providers.time, "sleep"):
            _, raw = gateway.complete_json(
                provider="openai", model="model", messages=[],
                reasoning_effort=None, max_tokens=10,
            )
        self.assertEqual(raw["transport_retries"], 1)

    def test_gateway_does_not_retry_exhausted_credit(self):
        gateway = providers.Gateway()
        error = type("Quota", (Exception,), {
            "status_code": 429,
            "body": {"error": {"code": "credit_balance_exhausted"}},
        })()
        with patch.object(gateway, "_openai_json", side_effect=error) as call, patch.object(providers.time, "sleep"):
            with self.assertRaises(providers.ProviderError):
                gateway.complete_json(
                    provider="openai", model="model", messages=[],
                    reasoning_effort=None, max_tokens=10,
                )
        self.assertEqual(call.call_count, 1)

    def test_parallel_grader_ids_are_path_specific(self):
        first = loop.evaluation_run_id("runs/x/task/cheap/grade-01", "official")
        second = loop.evaluation_run_id("runs/x/task/strong/grade-01", "official")
        self.assertNotEqual(first, second)

    def test_configuration_contains_no_secret_values(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "secret-openai-value",
            "JEV_API_KEY": "secret-jev-value",
        }, clear=False):
            settings = config.load_settings()
            rendered = json.dumps(benchmark._model_config(settings))
        self.assertNotIn("secret-openai-value", rendered)
        self.assertNotIn("secret-jev-value", rendered)

    def test_comparison_cell_has_required_fields(self):
        result = {
            "resolved": True,
            "status": "passed",
            "cost": .1234,
            "calls": {"cheap": 2, "strong": 1, "jev": 7},
            "agent_work_seconds": 125,
        }
        self.assertEqual(
            benchmark._cell(result, "jev_cascade"),
            "Pass · $0.123 · 2/1/7 calls · 2:05 agent",
        )


if __name__ == "__main__":
    unittest.main()

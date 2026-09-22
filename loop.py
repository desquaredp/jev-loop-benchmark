"""Shared filesystem, Docker, grading, and model-call helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import uuid


ROOT = Path(__file__).resolve().parent
DATASET = "princeton-nlp/SWE-bench_Verified"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, default=str) + "\n")


def command(args, *, stdin=None, timeout=180, cwd=None):
    process = subprocess.run(
        args,
        input=stdin,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
    )
    return {
        "exit_code": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }


def require_ok(result):
    if result["exit_code"] != 0:
        raise RuntimeError((result["stderr"] + result["stdout"])[-3000:])
    return result["stdout"]


class Box:
    """A fresh, network-disabled benchmark checkout."""

    def __init__(self, task):
        self.task = task
        self.name = "jev-loop-" + uuid.uuid4().hex[:10]

    def __enter__(self):
        require_ok(
            command(
                [
                    "docker", "run", "-d", "--platform", "linux/amd64",
                    "--network", "none", "--memory", "4g", "--cpus", "2",
                    "--name", self.name, self.task["image"], "sleep", "infinity",
                ],
                timeout=180,
            )
        )
        try:
            require_ok(self.shell("git reset --hard " + shlex.quote(self.task["base_commit"])))
            head = require_ok(self.shell("git rev-parse HEAD")).strip()
            if head != self.task["base_commit"]:
                raise RuntimeError(f"Image commit mismatch: {head}")
            dirty = require_ok(self.shell("git diff --name-only")).strip()
            if dirty:
                raise RuntimeError(f"Image has tracked modifications: {dirty}")
        except Exception:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc):
        command(["docker", "rm", "-f", self.name], timeout=30)

    def shell(self, script, stdin=None, timeout=90):
        script = "source /opt/miniconda3/bin/activate testbed && cd /testbed && " + script
        return command(
            ["docker", "exec", "-i", self.name, "bash", "-lc", script],
            stdin=stdin,
            timeout=timeout,
        )

    def diff(self):
        return require_ok(self.shell("git diff --binary HEAD"))


def grade(out, task, patch, label):
    """Run the official SWE-bench evaluator and preserve every receipt."""
    directory = Path(out) / label
    directory.mkdir(exist_ok=False)
    run_id = evaluation_run_id(out, label)
    append(directory / "predictions.jsonl", {
        "instance_id": task["id"],
        "model_name_or_path": run_id,
        "model_patch": patch,
    })
    dataset = Path(task.get("grading_dataset", DATASET))
    if not dataset.is_absolute() and str(dataset) != DATASET:
        dataset = ROOT / dataset
    timeout = int(task.get("grading_timeout_seconds", 180))
    args = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", str(dataset), "--split", "test",
        "--instance_ids", task["id"],
        "--predictions_path", str((directory / "predictions.jsonl").resolve()),
        "--run_id", run_id, "--max_workers", "1", "--namespace", "swebench",
        "--cache_level", "instance", "--timeout", str(timeout),
    ]
    with (directory / "grader.log").open("w") as log:
        completed = subprocess.run(
            args, cwd=directory, stdout=log, stderr=subprocess.STDOUT,
            timeout=timeout + 120,
        )
    report = directory / "logs/run_evaluation" / run_id / run_id / task["id"] / "report.json"
    # SWE-bench can finish and write a valid per-instance report, then fail
    # during unrelated Docker image cleanup. The report is authoritative.
    if completed.returncode and not report.exists():
        detail = (directory / "grader.log").read_text(errors="replace")[-3000:]
        raise RuntimeError(f"Official evaluator exited {completed.returncode}:\n{detail}")


def api_cost(usage, price):
    prompt = int(usage.get("prompt_tokens") or 0)
    output = int(usage.get("completion_tokens") or 0)
    details = usage.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0)
    writes = int(details.get("cache_write_tokens") or details.get("cache_creation_tokens") or 0)
    if cached + writes > prompt:
        raise ValueError("Unexpected cache accounting")
    return (
        (prompt - cached - writes) * price["input"]
        + cached * price["cached"]
        + writes * price["input"] * 1.25
        + output * price["output"]
    ) / 1_000_000


def _reserved_tokens(messages, task):
    raw = json.dumps(messages)
    if not task.get("token_reservation"):
        return len(raw.encode()) + 1024
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("o200k_base")
        measured = sum(
            len(encoding.encode(message["content"], disallowed_special=()))
            for message in messages
        )
        return int(measured * 1.25) + 1024
    except ImportError:
        return len(raw.encode()) + 1024


def call_model(gateway, out, task, arm, model, payload, system):
    """Make one stateless provider call with harness-reconstructed task state."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload)},
    ]
    ledger = Path(out) / "calls.jsonl"
    spent = sum(row["cost"] for row in jsonl(ledger)) if ledger.exists() else 0
    price = task["prices"][model]
    reserved_input = _reserved_tokens(messages, task)
    bound = (
        reserved_input * price["input"] * 1.25
        + task["max_completion_tokens"] * price["output"]
    ) / 1_000_000
    if spent + bound > task["budget_usd"]:
        raise RuntimeError("Arm budget would be exceeded before the next model call")
    provider = task["providers"][model]
    decision, raw = gateway.complete_json(
        provider=provider,
        model=model,
        messages=messages,
        reasoning_effort=task.get("reasoning_effort"),
        max_tokens=task["max_completion_tokens"],
    )
    cost = api_cost(raw["usage"], price)
    append(ledger, {
        "arm": arm,
        "provider": provider,
        "model": model,
        "cost": cost,
        "request": messages,
        "response": raw,
        "reserved_cost": bound,
    })
    return decision, cost


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluation_run_id(out, label):
    """A deterministic ID unique to a task/arm/grade path for parallel Docker runs."""
    path = Path(out)
    try:
        scope = str(path.resolve().relative_to(ROOT))
    except ValueError:
        scope = str(path.resolve())
    suffix = hashlib.sha256((scope + ":" + label).encode()).hexdigest()[:10]
    return f"{label}-{suffix}"

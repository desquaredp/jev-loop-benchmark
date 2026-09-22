#!/usr/bin/env python3
"""Run five repair tasks across cheap-only, JEV-led, and strong-only arms."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time

from rich import box
from rich.console import Console
from rich.table import Table

import config
import grader
import loop
import providers
import repo_agent as agent
import repo_tools
import split_router
import spike


TASKS = {
    "django__django-13344": "Django 13344",
    "django__django-15128": "Django 15128",
    "astropy__astropy-13579": "Astropy 13579",
    "scikit-learn__scikit-learn-25102": "Scikit-learn 25102",
    "pytest-dev__pytest-6197": "Pytest 6197",
}
ARMS = ("cheap_only", "jev_cascade", "strong")
ARM_LABELS = {
    "cheap_only": "Cheap-only",
    "jev_cascade": "JEV-led",
    "strong": "Strong-only",
}
INSTRUCTION = """
The repair_control check ledger persists across turns. Grade/finish requires
official AND any supplied extra checks. After making a repair, verify it before
broad new exploration. Aim to implement an initial candidate within eight coding
calls. A candidate is automatically graded before the next call; use the resulting
failures to repair it. Bundle related edits in one action. Reserve the last eight
calls for implementation and regression repair, not general investigation.
Official regression test names may not exist in the base checkout: the grader
installs them separately. Do not repeatedly search for absent tests. The baseline
feedback is NOT a previous model attempt. Do not invent prior attempts or assume
a local pass is enough.
"""


def log(message):
    print(message, file=sys.stderr, flush=True)


def _inside_root(path):
    path = Path(path)
    absolute = path.resolve() if path.is_absolute() else (loop.ROOT / path).resolve()
    try:
        return absolute.relative_to(loop.ROOT)
    except ValueError as exc:
        raise ValueError("Run directory must be inside this repository") from exc


def _model_config(settings):
    if settings.cheap.name == settings.strong.name:
        raise ValueError("CHEAP_MODEL and STRONG_MODEL must have distinct names for call accounting")
    return {
        "cheap": settings.cheap.name,
        "strong": settings.strong.name,
        "providers": {
            settings.cheap.name: settings.cheap.provider,
            settings.strong.name: settings.strong.provider,
        },
        "prices": {
            settings.cheap.name: settings.cheap.prices,
            settings.strong.name: settings.strong.prices,
        },
        "reasoning_effort": settings.reasoning_effort,
        "max_completion_tokens": settings.max_completion_tokens,
    }


def prepare(run_dir, settings, *, resume=False):
    if run_dir.exists() and not resume:
        raise RuntimeError(f"Run directory already exists: {run_dir}. Use --resume or choose a new path.")
    run_dir.mkdir(parents=True, exist_ok=True)
    model_config = _model_config(settings)
    for task_id in TASKS:
        source = loop.ROOT / "tasks" / task_id
        target = run_dir / task_id
        target.mkdir(exist_ok=True)
        for filename in ("grader-dataset.json", "grader-only.json"):
            destination = target / filename
            if not destination.exists():
                shutil.copyfile(source / filename, destination)
        task_path = target / "task.json"
        if not task_path.exists():
            task = loop.read_json(source / "task.json")
            task.update(model_config)
            task.update({
                "budget_usd": max(settings.cheap_budget, settings.jev_budget, settings.strong_budget),
                "token_reservation": True,
                "grading_dataset": str(target / "grader-dataset.json"),
                "repair_policy_v2": True,
            })
            task.pop("budget_group", None)
            task.pop("group_budget_usd", None)
            loop.write_json(task_path, task)
    protocol = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tasks": list(TASKS),
        "arms": list(ARMS),
        "parallel_paid_jobs": settings.max_parallel_runs,
        "parallel_official_graders": settings.max_parallel_graders,
        "models": {
            "cheap": {"provider": settings.cheap.provider, "name": settings.cheap.name, "prices": settings.cheap.prices},
            "strong": {"provider": settings.strong.provider, "name": settings.strong.name, "prices": settings.strong.prices},
            "jev": {"name": settings.jev_model, "input_price": settings.jev_input_price},
        },
        "budgets": {
            "cheap_only": settings.cheap_budget,
            "jev_cascade": settings.jev_budget,
            "strong": settings.strong_budget,
        },
        "policy": "32 coding calls, 6 official grading attempts, 4-call JEV reassessment, at most 4 strong diagnostic calls in the JEV arm.",
        "isolation": "Fresh Docker checkout and stateless provider requests for every task/arm pair.",
        "grading": "Official SWE-bench tests on every candidate; scikit-learn also requires two preserved regression probes.",
        "cost_scope": "Provider inference only; excludes Docker/host compute and engineering time.",
        "warning": "Five known development examples, not a statistically significant or blind benchmark.",
    }
    loop.write_json(run_dir / "protocol.json", protocol)


def run_preflights(run_dir, max_workers=5):
    def one(task_id):
        task_dir = run_dir / task_id
        receipt = task_dir / "preflight/result.json"
        if receipt.exists():
            result = loop.read_json(receipt)
        else:
            log(f"preflight started: {task_id}")
            result = grader.preflight(task_dir)
        log(f"preflight {'passed' if result.get('passed') else 'failed'}: {task_id}")
        return task_id, result

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        return dict(pool.map(one, TASKS))


def extra_tests(task_id):
    if task_id != "scikit-learn__scikit-learn-25102":
        return {}
    return {
        "dtype": ["python", "-c", (loop.ROOT / "checks/scikit_dtype.py").read_text()],
        "scaling": ["python", "-c", (loop.ROOT / "checks/scikit_scaling.py").read_text()],
    }


def combine_grade(official, probes):
    passed = official["resolved"] and all(result["exit_code"] == 0 for result in probes.values())
    return {
        **official,
        "status": "passed" if passed else "failed",
        "resolved": passed,
        "extra_checks": probes,
        "feedback": official.get("feedback", "")
        + "\nAdditional required regression checks:\n"
        + json.dumps(probes),
    }


def grade_candidate(source, directory, task, patch):
    value = agent.grade_patch(source, directory, task, patch)
    commands = extra_tests(task["id"])
    if not commands:
        return value
    probes = {}
    with loop.Box(task) as box_instance:
        if patch:
            loop.require_ok(box_instance.shell("git apply -", patch))
        for name, argv in commands.items():
            result = agent.tool(box_instance, "test", {"argv": argv, "timeout_seconds": 60})
            if "exit_code" not in result:
                raise agent.InfrastructureError("Extra regression check transport failed: " + str(result))
            probes[name] = {**result, "argv": argv}
    value = combine_grade(value, probes)
    loop.write_json(directory / "combined-grade.json", value)
    return value


def initial_failure(source):
    receipt = loop.read_json(source / "preflight/unfixed.json")
    log_path = Path(receipt["report"]).parent / "test_output.txt"
    return {
        "status": "failed",
        "resolved": False,
        "official_resolved": False,
        "tests": receipt["details"]["tests_status"],
        "feedback": repo_tools.clip(log_path.read_text(errors="replace"), 24000),
    }


def add_worker_context(arm, task, payload, system, router, initial):
    if len(router.ledger.events) == 1 and not payload["action_history"]:
        payload["working_memory"] = "Fresh task. The baseline fails the issue tests. No model repair has been attempted yet."
    observations = payload["recent_observations"]
    if len(router.ledger.events) <= 8 and not any(item["action"] == "grade" for item in observations):
        payload["recent_observations"] = ([{"action": "grade", "args": {}, "result": initial}] + observations)[-8:]
    payload["repair_control"] = router.packet(arm == "jev_cascade" and router.task.get("consulted", False))
    payload["required_extra_tests"] = extra_tests(task["id"])
    if arm == "cheap_only":
        payload["repair_control"]["models"]["strong"]["role"] = "Not available in this control."
        payload["repair_control"]["progress_evidence"]["consultation_available"] = False
    elif arm == "strong":
        payload["repair_control"]["models"]["cheap"]["role"] = "Not available in this control."
        payload["repair_control"]["models"]["strong"]["role"] = "Full-repository investigator and patch implementer."
        payload["repair_control"]["progress_evidence"]["consultation_available"] = False
    return payload, system + INSTRUCTION


def run_arm(run_dir, task_id, arm, settings):
    source = run_dir / task_id
    out = source / arm
    if (out / "result.json").exists():
        return loop.read_json(out / "result.json")
    budget = {
        "cheap_only": settings.cheap_budget,
        "jev_cascade": settings.jev_budget,
        "strong": settings.strong_budget,
    }[arm]
    task = agent.frozen_task(source, out, budget)
    initial = initial_failure(source)
    seed = [{"decision": {"action": "grade"}, "result": initial, "patch_sha256": agent.sha("")}]
    gateway = providers.Gateway()
    started = time.monotonic()
    consultant_calls = 0
    result = None
    try:
        with loop.Box(task) as box_instance:
            router = split_router.SplitRouter(box_instance if arm == "jev_cascade" else None, out, task, seed, max_turns=32)
            first_route = True

            def route(state):
                nonlocal first_route
                if first_route:
                    first_route = False
                    state["observations"] = [{"action": "grade", "args": {}, "result": initial}]
                    state["note"] = "Fresh task. The baseline fails the issue tests. No model repair has been attempted yet."
                return router(state)

            def ask(model, payload, system):
                nonlocal consultant_calls
                if arm != "jev_cascade":
                    payload, system = add_worker_context(arm, task, payload, system, router, initial)
                else:
                    payload["repair_control"] = router.packet(consultant_calls > 0)
                    payload["required_extra_tests"] = extra_tests(task_id)
                    system += INSTRUCTION
                    if model == task["strong"]:
                        consultant_calls += 1
                        payload = {key: payload[key] for key in (
                            "issue", "jev_guidance", "repair_control", "required_extra_tests", "current_patch"
                        )}
                        payload["current_patch"] = repo_tools.clip(payload["current_patch"], 12000)
                        trace = loop.jsonl(out / "trace.jsonl") if (out / "trace.jsonl").exists() else []
                        payload["consultant_observations"] = [row for row in trace if row.get("role") == "consultant"][-3:]
                        system += f"\nConsultant call {consultant_calls}/4. Return actionable diagnosis and repair_plan with advise no later than call 4.\n"
                return agent.call_worker(gateway, out, task, arm, model, payload, system)

            def no_route(state):
                raise AssertionError(f"{arm} must not invoke JEV routing")

            result = agent.run_session(
                box_instance,
                task,
                out,
                arm,
                ask,
                route if arm == "jev_cascade" else no_route,
                lambda directory, patch: grade_candidate(source, directory, task, patch),
                max_turns=32,
                max_grades=6,
                max_seconds=5400,
                specialist_turns=4,
                route_every=4 if arm == "jev_cascade" else None,
            )
    except Exception as exc:
        result = {"status": "incomplete", "official_resolved": None, "error": str(exc), "type": type(exc).__name__}
    calls = loop.jsonl(out / "calls.jsonl") if (out / "calls.jsonl").exists() else []
    counts = {
        "cheap": sum(row.get("model") == task["cheap"] for row in calls),
        "strong": sum(row.get("model") == task["strong"] for row in calls),
        "jev": sum(row.get("model") == spike.MODEL for row in calls),
    }
    result.update({
        "arm": arm,
        "resolved": result.get("status") == "passed" and bool((result.get("final_grade") or {}).get("resolved")),
        "cost": sum(row.get("cost", 0) for row in calls),
        "calls": counts,
        "billing_complete": not any(row.get("billing_unknown") for row in calls),
        "wall_seconds": time.monotonic() - started,
    })
    loop.write_json(out / "result.json", result)
    return result


def run_paid(run_dir, settings):
    jobs = [(task_id, arm) for task_id in TASKS for arm in ARMS]
    log(f"starting {len(jobs)} paid jobs with {settings.max_parallel_runs} workers")
    with concurrent.futures.ThreadPoolExecutor(max_workers=settings.max_parallel_runs) as pool:
        futures = {
            pool.submit(run_arm, run_dir, task_id, arm, settings): (task_id, arm)
            for task_id, arm in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            task_id, arm = futures[future]
            try:
                result = future.result()
                log(f"finished {task_id} {arm}: {result.get('status')} ${result.get('cost', 0):.4f}")
            except Exception as exc:
                log(f"worker crashed {task_id} {arm}: {exc}")


def archive_incomplete_arms(run_dir):
    archived = []
    for task_id in TASKS:
        for arm in ARMS:
            source = run_dir / task_id / arm
            result_path = source / "result.json"
            if not result_path.exists() or loop.read_json(result_path).get("status") != "incomplete":
                continue
            number = 1
            while (run_dir / task_id / f"{arm}-infrastructure-{number:02d}").exists():
                number += 1
            target = run_dir / task_id / f"{arm}-infrastructure-{number:02d}"
            source.rename(target)
            archived.append(str(target))
    if archived:
        receipt_path = run_dir / "infrastructure-attempts.json"
        previous = loop.read_json(receipt_path).get("directories", []) if receipt_path.exists() else []
        loop.write_json(run_dir / "infrastructure-attempts.json", {
            "excluded_from_comparison": True,
            "reason": "Provider/Docker transport failure; no correctness label was assigned.",
            "directories": list(dict.fromkeys(previous + archived)),
        })
    return archived


def _duration(seconds):
    if seconds is None:
        return "time unavailable"
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def _cell(result, arm):
    if not result:
        return "Not run"
    outcome = "Pass" if result.get("resolved") else ("Error" if result.get("status") == "incomplete" else "Fail")
    calls = result.get("calls", {})
    if arm == "jev_cascade":
        call_text = f"{calls.get('cheap', 0)}/{calls.get('strong', 0)}/{calls.get('jev', 0)} calls"
    elif arm == "cheap_only":
        call_text = f"{calls.get('cheap', 0)} calls"
    else:
        call_text = f"{calls.get('strong', 0)} calls"
    return f"{outcome} · ${result.get('cost', 0):.3f} · {call_text} · {_duration(result.get('wall_seconds'))}"


def collect(run_dir):
    cases = {}
    for task_id, label in TASKS.items():
        cases[task_id] = {"label": label, "arms": {}}
        for arm in ARMS:
            path = run_dir / task_id / arm / "result.json"
            cases[task_id]["arms"][arm] = loop.read_json(path) if path.exists() else None
    return cases


def write_reports(run_dir):
    cases = collect(run_dir)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": loop.read_json(run_dir / "protocol.json"),
        "cases": cases,
        "totals": {},
        "infrastructure_attempts": loop.read_json(run_dir / "infrastructure-attempts.json")
        if (run_dir / "infrastructure-attempts.json").exists() else None,
    }
    for arm in ARMS:
        results = [entry["arms"][arm] for entry in cases.values() if entry["arms"][arm]]
        summary["totals"][arm] = {
            "passed": sum(bool(result.get("resolved")) for result in results),
            "attempted": len(results),
            "cost": sum(result.get("cost", 0) for result in results),
            "calls": {
                role: sum(result.get("calls", {}).get(role, 0) for result in results)
                for role in ("cheap", "strong", "jev")
            },
        }
    loop.write_json(run_dir / "summary.json", summary)

    headers = ["Case", "Cheap-only", "JEV-led", "Strong-only"]
    rows = [
        [entry["label"], *[_cell(entry["arms"][arm], arm) for arm in ARMS]]
        for entry in cases.values()
    ]
    with (run_dir / "comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)
    markdown = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    markdown.extend("| " + " | ".join(row) + " |" for row in rows)
    (run_dir / "comparison.md").write_text("\n".join(markdown) + "\n")
    return cases


def sanitize_run_artifacts(run_dir):
    """Remove local paths or accidentally echoed credentials from shareable receipts."""
    replacements = {
        str(loop.ROOT): "<repo>",
        str(Path.home()): "<home>",
    }
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "JEV_API_KEY", "TYPESAFE_API_KEY"):
        value = os.getenv(name)
        if value and len(value) >= 8:
            replacements[value] = f"<redacted-{name.lower()}>"
    for path in Path(run_dir).rglob("*"):
        if not path.is_file() or path.stat().st_size > 50_000_000:
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        updated = text
        for old, new in replacements.items():
            updated = updated.replace(old, new)
        if updated != text:
            path.write_text(updated)
    remaining = []
    for path in Path(run_dir).rglob("*"):
        if not path.is_file() or path.stat().st_size > 50_000_000:
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        for needle in replacements:
            if needle and needle in text:
                remaining.append(str(path))
                break
    if remaining:
        raise RuntimeError("Sensitive local data remains in run artifacts: " + ", ".join(remaining[:5]))


def print_table(cases):
    table = Table(box=box.SIMPLE_HEAVY, show_lines=True, expand=False)
    table.add_column("Case", no_wrap=True)
    table.add_column("Cheap-only", no_wrap=True)
    table.add_column("JEV-led", no_wrap=True)
    table.add_column("Strong-only", no_wrap=True)
    for entry in cases.values():
        table.add_row(entry["label"], *[_cell(entry["arms"][arm], arm) for arm in ARMS])
    Console(file=sys.stdout, force_terminal=False, color_system=None, width=160).print(table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Required for paid model/JEV calls")
    parser.add_argument("--resume", action="store_true", help="Reuse passed preflights and completed arms")
    parser.add_argument("--parallel", type=int, help="Override MAX_PARALLEL_RUNS")
    args = parser.parse_args()
    os.chdir(loop.ROOT)
    run_dir = _inside_root(args.run_dir)
    settings = config.load_settings()
    if args.parallel:
        settings = settings.__class__(**{**settings.__dict__, "max_parallel_runs": args.parallel})
    if args.command == "run":
        if not args.execute:
            parser.error("Paid benchmark requires --execute")
        settings.validate(paid=True)
        prepare(run_dir, settings, resume=args.resume)
        preflights = run_preflights(run_dir, settings.max_parallel_graders)
        if all(result.get("passed") for result in preflights.values()):
            archive_incomplete_arms(run_dir)
            run_paid(run_dir, settings)
        else:
            log("paid jobs skipped for failed preflight cases")
    cases = write_reports(run_dir)
    sanitize_run_artifacts(run_dir)
    print_table(cases)


if __name__ == "__main__":
    main()

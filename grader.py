"""Official SWE-bench grading adapter and mandatory red/green preflight."""

from __future__ import annotations

import re
from pathlib import Path
import shlex
import os
import threading

import loop


_GRADER_SLOTS = threading.BoundedSemaphore(max(1, int(os.getenv("MAX_PARALLEL_GRADERS", "2"))))


def digest(path):
    return loop.digest(path)


def inert_patch(task):
    """An empty patch skips official tests, so make a harmless tracked change."""
    with loop.Box(task) as box:
        path = task["baseline_order"][0]
        source = loop.require_ok(box.shell("cat -- " + shlex.quote(path)))
        marker = "\n# Unfixed grading control.\n"
        loop.require_ok(box.shell("python -c " + shlex.quote(
            "from pathlib import Path; p=Path(" + repr(path) + "); p.write_text(p.read_text()+" + repr(marker) + ")"
        )))
        return box.diff()


def _report_path(out, task, label):
    run_id = loop.evaluation_run_id(out, label)
    return Path(out) / label / "logs/run_evaluation" / run_id / run_id / task["id"] / "report.json"


def logged_statuses(spec, log_path, task):
    from swebench.harness.grading import get_logs_eval

    statuses, valid = get_logs_eval(spec, str(log_path))
    if not valid or not task.get("django_subtest_parser"):
        return statuses, valid
    text = Path(log_path).read_text(errors="replace")
    text = text.split(">>>>> Start Test Output", 1)[-1].split(">>>>> End Test Output", 1)[0]
    for name in spec.FAIL_TO_PASS + spec.PASS_TO_PASS:
        if re.search(r"(?<!\w)" + re.escape(name) + r" \.\.\. (?:ok|OK)(?=\s|$)", text):
            statuses[name] = "PASSED"
        for prefix, status in (("FAIL", "FAILED"), ("ERROR", "ERROR")):
            if re.search(r"^" + prefix + r": " + re.escape(name) + r"(?: \([^\n]*\))?\s*$", text, re.M):
                statuses[name] = status
    return statuses, valid


def validate_report(task_dir, report_path, task):
    report_path = Path(report_path)
    report = loop.read_json(report_path)[task["id"]]
    if not report.get("patch_successfully_applied") or "tests_status" not in report:
        raise RuntimeError("Official evaluator did not execute valid tests")
    from swebench.harness.grading import get_eval_tests_report, get_logs_eval, get_resolution_status
    from swebench.harness.test_spec.test_spec import make_test_spec

    spec = make_test_spec(loop.read_json(Path(task_dir) / "grader-only.json"), namespace="swebench")
    statuses, valid = logged_statuses(spec, report_path.parent / "test_output.txt", task)
    missing = set(spec.FAIL_TO_PASS + spec.PASS_TO_PASS) - set(statuses)
    if not valid or missing:
        raise RuntimeError(f"Official test execution incomplete: {sorted(missing)}")
    if task.get("django_subtest_parser"):
        from swebench.harness.constants import ResolvedStatus

        tests = get_eval_tests_report(statuses, {
            "instance_id": task["id"],
            "FAIL_TO_PASS": spec.FAIL_TO_PASS,
            "PASS_TO_PASS": spec.PASS_TO_PASS,
        })
        report = {
            **report,
            "tests_status": tests,
            "resolved": get_resolution_status(tests) == ResolvedStatus.FULL.value,
            "parser_adapter": "Explicit full unittest subtest names; original report and log preserved.",
        }
        loop.write_json(report_path.parent / "adapted-report.json", {task["id"]: report})
    return report


def official_grade(task_dir, out, task, patch, label):
    out = Path(out)
    actual_label = label
    for attempt in range(2):
        actual_label = label if not attempt else label + "-transport-retry"
        try:
            with _GRADER_SLOTS:
                loop.grade(out, task, patch, actual_label)
            break
        except (RuntimeError, OSError):
            log_path = out / actual_label / "grader.log"
            log = log_path.read_text(errors="replace") if log_path.exists() else ""
            if attempt or not any(term in log.lower() for term in ("connection refused", "connection reset", "500 server error")):
                raise
            loop.write_json(out / (actual_label + "-retry.json"), {
                "reason": "Docker transport error; identical patch and no model call."
            })
    report_path = _report_path(out, task, actual_label)
    report = validate_report(task_dir, report_path, task)
    return {"resolved": report["resolved"], "report": str(report_path), "details": report}


def preflight(task_dir):
    task_dir = Path(task_dir)
    directory = task_dir / "preflight"
    directory.mkdir(exist_ok=False)
    task = loop.read_json(task_dir / "task.json")
    try:
        if loop.command(["docker", "image", "inspect", task["image"]])["exit_code"]:
            pull = loop.command(["docker", "pull", "--platform", "linux/amd64", task["image"]], timeout=1200)
            loop.write_json(directory / "image-pull.json", pull)
            loop.require_ok(pull)
        control = official_grade(task_dir, directory, task, inert_patch(task), "unfixed")
        loop.write_json(directory / "unfixed.json", control)
        tests = control["details"]["tests_status"]
        assert control["resolved"] is False and tests["FAIL_TO_PASS"]["failure"]
        assert not tests["PASS_TO_PASS"]["failure"], "Base regressions failed; task is ineligible"
        golden_patch = loop.read_json(task_dir / "grader-only.json")["patch"]
        golden = official_grade(task_dir, directory, task, golden_patch, "golden")
        loop.write_json(directory / "golden.json", golden)
        assert golden["resolved"] is True, "Golden patch did not resolve under official grading"
        result = {"passed": True, "task_sha256": digest(task_dir / "task.json")}
    except Exception as exc:
        result = {"passed": False, "error": str(exc), "type": type(exc).__name__}
    loop.write_json(directory / "result.json", result)
    return result


def require_preflight(task_dir):
    task_dir = Path(task_dir)
    result = loop.read_json(task_dir / "preflight/result.json")
    assert result["passed"], "No paid calls allowed without official red/green preflight"
    assert result["task_sha256"] == digest(task_dir / "task.json"), "Task changed after preflight"
    task = loop.read_json(task_dir / "task.json")
    for label, expected in (("unfixed", False), ("golden", True)):
        receipt = loop.read_json(task_dir / "preflight" / (label + ".json"))
        actual = validate_report(task_dir, receipt["report"], task)
        assert actual["resolved"] is expected

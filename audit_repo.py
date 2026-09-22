#!/usr/bin/env python3
"""Fail if shareable source files contain credentials or local absolute paths."""

import os
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
SKIP = {".git", ".venv", "runs", "__pycache__"}
KEY_ASSIGNMENT = re.compile(r"(?:OPENAI|ANTHROPIC|JEV|TYPESAFE)_[A-Z_]*KEY\s*=\s*[^\s#]+")


def main():
    findings = []
    secret_values = [
        value for name, value in os.environ.items()
        if name in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "JEV_API_KEY", "TYPESAFE_API_KEY"}
        and len(value) >= 8
    ]
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP for part in path.relative_to(ROOT).parts):
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        mac_home = "/" + "Users/"
        linux_home = r"/" + r"home/[^/<\s]+"
        if mac_home in text or re.search(linux_home, text):
            findings.append(f"local path: {path.relative_to(ROOT)}")
        if KEY_ASSIGNMENT.search(text) and path.name != ".env.example":
            findings.append(f"non-placeholder key assignment: {path.relative_to(ROOT)}")
        if any(secret in text for secret in secret_values):
            findings.append(f"live credential: {path.relative_to(ROOT)}")
    if findings:
        raise SystemExit("\n".join(findings))
    print("source audit passed")


if __name__ == "__main__":
    main()

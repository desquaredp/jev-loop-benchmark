# JEV-led repair loop demo

A small, reproducible comparison of three full-repository coding-agent setups:

- cheap model only
- JEV routing, cheap implementation, and bounded strong-model diagnosis
- strong model only

The five included SWE-bench tasks use official grading. Scikit-learn also uses two regression probes discovered during development. This is an illustrative five-case experiment, not a statistically significant benchmark.

## Run it

Requirements: Python 3.11+, Docker with `linux/amd64` emulation, and enough provider credit for 15 independent runs.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add keys, providers, model names, and current prices to .env.

python -m unittest discover -s tests -v
python benchmark.py run --execute --run-dir runs/demo-01
```

The paid command submits all 15 task/arm pairs concurrently. Official Docker graders use a separate small semaphore because Docker Desktop can fail when too many evaluators start at once. Tune `MAX_PARALLEL_RUNS` and `MAX_PARALLEL_GRADERS` independently.

Stdout contains only the final comparison table. Progress goes to stderr. Detailed artifacts are saved per arm under `runs/demo-01/`, including exact prompts and responses, token usage and cost, routing packets, tool traces, candidate patches, grader logs, `summary.json`, `comparison.csv`, and `comparison.md`.

Transient provider failures receive up to three bounded retries. `--resume` archives infrastructure-incomplete arm directories and reruns only those arms; completed outcomes are never silently overwritten.

To reprint a completed run:

```bash
python benchmark.py report --run-dir runs/demo-01
```

Every model call is stateless. The harness carries task-local continuity by explicitly sending the current patch, working-memory note, recent observations, action history, and grader feedback. Task contexts never mix.

## What JEV controls

JEV separately classifies the failure, progress, next model allocation, and useful implementation files. Its ordered top five is injected into every JEV-led coding prompt as inspect-first guidance, not a file-access restriction. If JEV requests escalation, the strong model receives up to four diagnostic calls; the cheap model still implements the repair.

Costs are calculated from provider-reported usage and the prices supplied in `.env`. They exclude Docker compute and engineering time.

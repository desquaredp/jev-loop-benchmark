# JEV-led repair loop demo

A small, reproducible comparison of three full-repository coding-agent setups:

- cheap model only
- JEV routing, cheap implementation, and bounded strong-model diagnosis
- strong model only

The five included SWE-bench tasks use official grading. Scikit-learn also uses two regression probes discovered during development. This is an illustrative five-case experiment, not a statistically significant benchmark.

## Demonstration result

| Case | Cheap-only | JEV-led | Strong-only |
|---|---|---|---|
| Django 13344 | Pass · $0.063 · 13 calls · 1:57 agent | Pass · $0.331 · 2/4/10 calls · 1:47 agent | Pass · $1.163 · 5 calls · 1:27 agent |
| Django 15128 | Pass · $0.028 · 6 calls · 1:33 agent | Pass · $0.410 · 3/4/10 calls · 2:14 agent | Pass · $1.454 · 6 calls · 1:41 agent |
| Astropy 13579 | Pass · $0.017 · 4 calls · 1:03 agent | Pass · $0.382 · 3/3/10 calls · 2:07 agent | Pass · $0.685 · 3 calls · 1:10 agent |
| Scikit-learn 25102 | Fail · $0.238 · 32 calls · 6:13 agent | Pass · $0.414 · 2/4/10 calls · 1:33 agent | Pass · $2.023 · 8 calls · 2:00 agent |
| Pytest 6197 | Fail · $0.122 · 32 calls · 3:45 agent | Pass · $0.464 · 2/4/10 calls · 1:43 agent | Pass · $1.060 · 5 calls · 1:19 agent |

JEV-led and strong-only both resolved 5/5 cases. JEV-led cost $2.001 versus $6.385 for strong-only, a 68.7% reduction on this small selected sample. Cheap-only resolved 3/5. JEV-led calls are shown as cheap/strong/JEV. Sanitized receipts are under [`results/shareable-01`](results/shareable-01).

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

The table reports active agent time: model calls, repository inspection, edits, and local tool use. It excludes official-grader execution and time queued for a grader slot. Detailed JSON retains active agent time, grading wall time, and total wall time separately.

Transient provider failures receive up to three bounded retries. `--resume` archives infrastructure-incomplete arm directories and reruns only those arms; completed outcomes are never silently overwritten.

To reprint a completed run:

```bash
python benchmark.py report --run-dir runs/demo-01
```

Every model call is stateless. The harness carries task-local continuity by explicitly sending the current patch, working-memory note, recent observations, action history, and grader feedback. Task contexts never mix.

## What JEV controls

JEV separately classifies the failure, progress, next model allocation, and useful implementation files. Its ordered top five is injected into every JEV-led coding prompt as inspect-first guidance, not a file-access restriction. If JEV requests escalation, the strong model receives up to four diagnostic calls; the cheap model still implements the repair.

Costs are calculated from provider-reported usage and the prices supplied in `.env`. They exclude Docker compute and engineering time.

## Project notes

See [CONTRIBUTING.md](CONTRIBUTING.md) for local checks, [SECURITY.md](SECURITY.md) for private vulnerability reports, and [CITATION.cff](CITATION.cff) for citation metadata.

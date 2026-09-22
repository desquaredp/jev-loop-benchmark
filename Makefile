PYTHON ?= python
RUN_DIR ?= runs/demo-01

.PHONY: test audit report

test:
	$(PYTHON) -m unittest discover -s tests -v

audit:
	$(PYTHON) audit_repo.py

report:
	$(PYTHON) benchmark.py report --run-dir $(RUN_DIR)

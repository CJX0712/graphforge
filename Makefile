# graphForge 开发便捷命令
PY ?= python

.PHONY: install test lint format ci demo bench clean

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

ci: lint test
	$(PY) examples/run_demo.py benchmark.json

demo:
	$(PY) examples/run_demo.py benchmark.json

bench:
	$(PY) -m graphforge.cli benchmark --out benchmark.json

clean:
	rm -rf __pycache__ .pytest_cache build dist *.egg-info benchmark.json

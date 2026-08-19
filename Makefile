.PHONY: install format

install:
	uv sync
	uv pip install xautodl --no-deps

format:
	ruff check --fix . && ruff format .

baseline200:
	uv run python -m kd_vs_hpo.common.train_pipeline -m kd.student=11570,8712,1342 general.seed=42,17,84

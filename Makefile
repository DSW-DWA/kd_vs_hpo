.PHONY: install format

install:
	uv sync
	uv pip install xautodl --no-deps

format:
	ruff check --fix . && ruff format .

baseline200:
	uv run python -m kd_vs_hpo.common.train_pipeline --config-name base_models -m student=11570,8712,559,1676,5750,2514,7153,13197,3358,11898  seed=42,17,84


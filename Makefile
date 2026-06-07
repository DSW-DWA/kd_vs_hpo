.PHONY: install format

install:
	uv sync
	uv pip install xautodl --no-deps

format:
	ruff check --fix . && ruff format .
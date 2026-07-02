.PHONY: dotenv_init dvc_init format lint check

dotenv_init: .bash_scripts/dotenv_init.sh
dvc_init:
	@if [ ! -d ".dvc" ]; then \
		dvc init; \
	else \
		echo "DVC вже ініціалізовано."; \
	fi
	touch dvc.yaml
format:
	uv run --active ruff format .
	uv run --active ruff check --fix .
lint:
	uv run --active ruff format --check .
	uv run --active ruff check .
	uv run --active mypy src/
check: lint
clean:
	rm -rf .mypy_cache .ruff_cache
	uv cache clean

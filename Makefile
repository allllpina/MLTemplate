.PHONY: dotenv_init dvc_init

dotenv_init: .bash_scripts/dotenv_init.sh
dvc_init:
	@if [ ! -d ".dvc" ]; then \
		dvc init; \
	else \
		echo "DVC вже ініціалізовано."; \
	fi
	touch dvc.yaml

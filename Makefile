.DEFAULT_GOAL := help
PY ?= python3
VENV := .venv
BIN := $(VENV)/bin
SPEC ?= specs/welfare.yaml
EVAL_ARGS ?=

$(VENV):
	$(PY) -m venv $(VENV)

.PHONY: install
install: $(VENV) ## create the venv and install the package with dev extras
	$(BIN)/pip install --quiet --upgrade pip
	$(BIN)/pip install --quiet -e ".[dev]"
	@echo "installed. try: make test"

.PHONY: data
data: ## re-download the public sources and rebuild data/vendor (needs ~650 MB and poppler)
	$(BIN)/python scripts/fetch_data.py --all

.PHONY: demo
demo: install ## run the welfare wording experiment end to end, from a clean clone
	$(BIN)/quorum validate $(SPEC)
	$(BIN)/quorum run $(SPEC)

.PHONY: eval
eval: ## run the ablation grid over the question bank and regenerate EVAL.md
	$(BIN)/quorum eval $(EVAL_ARGS)

.PHONY: test
test: ## run the test suite and the quality gates
	$(BIN)/pytest --cov=quorum --cov-report=term-missing --cov-fail-under=85

.PHONY: fast-test
fast-test: ## run the test suite without coverage
	$(BIN)/pytest

.PHONY: clean
clean: ## remove build and test artifacts
	rm -rf build dist .pytest_cache .coverage htmlcov src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

.PHONY: help
help: ## show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

#  WSQ Trading Model — Makefile
#  Usage: make <target>

PYTHON   := python
PIP      := pip
PYTEST   := pytest
BLACK    := black
RUFF     := ruff
MYPY     := mypy

SRC      := wsq_trading
TESTS    := wsq_trading/tests

.PHONY: help install install-dev lint format typecheck test test-cov clean data-dirs

# Default
help:
	@echo ""
	@echo "WSQ Trading Model — available targets:"
	@echo "  install       Install package in editable mode"
	@echo "  install-dev   Install with dev dependencies"
	@echo "  lint          Run ruff linter"
	@echo "  format        Auto-format with black"
	@echo "  typecheck     Run mypy type checking"
	@echo "  test          Run pytest"
	@echo "  test-cov      Run pytest with coverage report"
	@echo "  clean         Remove build artefacts and caches"
	@echo "  data-dirs     Create / verify data directory structure"
	@echo ""

# Install
install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

# Code quality
lint:
	$(RUFF) check $(SRC)

format:
	$(BLACK) $(SRC)

typecheck:
	$(MYPY) $(SRC)

# Testing
test:
	MKL_THREADING_LAYER=GNU $(PYTEST) $(TESTS)

test-cov:
	MKL_THREADING_LAYER=GNU $(PYTEST) $(TESTS) --cov=$(SRC) --cov-report=html --cov-report=term-missing

# Data
data-dirs:
	mkdir -p data/raw/futures data/processed/futures data/synthetic \
	         models/checkpoints \
	         results/backtest_reports results/metrics results/plots
	@echo "Data directories verified."

# Clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/ htmlcov/ .coverage
	@echo "Clean complete."

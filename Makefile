# AOT Stock Network — Task automation
# ====================================

.PHONY: install install-dev lint typecheck test test-cov clean docker-build docker-run

# ── Installation ────────────────────────────────────────────────────────────
install:
	pip install -e ".[ml,dev]"

install-dev:
	pip install -e ".[dev]"

# ── Lint & Type check ───────────────────────────────────────────────────────
lint:
	ruff check src/ tests/
	ruff format --check src/ tests/ --diff

typecheck:
	mypy src/ --ignore-missing-imports

format:
	ruff format src/ tests/

# ── Testing ─────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short -m "not slow"

test-all:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ --cov=aot_stock_network --cov-report=term-missing --cov-report=html

test-fast:
	pytest tests/ -v --tb=short -m "smoke" -x

# ── CLI ─────────────────────────────────────────────────────────────────────
fetch:
	aot-fetch

train:
	aot-train

preprocess:
	aot-preprocess

dashboard:
	streamlit run Home.py

# ── Docker ──────────────────────────────────────────────────────────────────
docker-build:
	docker compose build

docker-run:
	docker compose up -d

docker-stop:
	docker compose down

# ── Data ────────────────────────────────────────────────────────────────────
validate:
	python -c "from aot_stock_network.data.loader import DataLoader; dl = DataLoader(); dl.fetch_all(); dl.validate_all()"

clean-cache:
	python -c "from aot_stock_network.data.loader import DataLoader; DataLoader().clear_cache()"

clean:
	rm -rf data/.cache data/processed/*.csv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache htmlcov coverage.xml

# ── Documentation ───────────────────────────────────────────────────────────
docs:
	cd docs && sphinx-build -b html source build

# ── Full pipeline ───────────────────────────────────────────────────────────
all: install fetch preprocess train

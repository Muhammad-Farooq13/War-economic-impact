.PHONY: help install install-dev lint format test coverage clean train predict app docs

PYTHON   = python
PIP      = pip
SRC      = src
TESTS    = tests
NOTEBOOK_DIR = notebooks

help:           ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Setup ─────────────────────────────────────────────────────────────────────
install:        ## Install production dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev:    ## Install all dev + production dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .
	pre-commit install

# ── Code Quality ──────────────────────────────────────────────────────────────
lint:           ## Run flake8 linter
	flake8 $(SRC) $(TESTS) --max-line-length=100 --ignore=E203,W503

format:         ## Auto-format with black and isort
	black $(SRC) $(TESTS) app --line-length 100
	isort $(SRC) $(TESTS) app --profile black

typecheck:      ## Run mypy type checking
	mypy $(SRC) --ignore-missing-imports

# ── Tests ─────────────────────────────────────────────────────────────────────
test:           ## Run pytest
	pytest $(TESTS) -v

coverage:       ## Run tests with coverage report
	pytest $(TESTS) --cov=$(SRC) --cov-report=html --cov-report=term-missing
	@echo "HTML report: htmlcov/index.html"

# ── Pipeline ──────────────────────────────────────────────────────────────────
data:           ## Run data preprocessing pipeline
	$(PYTHON) -m src.data.make_dataset

features:       ## Run feature engineering pipeline
	$(PYTHON) -m src.features.build_features

train:          ## Train all models
	$(PYTHON) -m src.models.train_model

predict:        ## Run batch predictions
	$(PYTHON) -m src.models.predict_model

evaluate:       ## Evaluate trained models
	$(PYTHON) -m src.models.evaluate_model

pipeline: data features train evaluate  ## Run full ML pipeline end-to-end

# ── Application ───────────────────────────────────────────────────────────────
app:            ## Launch Streamlit web application
	streamlit run app/app.py

mlflow-ui:      ## Launch MLflow tracking UI
	mlflow ui --backend-store-uri mlruns --port 5001

# ── Notebooks ─────────────────────────────────────────────────────────────────
notebooks:      ## Execute all notebooks top-to-bottom (CI use)
	jupyter nbconvert --to notebook --execute $(NOTEBOOK_DIR)/*.ipynb \
		--output-dir $(NOTEBOOK_DIR)/executed/

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:          ## Remove build artefacts, caches, and compiled files
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache htmlcov .coverage mlruns
	rm -rf dist build *.egg-info

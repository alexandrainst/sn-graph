PROJECT_NAME := sn_graph
SRC_DIR := $(CURDIR)/src
TESTS_DIR := $(CURDIR)/tests
TEST_REPORTS_DIR := $(CURDIR)/test-reports
NOTEBOOKS_DIR := $(CURDIR)/notebooks

activate:
	@echo "Activating virtual environment..."
	@poetry shell

install:  # Install the app locally
	@echo "Installing dependencies..."
	@poetry env info
	@poetry install
	@echo "Setting up pre-commit hooks..."
	@poetry run pre-commit install


lint:  ## Run linting and typechecking
	@echo "Running linting and typechecking..."
	@poetry run black $(SRC_DIR) $(TESTS_DIR)/*.py --check
	@poetry run ruff check $(SRC_DIR) $(TESTS_DIR) /*.py
	@poetry run mypy $(SRC_DIR) $(TESTS_DIR) /*.py --check-untyped-defs

format:  ## Run autoformatters
	@echo "Running autoformatters..."
	@poetry run black $(SRC_DIR) $(TESTS_DIR)/*.py
	@poetry run ruff check $(SRC_DIR) $(TESTS_DIR)/*.py --fix

test:  ## Run all tests
	@echo "Running tests..."
	@poetry run pytest $(TESTS_DIR) -svv --cov-branch --cov-report=xml:$(TEST_REPORTS_DIR)/coverage.xml --cov-report=html:$(TEST_REPORTS_DIR)/htmlcov --cov-report=json:$(TEST_REPORTS_DIR)/coverage.json --junitxml=$(TEST_REPORTS_DIR)/results.xml
	@poetry run coverage-badge -o assets/coverage.svg -f

mkdocs:
	@echo "Running notebooks in $(NOTEBOOKS_DIR)..."
	@poetry run jupyter nbconvert --to notebook --execute --inplace $(NOTEBOOKS_DIR)/*.ipynb
	@echo "Building and deploying MkDocs..."
	@poetry run mkdocs build && poetry run mkdocs gh-deploy


.DEFAULT_GOAL := help
help: Makefile
	@grep -E '(^[a-zA-Z_-]+:.*?##.*$$)|(^##)' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[32m%-30s\033[0m %s\n", $$1, $$2}' | sed -e 's/\[

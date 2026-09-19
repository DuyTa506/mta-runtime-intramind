PYTHON := .venv/bin/python
COMPOSE := docker compose -p intramind-runtime-dev -f dev/compose.yaml
AI_TESTS := $(wildcard tests/test_ai_*.py)
LIBRARY_TESTS := $(foreach path,$(AI_TESTS),--ignore=$(path))

.PHONY: setup check test integration build dev-up dev-down dev-status

setup:
	uv sync --frozen

check:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest -m 'not integration' -q

integration:
	RUNTIME_TEST_DATABASE_URL=postgresql+asyncpg://runtime_test:runtime_test_only@127.0.0.1:55440/runtime_test \
	RUNTIME_TEST_ALLOW_RESET=yes RUNTIME_TEST_TEMPORAL_ADDRESS=127.0.0.1:17234 \
	RUNTIME_TEST_MINIO_ENDPOINT=127.0.0.1:19010 \
	$(PYTHON) -m pytest -q $(LIBRARY_TESTS) --junitxml=.test-data/library.xml

build:
	uv build

dev-up:
	$(COMPOSE) up -d --wait --wait-timeout 180

dev-down:
	$(COMPOSE) down

dev-status:
	$(COMPOSE) ps

.PHONY: install install-dev install-eval run dev test eval lint fmt docker-up docker-down docker-logs db-init

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

install-eval:
	pip install -r requirements-eval.txt

## Run the API with auto-reload for local development.
dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

## Run the API without auto-reload (production-like).
run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

## Run the test suite.
test:
	pytest

## Run the RAGAS evaluation harness and write evals/report.md.
eval:
	python -m evals.run_eval

## Build and start the app + Postgres/pgvector containers.
docker-up:
	docker compose up --build -d

## Stop and remove the containers.
docker-down:
	docker compose down

## Tail container logs.
docker-logs:
	docker compose logs -f

## Create the pgvector extension and ORM tables (idempotent).
db-init:
	python -c "from app.db import init_db; init_db()"

lint:
	python -m compileall -q app evals tests

fmt:
	python -m ruff format app evals tests

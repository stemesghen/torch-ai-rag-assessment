.PHONY: build up down logs ingest run clean

build:
	docker compose build

up:
	docker compose up -d elasticsearch

ingest:
	docker compose run rag-app python -m src.ingest

run:
	docker compose run rag-app python -m src.main

down:
	docker compose down

logs:
	docker compose logs -f

clean:
	docker compose down -v
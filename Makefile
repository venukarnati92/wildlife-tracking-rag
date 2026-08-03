.PHONY: install download pipeline index bootstrap app dashboard db-init sample-data generate compose-up compose-down clean

install:
	uv sync

download:
	uv run python scripts/download_movebank.py --limit 3000

pipeline:
	uv run python pipelines/movebank_pipeline.py

index:
	uv run python ingest.py

sample-data:
	cp data/movebank_studies.sample.json data/movebank_studies.json

bootstrap: sample-data index
	@echo "Bootstrapped with the bundled sample dataset. For live data run: make download pipeline index"

# Requires Postgres to already be running and reachable via POSTGRES_HOST/PORT
# (see .env). If you don't have a local Postgres, start one first with:
#   docker compose up -d postgres
db-init:
	uv run python db_init.py

app:
	uv run streamlit run app.py --server.port 8501

dashboard:
	uv run streamlit run dashboard.py --server.port 8502

generate:
	uv run python generate_data.py

compose-up:
	docker compose up --build

compose-down:
	docker compose down -v

clean:
	rm -rf data/index data/raw data/movebank_pipeline.duckdb data/movebank_studies.json

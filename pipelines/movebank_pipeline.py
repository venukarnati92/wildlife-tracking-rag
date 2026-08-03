"""dlt pipeline: load Movebank study CSV(s) into DuckDB.

Reads any CSV placed under `data/raw/` (via the dlt filesystem source), lands
one row per study into DuckDB table `movebank.studies`, and prints a load
summary. Downstream `ingest.py` reads from this DuckDB dataset to build the
minsearch + vector indexes.

Run:
    uv run python pipelines/movebank_pipeline.py
"""

from __future__ import annotations

from pathlib import Path

import dlt
from dlt.sources.filesystem import filesystem, read_csv


def run() -> None:
    raw_dir = Path("data/raw").resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)

    pipeline = dlt.pipeline(
        pipeline_name="movebank_pipeline",
        destination=dlt.destinations.duckdb("data/movebank_pipeline.duckdb"),
        dataset_name="movebank",
    )

    files = filesystem(
        bucket_url=f"file://{raw_dir}",
        file_glob="*.csv",
    )
    reader = (files | read_csv()).with_name("studies")
    reader.max_table_nesting = 0

    load_info = pipeline.run(reader, write_disposition="replace")
    print(load_info)


if __name__ == "__main__":
    run()

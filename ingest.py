"""Build searchable indexes from Movebank study documents.

Produces three artifacts written under `data/index/`:
  - `wildlife_documents.json` : list of study documents
  - `wildlife_index.pkl`      : minsearch text index (fitted)
  - `wildlife_embeddings.npy` : dense vector matrix (rows == documents)

Two entry points:
  - `build_from_pipeline_output()` reads the parquet exports written by
    `pipelines/movebank_pipeline.py` (or falls back to the JSON in `data/`).
  - `load_documents()` reads the persisted `wildlife_documents.json`
    (used by the app / eval notebooks).
"""

from __future__ import annotations

import json
import math
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from minsearch import Index

INDEX_PATH = Path(os.getenv("INDEX_PATH", "data/index/wildlife_index.pkl"))
EMBEDDINGS_PATH = Path(os.getenv("EMBEDDINGS_PATH", "data/index/wildlife_embeddings.npy"))
DOCUMENTS_PATH = Path(os.getenv("DOCUMENTS_PATH", "data/index/wildlife_documents.json"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

TEXT_FIELDS = ["name", "taxa", "study_objective", "location", "principal_investigator", "citation"]
KEYWORD_FIELDS = ["study_id", "license"]


def _norm(value: Any) -> str | None:
    """Normalize CSV cells to strings, dropping NaN/None/empty."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    return text


def _time_period(start: Any, end: Any) -> str | None:
    start = _norm(start)
    end = _norm(end)
    if not start and not end:
        return None
    return f"{start or 'unknown start'} to {end or 'unknown end'}"


def row_to_document(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a Movebank study row (raw JSON/CSV) into a RAG document."""
    return {
        "study_id": _norm(row.get("id") or row.get("study_id")),
        "name": _norm(row.get("name")),
        "taxa": _norm(row.get("taxon_ids") or row.get("taxa")),
        "principal_investigator": _norm(row.get("principal_investigator_name")),
        "contact": _norm(row.get("contact_person_name") or row.get("principal_investigator_email")),
        "location": _norm(_format_location(row)),
        "time_period": _time_period(
            row.get("timestamp_first_deployed_location"),
            row.get("timestamp_last_deployed_location"),
        ),
        "number_of_individuals": _norm(row.get("number_of_individuals")),
        "number_of_tags": _norm(row.get("number_of_tags")),
        "sensor_types": _norm(row.get("sensor_type_ids") or row.get("sensor_types")),
        "study_objective": _norm(row.get("study_objective") or row.get("study_type")),
        "citation": _norm(row.get("citation")),
        "license": _norm(row.get("license_type") or row.get("license")),
        "acknowledgements": _norm(row.get("acknowledgements")),
    }


def _format_location(row: dict[str, Any]) -> str | None:
    lat = _norm(row.get("main_location_lat"))
    lon = _norm(row.get("main_location_long") or row.get("main_location_lon"))
    if lat and lon:
        return f"lat {lat}, lon {lon}"
    return _norm(row.get("main_location"))


def build_search_text(doc: dict[str, Any]) -> str:
    """Concatenate all searchable fields for dense retrieval."""
    parts = [doc.get(f) for f in TEXT_FIELDS]
    parts += [doc.get("acknowledgements")]
    return " | ".join(p for p in parts if p)


def build_text_index(documents: list[dict[str, Any]]) -> Index:
    index = Index(text_fields=TEXT_FIELDS, keyword_fields=KEYWORD_FIELDS)
    index.fit(documents)
    return index


def build_embeddings(documents: list[dict[str, Any]], model_name: str = EMBED_MODEL) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    texts = [build_search_text(d) for d in documents]
    return np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=True))


def save_artifacts(
    documents: list[dict[str, Any]],
    text_index: Index,
    embeddings: np.ndarray,
) -> None:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DOCUMENTS_PATH.open("w") as f:
        json.dump(documents, f, indent=2, default=str)
    with INDEX_PATH.open("wb") as f:
        pickle.dump(text_index, f)
    np.save(EMBEDDINGS_PATH, embeddings)


def load_documents() -> list[dict[str, Any]]:
    with DOCUMENTS_PATH.open() as f:
        return json.load(f)


def load_text_index() -> Index:
    with INDEX_PATH.open("rb") as f:
        return pickle.load(f)


def load_embeddings() -> np.ndarray:
    return np.load(EMBEDDINGS_PATH)


def build_from_pipeline_output(source: str = "auto") -> list[dict[str, Any]]:
    """Load studies from the dlt pipeline output (parquet/duckdb) or the JSON fallback."""
    from pathlib import Path

    documents: list[dict[str, Any]] = []
    if source in {"auto", "duckdb"}:
        duckdb_path = Path("data/movebank_pipeline.duckdb")
        if duckdb_path.exists():
            import duckdb

            con = duckdb.connect(str(duckdb_path), read_only=True)
            try:
                rows = con.execute("SELECT * FROM movebank.studies").fetchall()
                columns = [c[0] for c in con.description]
            finally:
                con.close()
            documents = [row_to_document(dict(zip(columns, r))) for r in rows]

    if not documents:
        fallback = Path("data/movebank_studies.json")
        if fallback.exists():
            raw = json.loads(fallback.read_text())
            documents = [row_to_document(r) for r in raw]

    documents = [d for d in documents if d.get("study_id") and d.get("name")]
    return documents


def build_all(documents: list[dict[str, Any]] | None = None) -> None:
    docs = documents if documents is not None else build_from_pipeline_output()
    if not docs:
        raise RuntimeError(
            "No documents found. Run `python scripts/download_movebank.py` first, "
            "or place a JSON at data/movebank_studies.json."
        )
    print(f"Indexing {len(docs)} study documents...")
    text_index = build_text_index(docs)
    embeddings = build_embeddings(docs)
    save_artifacts(docs, text_index, embeddings)
    print(f"Saved: {DOCUMENTS_PATH}, {INDEX_PATH}, {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    build_all()

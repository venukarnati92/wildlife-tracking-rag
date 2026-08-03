"""Download public Movebank studies to `data/raw/movebank_studies.csv`.

Movebank exposes a public REST endpoint that returns CSV without auth for
studies flagged as publicly viewable::

    https://www.movebank.org/movebank/service/direct-read?entity_type=study

Not every study is fully public; some rows are truncated. This is fine for our
knowledge base since we only need study metadata (name, taxa, PI, dates, ...).

Optional environment variables:
    MOVEBANK_USERNAME / MOVEBANK_PASSWORD -- if set, use basic auth to also
    fetch studies the account can see.

Usage:
    uv run python scripts/download_movebank.py [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

DIRECT_READ = "https://www.movebank.org/movebank/service/direct-read"
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def parse_csv(text: str) -> list[dict]:
    reader = csv.DictReader(io.StringIO(text))
    return [row for row in reader]


def fetch_studies(max_retries: int = 3, backoff_seconds: float = 2.0) -> list[dict]:
    """Fetch and parse the study list, retrying on network errors or truncated responses.

    Movebank's `direct-read` endpoint occasionally returns just the CSV header
    with no data rows (a transient server-side issue), so we treat an
    empty-but-200 response the same as a network failure and retry it too.
    """
    auth = None
    user = os.getenv("MOVEBANK_USERNAME")
    pw = os.getenv("MOVEBANK_PASSWORD")
    if user and pw:
        auth = (user, pw)

    params = {"entity_type": "study"}
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(DIRECT_READ, params=params, auth=auth, timeout=60)
            resp.raise_for_status()
            rows = parse_csv(resp.text)
            if not rows:
                raise ValueError("received an empty study list (likely a truncated response)")
            return rows
        except Exception as exc:  # noqa: BLE001 - broad on purpose, we retry any failure
            last_exc = exc
            if attempt < max_retries:
                wait = backoff_seconds * attempt
                print(
                    f"Attempt {attempt}/{max_retries} failed ({exc}); retrying in {wait:.0f}s...",
                    file=sys.stderr,
                )
                time.sleep(wait)

    assert last_exc is not None
    raise last_exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="If >0, keep only the first N rows")
    parser.add_argument("--out-csv", default=str(DATA_DIR / "movebank_studies.csv"))
    parser.add_argument("--out-json", default="data/movebank_studies.json")
    parser.add_argument("--max-retries", type=int, default=3, help="Retry attempts for transient/truncated responses")
    parser.add_argument("--retry-backoff", type=float, default=2.0, help="Base seconds to wait between retries")
    args = parser.parse_args()

    print(f"Fetching Movebank studies from {DIRECT_READ}...")
    try:
        rows = fetch_studies(max_retries=args.max_retries, backoff_seconds=args.retry_backoff)
    except Exception as exc:
        print(f"Live fetch failed ({exc}). Falling back to bundled sample data if present.", file=sys.stderr)
        seed = Path("data/movebank_studies.sample.json")
        if not seed.exists():
            print("No sample data available. Cannot proceed.", file=sys.stderr)
            return 1
        rows = json.loads(seed.read_text())
    else:
        if args.limit:
            rows = rows[: args.limit]

        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {args.out_csv}")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {len(rows)} rows to {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

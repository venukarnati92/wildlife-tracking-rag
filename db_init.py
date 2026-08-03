"""Postgres schema for conversations + feedback (used by app and dashboard)."""

from __future__ import annotations

import os
from datetime import datetime

import psycopg

DB_TIMEZONE = datetime.now().astimezone().tzinfo


def get_db_connection():
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "wildlife_rag"),
        user=os.getenv("POSTGRES_USER", "wildlife"),
        password=os.getenv("POSTGRES_PASSWORD", "wildlife"),
    )


DDL_CONVERSATIONS = """
CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    strategy TEXT NOT NULL,
    model TEXT NOT NULL,
    instructions TEXT NOT NULL,
    prompt TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    response_time DOUBLE PRECISION NOT NULL,
    cost DOUBLE PRECISION NOT NULL,
    retrieved_ids TEXT,
    timestamp TIMESTAMPTZ NOT NULL
);
"""

DDL_FEEDBACK = """
CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
    source TEXT NOT NULL,
    relevance TEXT,
    explanation TEXT,
    score INTEGER,
    timestamp TIMESTAMPTZ NOT NULL
);
"""


def init_db(drop: bool = False) -> None:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if drop:
                cur.execute("DROP TABLE IF EXISTS feedback")
                cur.execute("DROP TABLE IF EXISTS conversations")
            cur.execute(DDL_CONVERSATIONS)
            cur.execute(DDL_FEEDBACK)
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    print("Database initialized")

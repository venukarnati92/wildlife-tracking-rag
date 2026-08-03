"""Persist a feedback row (user thumbs or LLM judge verdict)."""

from __future__ import annotations

from datetime import datetime

from db_init import DB_TIMEZONE, get_db_connection


def save_feedback(
    conversation_id: int,
    source: str,
    relevance: str | None = None,
    explanation: str | None = None,
    score: int | None = None,
) -> None:
    timestamp = datetime.now(DB_TIMEZONE)
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO feedback (
                    conversation_id, source, relevance, explanation, score, timestamp
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (conversation_id, source, relevance, explanation, score, timestamp),
            )
        conn.commit()
    finally:
        conn.close()

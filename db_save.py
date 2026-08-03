"""Persist a RAG conversation to Postgres."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

from db_init import DB_TIMEZONE, get_db_connection


def save_conversation(
    record,
    question: str,
    strategy: str,
    retrieved_ids: Iterable[str] | None = None,
) -> int:
    timestamp = datetime.now(DB_TIMEZONE)
    retrieved_str = ",".join(str(i) for i in retrieved_ids) if retrieved_ids else None

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations (
                    question, answer, strategy, model, instructions, prompt,
                    prompt_tokens, completion_tokens, total_tokens,
                    response_time, cost, retrieved_ids, timestamp
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    question,
                    record.answer,
                    strategy,
                    record.model,
                    record.instructions,
                    record.prompt,
                    record.prompt_tokens,
                    record.completion_tokens,
                    record.total_tokens,
                    record.response_time,
                    record.cost,
                    retrieved_str,
                    timestamp,
                ),
            )
            conversation_id = cur.fetchone()[0]
        conn.commit()
    finally:
        conn.close()
    return conversation_id

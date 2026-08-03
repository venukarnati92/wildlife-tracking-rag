"""Read queries powering the monitoring dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from db_init import get_db_connection


@dataclass
class Stats:
    total: int
    avg_response_time: float
    total_cost: float
    avg_tokens: float


def _fetch_all(sql: str, params: tuple = ()) -> list[tuple]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    finally:
        conn.close()


def get_stats() -> Stats:
    rows = _fetch_all(
        "SELECT COUNT(*), COALESCE(AVG(response_time), 0), COALESCE(SUM(cost), 0), COALESCE(AVG(total_tokens), 0) FROM conversations"
    )
    row = rows[0] if rows else (0, 0, 0, 0)
    return Stats(total=row[0] or 0, avg_response_time=row[1] or 0.0, total_cost=row[2] or 0.0, avg_tokens=row[3] or 0.0)


def get_conversations(limit: int = 100) -> list[dict[str, Any]]:
    rows = _fetch_all(
        """
        SELECT id, question, answer, strategy, model, prompt_tokens,
               completion_tokens, total_tokens, response_time, cost, timestamp
        FROM conversations
        ORDER BY timestamp DESC
        LIMIT %s
        """,
        (limit,),
    )
    cols = [
        "id",
        "question",
        "answer",
        "strategy",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "response_time",
        "cost",
        "timestamp",
    ]
    return [dict(zip(cols, r)) for r in rows]


def get_relevance_stats() -> dict[str, int]:
    rows = _fetch_all(
        "SELECT relevance, COUNT(*) FROM feedback WHERE source = 'judge' GROUP BY relevance"
    )
    return {r[0] or "UNKNOWN": r[1] for r in rows}


def get_user_feedback_stats() -> tuple[int, int]:
    rows = _fetch_all(
        """
        SELECT
          COALESCE(SUM(CASE WHEN score > 0 THEN 1 ELSE 0 END), 0),
          COALESCE(SUM(CASE WHEN score < 0 THEN 1 ELSE 0 END), 0)
        FROM feedback
        WHERE source = 'user'
        """
    )
    return int(rows[0][0]), int(rows[0][1])


def get_strategy_stats() -> list[dict[str, Any]]:
    rows = _fetch_all(
        """
        SELECT strategy, COUNT(*) AS n, AVG(response_time) AS avg_latency,
               AVG(cost) AS avg_cost, AVG(total_tokens) AS avg_tokens
        FROM conversations
        GROUP BY strategy
        ORDER BY n DESC
        """
    )
    return [
        {
            "strategy": r[0],
            "count": r[1],
            "avg_latency": float(r[2] or 0),
            "avg_cost": float(r[3] or 0),
            "avg_tokens": float(r[4] or 0),
        }
        for r in rows
    ]


def get_conversations_per_day() -> list[dict[str, Any]]:
    rows = _fetch_all(
        """
        SELECT DATE_TRUNC('hour', timestamp) AS bucket, COUNT(*) AS n
        FROM conversations
        GROUP BY bucket
        ORDER BY bucket
        """
    )
    return [{"bucket": r[0], "count": r[1]} for r in rows]


def get_top_questions(limit: int = 10) -> list[dict[str, Any]]:
    rows = _fetch_all(
        """
        SELECT LEFT(question, 80) AS q, COUNT(*) AS n
        FROM conversations
        GROUP BY q
        ORDER BY n DESC
        LIMIT %s
        """,
        (limit,),
    )
    return [{"question": r[0], "count": r[1]} for r in rows]

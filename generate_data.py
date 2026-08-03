"""Populate Postgres with sample conversations for dashboard testing.

Runs the actual RAG assistant on a set of realistic wildlife questions,
logs each answer + judge verdict, and mixes in random thumbs.
"""

from __future__ import annotations

import random
import time

from assistant import create_assistant
from db_feedback import save_feedback
from db_save import save_conversation
from judge import evaluate_relevance


SAMPLE_QUESTIONS = [
    "Which studies track turkey vultures?",
    "Are there any studies on African elephant movement?",
    "What sensors are used to track Galapagos albatrosses?",
    "Which studies focus on migratory storks in Europe?",
    "Which studies track sharks in the Atlantic?",
    "How many individual animals are tracked in the Galapagos albatross study?",
    "Who is the principal investigator for the Serengeti lion project?",
    "Which studies use GPS tags on seabirds?",
    "Is there tracking data for wolves in Yellowstone?",
    "What is the time period covered by the black stork migration study?",
]


def run_once(assistant, strategies):
    question = random.choice(SAMPLE_QUESTIONS)
    strategy = random.choice(strategies)

    answer = assistant.rag(question)
    record = assistant.last_call
    retrieved_ids = [d.get("study_id") for d in (assistant.last_search_results or []) if d.get("study_id")]

    conversation_id = save_conversation(record, question, strategy, retrieved_ids=retrieved_ids)

    try:
        relevance, explanation = evaluate_relevance(question, answer)
        save_feedback(conversation_id, "judge", relevance=relevance, explanation=explanation)
    except Exception:
        pass

    if random.random() < 0.6:
        save_feedback(conversation_id, "user", score=random.choice([1, 1, 1, -1]))


def main():
    strategies = ["hybrid_rerank_rewrite", "hybrid_rerank", "hybrid", "vector", "text"]
    print("Generating sample conversations (Ctrl+C to stop)...")

    for strategy in strategies:
        print(f"\n-- strategy: {strategy}")
        assistant = create_assistant(strategy=strategy)
        for _ in range(3):
            try:
                run_once(assistant, [strategy])
                time.sleep(0.5)
            except Exception as exc:  # noqa: BLE001
                print(f"skip: {exc}")


if __name__ == "__main__":
    main()

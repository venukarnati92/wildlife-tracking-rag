"""LLM-as-judge for RAG answer relevance (used in the app and eval notebooks)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from llmclient import get_llm_client


class RelevanceVerdict(BaseModel):
    relevance: Literal["NON_RELEVANT", "PARTLY_RELEVANT", "RELEVANT"]
    explanation: str


JUDGE_INSTRUCTIONS = """
You are an expert evaluator for a wildlife-tracking RAG system.
Given a question and a generated answer, classify the answer as:

- RELEVANT: the answer directly addresses the question and is consistent with
  the domain (Movebank wildlife tracking studies).
- PARTLY_RELEVANT: the answer partially addresses the question (some correct
  facts but incomplete or contains extra unrelated info).
- NON_RELEVANT: the answer does not address the question or contradicts the
  domain (e.g. hallucinates studies).

Return a short one-sentence explanation.
""".strip()


JUDGE_PROMPT = """
Question: {question}
Generated Answer: {answer}
""".strip()


def evaluate_relevance(question: str, answer: str, client=None, model: str | None = None):
    client = client or get_llm_client()
    model = model or client.model

    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_INSTRUCTIONS},
            {"role": "user", "content": JUDGE_PROMPT.format(question=question, answer=answer)},
        ],
        response_format=RelevanceVerdict,
        temperature=0.0,
    )
    verdict = response.choices[0].message.parsed
    return verdict.relevance, verdict.explanation


if __name__ == "__main__":
    r, e = evaluate_relevance(
        "Which studies track turkey vultures?",
        "The Turkey Vulture Acopian Center study tracks Cathartes aura across the Americas.",
    )
    print(r, "-", e)

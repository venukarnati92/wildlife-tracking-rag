"""RAG variant that records latency, tokens, cost per LLM call."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from rag_helper import RAGBase


PRICING_PER_M_TOKENS = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gcp/gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "llama3.1": {"input": 0.0, "output": 0.0},
    "llama3.2": {"input": 0.0, "output": 0.0},
}


@dataclass
class LLMCallRecord:
    model: str
    prompt: str
    instructions: str
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_time: float
    cost: float
    timestamp: datetime = field(default_factory=datetime.now)


def calculate_cost(model: str, usage: Any) -> float:
    rates = PRICING_PER_M_TOKENS.get(model)
    if rates is None:
        return 0.0
    return (
        usage.prompt_tokens * rates["input"] + usage.completion_tokens * rates["output"]
    ) / 1_000_000


class RAGWithMetrics(RAGBase):
    """RAG that stores the latest `LLMCallRecord` on `self.last_call`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_call: LLMCallRecord | None = None
        self.last_search_results: list[dict[str, Any]] | None = None

    def llm(self, prompt: str) -> str:
        start = time.time()
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.instructions},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        response_time = time.time() - start

        usage = response.usage
        answer = response.choices[0].message.content
        self.last_call = LLMCallRecord(
            model=self.model,
            prompt=prompt,
            instructions=self.instructions,
            answer=answer,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
            response_time=response_time,
            cost=calculate_cost(self.model, usage) if usage else 0.0,
        )
        return answer

    def rag(self, query: str) -> str:
        results = self.search(query)
        self.last_search_results = results
        prompt = self.build_prompt(query, results)
        return self.llm(prompt)

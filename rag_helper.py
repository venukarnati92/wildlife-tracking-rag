"""Core RAG helpers for Wildlife Tracking RAG.

Mirrors the course `rag_helper.py` (RAGBase) but adapted for wildlife
`study` documents from Movebank and supports pluggable searchers
(text, vector, hybrid, reranked) via the `SearchEngine` abstraction.
"""

from __future__ import annotations

from typing import Any, Protocol


INSTRUCTIONS = """
You are a wildlife tracking research assistant. Answer the user's question
using ONLY the study context provided below.

Rules:
- Ground every fact in the provided context. Do NOT invent studies, species,
  investigators, or numbers.
- When helpful, cite the study by name and Movebank study id, e.g.
  "Galapagos Albatrosses (study 2911040)".
- If the context does not contain the answer, reply exactly:
  "I don't know based on the tracked studies I have access to."
- Keep answers concise (2-6 sentences) unless the user asks for detail.
""".strip()


PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT (Movebank tracking studies):
{context}
""".strip()


class SearchEngine(Protocol):
    """Anything with a `search(query, num_results) -> list[dict]` works."""

    def search(self, query: str, num_results: int = 5) -> list[dict[str, Any]]: ...


def format_study_doc(doc: dict[str, Any]) -> str:
    """Render one study doc as a compact block for LLM context."""
    fields = [
        ("Study", doc.get("name")),
        ("Study ID", doc.get("study_id")),
        ("Species / taxa", doc.get("taxa")),
        ("Location", doc.get("location")),
        ("Principal investigator", doc.get("principal_investigator")),
        ("Contact", doc.get("contact")),
        ("Time period", doc.get("time_period")),
        ("Animals tracked", doc.get("number_of_individuals")),
        ("Tags deployed", doc.get("number_of_tags")),
        ("Sensor types", doc.get("sensor_types")),
        ("Study objective", doc.get("study_objective")),
        ("Citation", doc.get("citation")),
        ("License", doc.get("license")),
    ]
    lines = [f"{label}: {value}" for label, value in fields if value not in (None, "", "nan")]
    return "\n".join(lines)


class RAGBase:
    """RAG orchestration: search -> build prompt -> call LLM."""

    def __init__(
        self,
        search_engine: SearchEngine,
        llm_client,
        instructions: str = INSTRUCTIONS,
        prompt_template: str = PROMPT_TEMPLATE,
        model: str | None = None,
        num_results: int = 5,
    ):
        self.search_engine = search_engine
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model or getattr(llm_client, "model", "gpt-4o-mini")
        self.num_results = num_results

    def build_context(self, search_results: list[dict[str, Any]]) -> str:
        blocks = [format_study_doc(doc) for doc in search_results]
        return "\n\n---\n\n".join(blocks).strip()

    def build_prompt(self, query: str, search_results: list[dict[str, Any]]) -> str:
        context = self.build_context(search_results)
        return self.prompt_template.format(question=query, context=context)

    def search(self, query: str, num_results: int | None = None) -> list[dict[str, Any]]:
        return self.search_engine.search(query, num_results=num_results or self.num_results)

    def llm(self, prompt: str) -> str:
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.instructions},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content

    def rag(self, query: str) -> str:
        results = self.search(query)
        prompt = self.build_prompt(query, results)
        return self.llm(prompt)

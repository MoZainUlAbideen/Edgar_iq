"""The orchestrator ties every agent into one pipeline:

  question -> plan queries -> retrieve chunks -> draft answer
           -> check numbers are grounded -> critique claims -> final answer

Each stage is independently testable (see planner.py, retriever.py,
drafter.py, numeric_check.py, critic.py) — this module's only job is
sequencing them and turning their outputs into one AgentAnswer with
citations and any warnings surfaced, rather than silently swallowing them.
"""

from __future__ import annotations

from edgariq.agents.critic import critique_answer, is_approved
from edgariq.agents.drafter import draft_answer
from edgariq.agents.models import AgentAnswer, Citation
from edgariq.agents.numeric_check import find_ungrounded_numbers
from edgariq.agents.planner import plan_search_queries
from edgariq.agents.retriever import retrieve
from edgariq.indexing import OllamaEmbedder, VectorStore
from edgariq.llm import GroqClient


def _build_citations(chunks) -> list[Citation]:
    seen: dict[tuple[str, str], Citation] = {}
    for chunk, _ in chunks:
        citation = Citation(
            form_type=chunk.metadata.get("form_type", "unknown"),
            filing_date=chunk.metadata.get("filing_date", "unknown"),
            source_url=chunk.metadata.get("source_url", ""),
        )
        key = (citation.form_type, citation.filing_date)
        seen[key] = citation  # dedup by filing; last write wins, all equal
    return list(seen.values())


def answer_question(
    question: str,
    store: VectorStore,
    embedder: OllamaEmbedder,
    llm: GroqClient,
) -> AgentAnswer:
    queries = plan_search_queries(llm, question)
    chunks = retrieve(store, embedder, queries)

    if not chunks:
        return AgentAnswer(
            answer="I couldn't find any relevant information in the indexed filings for this question.",
            citations=[],
            warnings=[],
        )

    draft = draft_answer(llm, question, chunks)

    warnings: list[str] = []

    ungrounded = find_ungrounded_numbers(draft, chunks)
    if ungrounded:
        warnings.append(
            "These figures in the answer weren't found verbatim in the retrieved "
            f"filing excerpts, and may be inaccurate: {', '.join(ungrounded)}"
        )

    critique = critique_answer(llm, question, draft, chunks)
    if not is_approved(critique):
        warnings.append(f"The critic flagged possible unsupported claims:\n{critique.strip()}")

    return AgentAnswer(
        answer=draft,
        citations=_build_citations(chunks),
        warnings=warnings,
    )

"""Planner agent: turns a user's question into 1-3 targeted search queries.

A single vague question ("how's NVIDIA doing on data center and what risks
did they flag?") often needs multiple, differently-worded searches to hit
both topics well — one query optimized for "data center revenue" won't
also surface "export control risk factors". This is a thin LLM call, not
a general reasoning step, so it's kept isolated and easy to test with a
canned response.
"""

from __future__ import annotations

from edgariq.llm import GroqClient

_SYSTEM_PROMPT = (
    "You are a research planner for a financial analyst assistant. Given a "
    "user's question about a company's SEC filings, break it into 1 to 3 "
    "focused search queries that would each retrieve relevant passages. "
    "Respond with ONLY the queries, one per line, no numbering, no bullets, "
    "no extra commentary."
)


def plan_search_queries(llm: GroqClient, question: str, max_queries: int = 3) -> list[str]:
    raw = llm.complete(system=_SYSTEM_PROMPT, user=question)

    queries = [
        line.strip("-*0123456789. \t")
        for line in raw.splitlines()
        if line.strip()
    ]
    queries = [q for q in queries if q]  # drop anything that stripped to empty

    return queries[:max_queries] if queries else [question]

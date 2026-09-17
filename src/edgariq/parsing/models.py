"""Data models for the output of the parsing stage.

A ParsedFiling is what gets fed into the RAG index: clean, chunkable text
sections plus the small subset of tables that actually carry data (as
opposed to the hundreds of layout-only tables SEC filing software emits).
"""

from __future__ import annotations

from pydantic import BaseModel


class ParsedTable(BaseModel):
    """A single extracted data table, e.g. a revenue-by-segment breakdown."""

    headers: list[str]
    rows: list[list[str]]
    context: str = ""  # nearby heading/caption text, used to help retrieval

    def to_markdown(self) -> str:
        """Render as a markdown table — the format we feed to the LLM for
        Table QA, since models handle markdown tables far more reliably
        than raw HTML or flattened text."""
        if not self.headers:
            return ""
        lines = [
            "| " + " | ".join(self.headers) + " |",
            "| " + " | ".join(["---"] * len(self.headers)) + " |",
        ]
        for row in self.rows:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)


class ParsedFiling(BaseModel):
    """Clean text + extracted data tables from one filing document."""

    text_sections: list[str]
    tables: list[ParsedTable]

    @property
    def full_text(self) -> str:
        return "\n\n".join(self.text_sections)

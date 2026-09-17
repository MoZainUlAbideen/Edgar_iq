"""Parses raw SEC filing HTML into clean text + genuine data tables.

The hard problem here isn't extracting tables — BeautifulSoup does that in
one line. It's filtering: a real 10-Q has 200-400 <table> elements, and the
vast majority are pure layout (single-cell wrappers, spacer tables, logo
containers) rather than actual financial data. Feeding all of them to an
LLM would bury the 5-10 tables that matter (income statement, segment
revenue, etc.) in noise.

We filter with a simple, inspectable heuristic rather than a model: a table
counts as a "data table" if it has enough rows/cells AND a meaningful
fraction of its cells look numeric (dollar amounts, percentages, plain
numbers). This is deliberately simple — easy to explain in an interview,
easy to tune later against real mis-classifications.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup, Comment, Tag

from edgariq.parsing.models import ParsedFiling, ParsedTable

_NUMERIC_CELL_RE = re.compile(r"^[\$\(\)\-\+\d,\.\%\s]+$")
_MIN_ROWS_FOR_DATA_TABLE = 2
_MIN_CELLS_FOR_DATA_TABLE = 6
_MIN_NUMERIC_RATIO = 0.3
_MIN_PARAGRAPH_LENGTH = 40  # drop short nav/label fragments, keep real prose


def _looks_numeric(cell: str) -> bool:
    cell = cell.strip()
    return bool(cell) and bool(_NUMERIC_CELL_RE.match(cell)) and any(c.isdigit() for c in cell)


def _extract_rows(table: Tag) -> list[list[str]]:
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        # Drop fully-blank rows — common as visual spacers in filing HTML.
        if any(cell for cell in cells):
            rows.append(cells)
    return rows


def _is_data_table(rows: list[list[str]]) -> bool:
    if len(rows) < _MIN_ROWS_FOR_DATA_TABLE:
        return False
    all_cells = [cell for row in rows for cell in row]
    if len(all_cells) < _MIN_CELLS_FOR_DATA_TABLE:
        return False
    numeric_ratio = sum(_looks_numeric(c) for c in all_cells) / len(all_cells)
    return numeric_ratio >= _MIN_NUMERIC_RATIO


def _guess_context(table: Tag, max_len: int = 150) -> str:
    """Grab the nearest preceding non-empty text as a hint of what the
    table is about (e.g. a heading like 'Segment Information')."""
    for node in table.find_all_previous(string=True):
        if isinstance(node, Comment):
            continue  # HTML comments aren't real page content
        text = node.strip()
        if text and len(text) > 3:
            return text[:max_len]
    return ""


def _split_headers_and_data(rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    header, *data_rows = rows
    if any(_looks_numeric(cell) for cell in header):
        # First row already has numbers — it's data, not a header. Fall
        # back to generic column names.
        width = max(len(r) for r in rows)
        return [f"col_{i}" for i in range(width)], rows
    return header, data_rows


def parse_filing_html(html: str) -> ParsedFiling:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "head"]):
        tag.decompose()

    tables: list[ParsedTable] = []
    for table_tag in soup.find_all("table"):
        rows = _extract_rows(table_tag)
        if not _is_data_table(rows):
            table_tag.decompose()
            continue

        headers, data_rows = _split_headers_and_data(rows)
        # Normalize row lengths to match the header width so markdown
        # rendering never breaks on a ragged row.
        width = len(headers)
        normalized_rows = [row[:width] + [""] * (width - len(row)) for row in data_rows]

        tables.append(
            ParsedTable(
                headers=headers,
                rows=normalized_rows,
                context=_guess_context(table_tag),
            )
        )
        table_tag.decompose()

    raw_text = soup.get_text(separator="\n")
    sections = [
        line.strip()
        for line in raw_text.split("\n")
        if len(line.strip()) >= _MIN_PARAGRAPH_LENGTH
    ]

    return ParsedFiling(text_sections=sections, tables=tables)

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
import warnings

from bs4 import BeautifulSoup, Comment, Tag, XMLParsedAsHTMLWarning

from edgariq.parsing.models import ParsedFiling, ParsedTable

# SEC's iXBRL filings often start with an XML declaration even though the
# document is real, renderable XHTML — we're intentionally parsing it as
# HTML (which works fine), so this specific warning is just noise.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

_NUMERIC_CELL_RE = re.compile(r"^[\$\(\)\-\+\d,\.\%\s]+$")
_SYMBOL_ONLY_RE = re.compile(r"^[\$\(\)]+$")  # e.g. a cell that is just "$" or "("
_MIN_ROWS_FOR_DATA_TABLE = 2
_MIN_CELLS_FOR_DATA_TABLE = 6
_MIN_NUMERIC_RATIO = 0.3
_MIN_PARAGRAPH_LENGTH = 40  # drop short nav/label fragments, keep real prose


def _looks_numeric(cell: str) -> bool:
    cell = cell.strip()
    return bool(cell) and bool(_NUMERIC_CELL_RE.match(cell)) and any(c.isdigit() for c in cell)


def _merge_symbol_only_cells(row: list[str]) -> list[str]:
    """SEC table markup often puts a currency symbol or bare parenthesis in
    its own <td>, separate from the actual value (e.g. ["$", "1,234"]
    instead of ["$1,234"]). Left uncorrected, this silently shifts every
    later column over by one and can push real values past the table's
    column count when rows get normalized to header width. Merge each
    symbol-only cell into the next cell so the value stays intact."""
    merged: list[str] = []
    i = 0
    while i < len(row):
        cell = row[i]
        if _SYMBOL_ONLY_RE.match(cell) and i + 1 < len(row):
            merged.append((cell + row[i + 1]).strip())
            i += 2
        else:
            merged.append(cell)
            i += 1
    return merged


def _extract_rows(table: Tag) -> list[list[str]]:
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        # Drop fully-blank rows — common as visual spacers in filing HTML.
        if any(cell for cell in cells):
            rows.append(_merge_symbol_only_cells(cells))
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


def _normalize_row(row: list[str], width: int) -> list[str]:
    """Fit a row to the header's column count. If the row has extra
    (usually empty spacer) cells, drop empties first rather than blindly
    truncating from the end — that's what was silently deleting real
    values in tables with sparse spacer columns."""
    if len(row) > width:
        without_empties = [c for c in row if c != ""]
        row = without_empties if len(without_empties) <= width else row[:width]
    return row + [""] * (width - len(row))


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
        normalized_rows = [_normalize_row(row, width) for row in data_rows]

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
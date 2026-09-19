"""Numeric grounding check ("calculator agent"): verifies every number the
draft states actually appears as a real number in the retrieved excerpts,
rather than trusting the LLM's arithmetic or recall.

This is deliberately a code check, not another LLM call — the whole point
is to catch the LLM being wrong, so asking it to grade its own numbers
would defeat the purpose.

Comparison is done as an exact-match against the SET of numbers actually
extracted from the context — not a substring search over flattened text.
An earlier version of this that used substring matching had a real bug:
after stripping separators, a short number like "9" could coincidentally
appear *inside* an unrelated longer number (e.g. inside "75.292", formed
by two unrelated figures mashed together once punctuation was stripped),
silently passing a hallucinated figure as "grounded". Comparing against a
set of distinct extracted numbers avoids that failure mode entirely.
"""

from __future__ import annotations

import re

from edgariq.indexing import Chunk

# Matches numbers with optional currency/percent/scale decoration, e.g.
# "$75.2 billion", "92%", "1,144". Comma groups must be full triplets
# (",\d{3}") so a stray trailing comma (e.g. "note 7, the company...")
# is never swept into the match.
_NUMBER_RE = re.compile(
    r"\$?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s?(?:billion|million|thousand|%)?",
    re.IGNORECASE,
)


def _normalize(token: str) -> str:
    """Strip currency signs, commas, unit words, and whitespace so
    "$75.2 billion" and "75.2" compare equal, and "1,144" and "1144" do too."""
    token = re.sub(r"(billion|million|thousand)", "", token, flags=re.IGNORECASE)
    return re.sub(r"[^\d.]", "", token)


def _has_decoration(token: str) -> bool:
    return bool(re.search(r"[\$%,.]|billion|million|thousand", token, re.IGNORECASE))


_CALENDAR_YEAR_RE = re.compile(r"^(19|20)\d{2}$")


def _looks_like_a_bare_year(token: str, normalized: str) -> bool:
    """A plain 4-digit number in calendar-year range with no currency/percent/
    scale decoration is almost always a date reference ("in 2024", "fiscal
    2026"), not a financial figure — and dates are already verified via the
    citation list, not this check. Without this, every mention of a year
    gets flagged as "ungrounded" purely because filing prose rarely repeats
    the standalone year string that only lives in our chunk metadata."""
    return not _has_decoration(token) and bool(_CALENDAR_YEAR_RE.match(normalized))


def _extract_numbers(text: str) -> set[str]:
    return {
        _normalize(match.group())
        for match in _NUMBER_RE.finditer(text)
        if any(ch.isdigit() for ch in match.group())
    }


def find_ungrounded_numbers(answer: str, chunks: list[tuple[Chunk, float]]) -> list[str]:
    """Returns the list of number-like tokens in `answer` whose normalized
    value doesn't match any number actually extracted from the retrieved
    context — i.e. numbers the model may have hallucinated."""
    context_text = " ".join(chunk.text for chunk, _ in chunks)
    context_numbers = _extract_numbers(context_text)

    candidates = {
        match.group().strip()
        for match in _NUMBER_RE.finditer(answer)
        if any(ch.isdigit() for ch in match.group())
    }

    ungrounded = []
    for token in candidates:
        normalized = _normalize(token)
        # A bare, undecorated 1-2 digit number (no $, %, comma, decimal, or
        # unit word) is too ambiguous to treat as a financial figure — could
        # be a footnote/list marker. Anything with real decoration is always
        # checked regardless of length, since that's what actually matters.
        if not _has_decoration(token) and len(normalized) <= 2:
            continue
        if _looks_like_a_bare_year(token, normalized):
            continue
        if normalized not in context_numbers:
            ungrounded.append(token)

    return sorted(ungrounded)
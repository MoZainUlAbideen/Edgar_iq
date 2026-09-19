from edgariq.agents.numeric_check import find_ungrounded_numbers
from edgariq.indexing import Chunk


def _chunks(*texts: str):
    return [(Chunk(text=t, chunk_type="text", metadata={}), 0.9) for t in texts]


def test_flags_a_number_not_present_anywhere_in_context():
    chunks = _chunks("Data Center revenue was $75.2 billion, up 92% from a year ago.")
    answer = "Data center revenue grew to $75.2 billion, a 92% increase, driven by $500 million in new orders."

    ungrounded = find_ungrounded_numbers(answer, chunks)

    assert "$500 million" in ungrounded


def test_does_not_flag_numbers_that_are_present():
    chunks = _chunks("Data Center revenue was $75.2 billion, up 92% from a year ago.")
    answer = "Data center revenue was $75.2 billion, up 92% year over year."

    ungrounded = find_ungrounded_numbers(answer, chunks)

    assert ungrounded == []


def test_matches_numbers_despite_formatting_differences():
    chunks = _chunks("Total revenue reached $130,497 million for the year.")
    answer = "Full-year revenue was approximately $130497 million."

    ungrounded = find_ungrounded_numbers(answer, chunks)

    assert ungrounded == []


def test_ignores_bare_undecorated_short_numbers():
    chunks = _chunks("See note 5 for further detail on segment reporting.")
    answer = "As discussed in note 7, the company reports two segments."

    # "7" alone (no $, %, comma, decimal) is too ambiguous to be worth
    # flagging — it's likely a footnote/list marker, not a financial figure.
    ungrounded = find_ungrounded_numbers(answer, chunks)

    assert ungrounded == []


def test_flags_decorated_number_even_if_short():
    chunks = _chunks("Gross margin was 75% for the quarter.")
    answer = "Gross margin came in at 17% for the quarter."

    ungrounded = find_ungrounded_numbers(answer, chunks)

    assert "17%" in ungrounded


def test_does_not_flag_a_bare_calendar_year():
    # Regression test: a real run flagged "2024" as ungrounded even though
    # the answer's underlying claim (112% YoY growth) WAS correctly
    # grounded — the year itself is a date reference, not a financial
    # figure, and years rarely appear standalone in filing prose since the
    # date lives in metadata, not the text.
    chunks = _chunks("Data Center revenue was up 112% from a year ago.")
    answer = "Data center revenue grew 112% year-over-year in 2024."

    ungrounded = find_ungrounded_numbers(answer, chunks)

    assert "2024" not in ungrounded


def test_still_flags_a_decorated_number_that_happens_to_look_year_like():
    # A 4-digit figure with a % or $ is virtually never actually a year —
    # the year exclusion should not swallow a real hallucinated figure that
    # coincidentally falls in the 1900-2099 range.
    chunks = _chunks("Revenue was $1,800 million for the quarter.")
    answer = "Revenue reportedly hit $2024 million for the quarter."

    ungrounded = find_ungrounded_numbers(answer, chunks)

    assert "$2024 million" in ungrounded
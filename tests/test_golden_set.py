from edgariq.evaluation.golden_set import GOLDEN_SET


def test_all_case_ids_are_unique():
    ids = [c.id for c in GOLDEN_SET]
    assert len(ids) == len(set(ids))


def test_every_case_has_a_ticker_and_question():
    for case in GOLDEN_SET:
        assert case.ticker
        assert case.question.strip()


def test_quantitative_cases_have_expected_facts():
    for case in GOLDEN_SET:
        if case.category == "quantitative":
            assert case.expected_facts, f"{case.id} is quantitative but has no expected_facts"


def test_qualitative_cases_have_expected_topics():
    for case in GOLDEN_SET:
        if case.category == "qualitative":
            assert case.expected_topics, f"{case.id} is qualitative but has no expected_topics"


def test_golden_set_has_both_categories_represented():
    categories = {c.category for c in GOLDEN_SET}
    assert categories == {"quantitative", "qualitative"}

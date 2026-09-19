from tests.fakes import FakeLLM

from edgariq.agents.planner import plan_search_queries


def test_splits_multiline_response_into_queries():
    llm = FakeLLM(["NVIDIA data center revenue growth\nNVIDIA export control risk factors"])

    queries = plan_search_queries(llm, "How's data center revenue and export risk?")

    assert queries == [
        "NVIDIA data center revenue growth",
        "NVIDIA export control risk factors",
    ]


def test_strips_numbering_and_bullets():
    llm = FakeLLM(["1. revenue growth\n- risk factors\n* segment breakdown"])

    queries = plan_search_queries(llm, "question")

    assert queries == ["revenue growth", "risk factors", "segment breakdown"]


def test_respects_max_queries_limit():
    llm = FakeLLM(["a\nb\nc\nd\ne"])

    queries = plan_search_queries(llm, "question", max_queries=2)

    assert queries == ["a", "b"]


def test_falls_back_to_original_question_if_llm_returns_nothing_usable():
    llm = FakeLLM(["   \n  "])

    queries = plan_search_queries(llm, "What was the revenue?")

    assert queries == ["What was the revenue?"]

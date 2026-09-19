from tests.fakes import FakeLLM

from edgariq.agents.orchestrator import answer_question
from edgariq.indexing import Chunk, VectorStore


class FakeEmbedder:
    """Every query embeds to the same vector for simplicity — these tests
    care about pipeline wiring, not real semantic search (that's covered
    in test_retriever.py and test_vector_store.py)."""

    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]


def _make_store() -> VectorStore:
    store = VectorStore()
    store.add(
        chunks=[
            Chunk(
                text="Data Center revenue was $75.2 billion, up 92% from a year ago.",
                chunk_type="text",
                metadata={
                    "form_type": "10-Q",
                    "filing_date": "2026-08-26",
                    "source_url": "https://example.com/filing1",
                    "accession_number": "acc-1",
                },
            )
        ],
        vectors=[[1.0, 0.0]],
    )
    return store


def test_full_pipeline_returns_grounded_answer_with_citations_when_approved():
    llm = FakeLLM(
        [
            "data center revenue",  # planner
            "Data center revenue was $75.2 billion, up 92% year over year.",  # drafter
            "APPROVED",  # critic
        ]
    )
    store = _make_store()

    result = answer_question("What was data center revenue?", store, FakeEmbedder(), llm)

    assert "$75.2 billion" in result.answer
    assert result.warnings == []
    assert len(result.citations) == 1
    assert result.citations[0].form_type == "10-Q"
    assert result.citations[0].filing_date == "2026-08-26"


def test_pipeline_surfaces_warning_when_critic_flags_unsupported_claim():
    llm = FakeLLM(
        [
            "data center revenue",  # planner
            "Data center revenue was $75.2 billion, up 92%, driven by $9 billion in new deals.",  # drafter (hallucinated figure)
            "UNSUPPORTED: the $9 billion new-deals figure is not in the excerpts",  # critic
        ]
    )
    store = _make_store()

    result = answer_question("What was data center revenue?", store, FakeEmbedder(), llm)

    assert len(result.warnings) == 2  # both numeric check and critic should flag it
    assert any("9 billion" in w for w in result.warnings)
    assert any("critic flagged" in w.lower() for w in result.warnings)


def test_pipeline_returns_no_information_message_when_nothing_retrieved():
    llm = FakeLLM(["some query"])  # only the planner should get called
    empty_store = VectorStore()

    result = answer_question("Unrelated question", empty_store, FakeEmbedder(), llm)

    assert "couldn't find" in result.answer.lower()
    assert result.citations == []
    assert len(llm.calls) == 1  # drafter/critic should never have been called

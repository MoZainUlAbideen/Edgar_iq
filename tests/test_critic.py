from tests.fakes import FakeLLM

from edgariq.agents.critic import critique_answer, is_approved
from edgariq.indexing import Chunk


def _chunks(*texts: str):
    return [(Chunk(text=t, chunk_type="text", metadata={}), 0.9) for t in texts]


def test_is_approved_true_for_exact_approved_response():
    assert is_approved("APPROVED") is True
    assert is_approved("  approved  \n") is True  # tolerant of whitespace/case


def test_is_approved_false_for_anything_else():
    assert is_approved("UNSUPPORTED: the 92% figure is not in the excerpts") is False
    assert is_approved("") is False


def test_critique_answer_passes_question_draft_and_context_to_llm():
    llm = FakeLLM(["APPROVED"])
    chunks = _chunks("Revenue was $75.2 billion.")

    result = critique_answer(llm, "What was revenue?", "Revenue was $75.2 billion.", chunks)

    assert result == "APPROVED"
    system, user = llm.calls[0]
    assert "What was revenue?" in user
    assert "$75.2 billion" in user


def test_critic_sees_source_labels_not_just_bare_text():
    # Regression test: a real run showed the critic flagging correct filing
    # dates as "unsupported by the excerpts" — because an earlier version
    # built the critic's context from bare chunk.text only, never including
    # the [Source: ... filed ...] labels the drafter was given. The critic
    # can only fairly judge citations it was actually shown.
    llm = FakeLLM(["APPROVED"])
    chunk = Chunk(
        text="Revenue was $75.2 billion.",
        chunk_type="text",
        metadata={"form_type": "10-Q", "filing_date": "2026-08-26"},
    )

    critique_answer(llm, "What was revenue?", "draft", [(chunk, 0.9)])

    _, user = llm.calls[0]
    assert "10-Q" in user
    assert "2026-08-26" in user
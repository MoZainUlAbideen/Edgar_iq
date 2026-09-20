from edgariq.agents.retriever import retrieve
from edgariq.indexing import Chunk, VectorStore


class FakeEmbedder:
    """Maps specific query strings to specific vectors so we can control
    exactly which chunks each query should retrieve."""

    def __init__(self, vector_by_query: dict[str, list[float]]):
        self._vector_by_query = vector_by_query

    def embed(self, text: str) -> list[float]:
        return self._vector_by_query[text]


def _chunk(text: str, accession: str) -> Chunk:
    return Chunk(text=text, chunk_type="text", metadata={"accession_number": accession})


def test_merges_results_across_queries_without_duplicates():
    store = VectorStore()
    store.add(
        chunks=[_chunk("revenue info", "acc-1"), _chunk("risk info", "acc-2")],
        vectors=[[1.0, 0.0], [0.0, 1.0]],
    )
    embedder = FakeEmbedder({"revenue query": [1.0, 0.0], "risk query": [0.0, 1.0]})

    results = retrieve(store, embedder, queries=["revenue query", "risk query"], top_k_per_query=2)

    texts = {chunk.text for chunk, _ in results}
    assert texts == {"revenue info", "risk info"}
    assert len(results) == 2  # no duplicate entries despite two queries


def test_keeps_the_higher_score_when_same_chunk_matches_multiple_queries():
    store = VectorStore()
    store.add(chunks=[_chunk("revenue info", "acc-1")], vectors=[[1.0, 0.0]])
    embedder = FakeEmbedder({"query a": [1.0, 0.0], "query b": [0.9, 0.1]})

    results = retrieve(store, embedder, queries=["query a", "query b"])

    assert len(results) == 1
    # query a is an exact match (score 1.0), should win over query b's partial match
    assert results[0][1] == 1.0


def test_respects_max_total_across_all_queries_combined():
    store = VectorStore()
    chunks = [_chunk(f"chunk {i}", f"acc-{i}") for i in range(10)]
    vectors = [[1.0, float(i) * 0.01] for i in range(10)]
    store.add(chunks, vectors)
    embedder = FakeEmbedder({"q": [1.0, 0.0]})

    results = retrieve(store, embedder, queries=["q"], top_k_per_query=10, max_total=3)

    assert len(results) == 3


def test_extract_filing_date_hint_finds_iso_date_in_question():
    from edgariq.agents.retriever import extract_filing_date_hint

    assert extract_filing_date_hint("What was revenue in the 10-Q filed 2026-05-20?") == "2026-05-20"


def test_extract_filing_date_hint_returns_none_when_no_date_present():
    from edgariq.agents.retriever import extract_filing_date_hint

    assert extract_filing_date_hint("How did data center revenue grow year over year?") is None


def test_original_question_with_date_restricts_retrieval_to_that_filing():
    # Regression test for a real eval-harness finding: two filings can
    # describe near-identical growth in near-identical language, so pure
    # semantic similarity alone can pick the WRONG quarter. When the
    # question names a specific filing date, retrieval should use it as a
    # hard filter rather than trusting embedding similarity to sort it out.
    store = VectorStore()
    right_quarter = _chunk("Data Center revenue was $75.2 billion, up 92%.", "acc-right")
    right_quarter_meta = Chunk(
        text=right_quarter.text, chunk_type="text",
        metadata={"accession_number": "acc-right", "filing_date": "2026-05-20"},
    )
    wrong_quarter_meta = Chunk(
        text="Data Center revenue was $39.1 billion, up 73%.", chunk_type="text",
        metadata={"accession_number": "acc-wrong", "filing_date": "2025-05-28"},
    )
    store.add(
        chunks=[right_quarter_meta, wrong_quarter_meta],
        vectors=[[0.9, 0.1], [1.0, 0.0]],  # the WRONG quarter scores higher semantically
    )
    embedder = FakeEmbedder({"data center revenue": [1.0, 0.0]})

    results = retrieve(
        store, embedder, queries=["data center revenue"],
        original_question="What was data center revenue in the 10-Q filed 2026-05-20?",
    )

    assert len(results) == 1
    assert results[0][0].metadata["filing_date"] == "2026-05-20"


def test_date_filter_looks_deeper_to_rescue_a_lower_ranked_but_correct_chunk():
    # Regression test for a real diagnosed failure: within a single
    # filing's chunks, the one containing the actual figure ("up 112% from
    # a year ago") ranked 6th — a narrative chunk using similar keywords
    # scored higher despite not containing the number at all. A shallow
    # top-k (fine for an unfiltered, thousands-of-chunks search) silently
    # cut off the correct answer. Once a date filter has already narrowed
    # the pool this much, retrieval should look deeper rather than apply
    # the same shallow cutoff.
    store = VectorStore()
    narrative_chunks = [
        Chunk(
            text=f"Revenue growth was driven by data center demand, narrative {i}.",
            chunk_type="text",
            metadata={"accession_number": f"acc-narrative-{i}", "filing_date": "2024-11-20"},
        )
        for i in range(5)
    ]
    figure_chunk = Chunk(
        text="Data Center revenue was up 112% from a year ago.",
        chunk_type="text",
        metadata={"accession_number": "acc-figure", "filing_date": "2024-11-20"},
    )
    store.add(
        chunks=[*narrative_chunks, figure_chunk],
        # Every narrative chunk scores higher than the figure chunk —
        # exactly the real, diagnosed ranking behavior.
        vectors=[[0.99 - i * 0.01, 0.1] for i in range(5)] + [[0.90, 0.1]],
    )
    embedder = FakeEmbedder({"data center revenue growth": [1.0, 0.0]})

    # With the OLD shallow default (top_k_per_query=4), the figure chunk
    # (ranked 6th) would never appear. The date filter should widen the
    # search enough to include it anyway.
    results = retrieve(
        store, embedder, queries=["data center revenue growth"],
        top_k_per_query=4, max_total=8,
        original_question="What was growth in the 10-Q filed 2024-11-20?",
    )

    assert any("112%" in chunk.text for chunk, _ in results)
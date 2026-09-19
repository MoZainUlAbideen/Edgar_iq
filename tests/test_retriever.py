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

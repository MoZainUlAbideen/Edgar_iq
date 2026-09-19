import tempfile
from pathlib import Path

from edgariq.indexing.models import Chunk
from edgariq.indexing.vector_store import VectorStore


def _chunk(text: str) -> Chunk:
    return Chunk(text=text, chunk_type="text", metadata={"source": "test"})


def test_search_ranks_most_similar_vector_first():
    store = VectorStore()
    store.add(
        chunks=[_chunk("about cats"), _chunk("about dogs"), _chunk("about finance")],
        vectors=[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
    )

    results = store.search(query_vector=[1.0, 0.0], top_k=2)

    assert len(results) == 2
    assert results[0][0].text == "about cats"  # exact match should rank first
    assert results[0][1] > results[1][1]  # scores should be in descending order


def test_search_on_empty_store_returns_empty_list():
    store = VectorStore()
    assert store.search(query_vector=[1.0, 0.0]) == []


def test_search_respects_top_k_even_when_more_chunks_exist():
    store = VectorStore()
    store.add(
        chunks=[_chunk("a"), _chunk("b"), _chunk("c")],
        vectors=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    )

    results = store.search(query_vector=[1.0, 0.0], top_k=1)

    assert len(results) == 1


def test_add_rejects_mismatched_chunk_and_vector_counts():
    store = VectorStore()
    try:
        store.add(chunks=[_chunk("a"), _chunk("b")], vectors=[[1.0, 0.0]])
        assert False, "should have raised"
    except ValueError:
        pass


def test_save_and_load_round_trip_preserves_search_results():
    store = VectorStore()
    store.add(
        chunks=[_chunk("about cats"), _chunk("about finance")],
        vectors=[[1.0, 0.0], [0.0, 1.0]],
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "index.json"
        store.save(path)
        loaded = VectorStore.load(path)

    assert len(loaded) == 2
    results = loaded.search(query_vector=[1.0, 0.0], top_k=1)
    assert results[0][0].text == "about cats"


def test_len_reflects_number_of_stored_chunks():
    store = VectorStore()
    assert len(store) == 0
    store.add(chunks=[_chunk("a")], vectors=[[1.0]])
    assert len(store) == 1

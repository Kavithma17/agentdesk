import uuid

import chromadb
import pytest

from app.rag.chunking import Chunk
from app.rag.vector_store import ChromaVectorStore, EmbedderMismatchError

EMBEDDER = "fake/model@3"


def make_chunk(chunk_id: str, source: str = "faq.md", text: str | None = None) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text or f"text of {chunk_id}",
        source=source,
        section="FAQ",
        chunk_index=0,
        token_count=5,
        doc_hash="hash",
    )


@pytest.fixture
def client(tmp_path):
    return chromadb.PersistentClient(path=str(tmp_path))


@pytest.fixture
def store(client):
    return ChromaVectorStore(client, f"test-{uuid.uuid4().hex}", EMBEDDER, dim=3)


def test_search_returns_closest_chunks_first(store):
    store.upsert(
        [make_chunk("a"), make_chunk("b"), make_chunk("c")],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.7, 0.7, 0.0]],
    )

    results = store.search([1.0, 0.0, 0.0], k=2)

    assert [r.chunk_id for r in results] == ["a", "c"]
    assert results[0].score == pytest.approx(1.0, abs=1e-3)
    assert results[0].text == "text of a"
    assert results[0].source == "faq.md"
    assert results[0].section == "FAQ"


def test_upsert_overwrites_existing_id(store):
    store.upsert([make_chunk("a", text="old")], [[1.0, 0.0, 0.0]])
    store.upsert([make_chunk("a", text="new")], [[1.0, 0.0, 0.0]])

    assert store.count() == 1
    assert store.search([1.0, 0.0, 0.0], k=1)[0].text == "new"


def test_ids_for_source_and_delete(store):
    store.upsert(
        [make_chunk("a", "faq.md"), make_chunk("b", "faq.md"), make_chunk("c", "returns.md")],
        [[1.0, 0.0, 0.0]] * 3,
    )

    assert store.ids_for_source("faq.md") == {"a", "b"}
    store.delete(["a"])
    assert store.ids_for_source("faq.md") == {"b"}
    assert store.ids_for_source("missing.md") == set()


def test_delete_nothing_is_a_no_op(store):
    store.delete([])
    assert store.count() == 0


def test_asking_for_more_results_than_stored(store):
    store.upsert([make_chunk("a")], [[1.0, 0.0, 0.0]])
    assert len(store.search([1.0, 0.0, 0.0], k=10)) == 1


def test_data_survives_reopening(tmp_path):
    first = ChromaVectorStore.from_path(tmp_path, "kb-test", EMBEDDER, dim=3)
    first.upsert([make_chunk("a")], [[1.0, 0.0, 0.0]])

    reopened = ChromaVectorStore.from_path(tmp_path, "kb-test", EMBEDDER, dim=3)

    assert reopened.count() == 1


def test_refuses_collection_built_by_another_embedder(client):
    ChromaVectorStore(client, "kb-test", EMBEDDER, dim=3)

    with pytest.raises(EmbedderMismatchError, match="fake/model@3"):
        ChromaVectorStore(client, "kb-test", "other/model@3", dim=3)


def test_upsert_rejects_mismatched_input(store):
    with pytest.raises(ValueError, match="chunks but"):
        store.upsert([make_chunk("a")], [])
    with pytest.raises(ValueError, match="dimensions"):
        store.upsert([make_chunk("a")], [[1.0, 0.0]])


def test_search_rejects_bad_input(store):
    with pytest.raises(ValueError):
        store.search([1.0, 0.0, 0.0], k=0)
    with pytest.raises(ValueError, match="dimensions"):
        store.search([1.0, 0.0], k=1)
